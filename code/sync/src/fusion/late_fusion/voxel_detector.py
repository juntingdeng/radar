"""Small PointPillars-style BEV detector, SHARED by lidar and radar (different input channels, same architecture).

points (N, C: first two cols = fwd, lat) -> pillar-ise (scatter-max of a tiny PointNet) -> BEV pseudo-image ->
small 2D CNN -> center head (per-cell objectness logit + box regression dx,dy,w,l). Anchor-free / center-based.
Kept deliberately tiny (feat 32, 2 conv blocks) -- the point is the complete prediction path; swap in a bigger
backbone later with no interface change. decode() turns the dense head into BEV boxes + scores (topk + NMS)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import boxes as BX

# BEV grid shared by both detectors (metres): forward [0,32), lateral [-16,16), 0.5 m cells -> 64 x 64
GRID = (0.0, 32.0, -16.0, 16.0); CELL = 0.5

class VoxelDetector(nn.Module):
    def __init__(self, in_ch, feat=32, grid=GRID, cell=CELL):
        super().__init__()
        self.fmin, self.fmax, self.lmin, self.lmax = grid; self.cell = cell
        self.H = int(round((self.fmax - self.fmin) / cell)); self.W = int(round((self.lmax - self.lmin) / cell))
        self.pointnet = nn.Sequential(nn.Linear(in_ch + 2, feat), nn.ReLU(), nn.Linear(feat, feat), nn.ReLU())
        self.backbone = nn.Sequential(
            nn.Conv2d(feat, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(),
            nn.Conv2d(feat, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(),
            nn.Conv2d(feat, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU())
        self.head = nn.Conv2d(feat, 1 + 4, 1)                    # objectness logit + (dx, dy, w, l)
        self.feat = feat

    def voxelize(self, pts):
        """pts: (N, in_ch) tensor, cols 0,1 = fwd,lat. -> (feat, H, W) BEV pseudo-image via scatter-max."""
        dev = pts.device; H, W, feat = self.H, self.W, self.feat
        fwd, lat = pts[:, 0], pts[:, 1]
        m = (fwd >= self.fmin) & (fwd < self.fmax) & (lat >= self.lmin) & (lat < self.lmax)
        bev = torch.zeros(H * W, feat, device=dev)
        if m.any():
            p = pts[m]; fwd, lat = p[:, 0], p[:, 1]
            px = ((fwd - self.fmin) / self.cell).long().clamp(0, H - 1)
            py = ((lat - self.lmin) / self.cell).long().clamp(0, W - 1)
            pcx = self.fmin + (px.float() + 0.5) * self.cell; pcy = self.lmin + (py.float() + 0.5) * self.cell
            aug = torch.cat([p, (fwd - pcx)[:, None], (lat - pcy)[:, None]], 1)    # + offset-to-pillar-centre
            pf = self.pointnet(aug)                                               # (M, feat)
            flat = (px * W + py)[:, None].expand(-1, feat)
            bev = bev.scatter_reduce(0, flat, pf, reduce="amax", include_self=True)
        return bev.view(H, W, feat).permute(2, 0, 1)

    def forward(self, pts):
        bev = self.voxelize(pts).unsqueeze(0)       # (1, feat, H, W)
        out = self.head(self.backbone(bev))[0]      # (5, H, W)
        return out                                   # [0]=obj logit, [1:5]=dx,dy,w,l

    def cell_centers(self):
        fs = self.fmin + (torch.arange(self.H) + 0.5) * self.cell
        ls = self.lmin + (torch.arange(self.W) + 0.5) * self.cell
        return fs, ls

@torch.no_grad()
def decode(model, out, score_thr=0.3, topk=20, nms_thr=0.2):
    """Dense head -> BEV boxes [fmin,lmin,fmax,lmax] + scores. dx,dy are centre offsets within the cell (in m);
    w,l are box extents (m) via softplus."""
    obj = torch.sigmoid(out[0]); reg = out[1:]
    H, W = obj.shape; fs, ls = model.cell_centers()
    flat = obj.flatten(); k = min(topk, flat.numel())
    vals, idx = flat.topk(k); keep = vals > score_thr
    idx = idx[keep]; vals = vals[keep]
    if len(idx) == 0: return np.zeros((0, 4)), np.zeros((0,))
    yy = (idx // W); xx = (idx % W)
    cx = fs[yy] + reg[0][yy, xx]; cy = ls[xx] + reg[1][yy, xx]
    w = reg[2][yy, xx].clamp(0.5, 20.0); l = reg[3][yy, xx].clamp(0.5, 20.0)     # forward-extent, lateral-extent (m)
    bx = torch.stack([cx - w / 2, cy - l / 2, cx + w / 2, cy + l / 2], 1).cpu().numpy()
    sc = vals.cpu().numpy()
    kp = BX.nms(bx, sc, nms_thr)
    return bx[kp], sc[kp]

if __name__ == "__main__":
    import data_l2
    fr = data_l2.load_frames(("night_clean",))[:2]
    lidar = VoxelDetector(in_ch=4); radar = VoxelDetector(in_ch=3)
    for f in fr:
        lp = torch.tensor(f["lidar_pts"]).float()      # x,y,z,intensity -> but cols 0,1 must be fwd,lat
        # lidar cache is (x=fwd, y=lat, z, intensity) already in the lidar frame; confirm by ranges
        print(f"{f['fid']}: lidar pts {lp.shape} fwd[{lp[:,0].min():.0f},{lp[:,0].max():.0f}] lat[{lp[:,1].min():.0f},{lp[:,1].max():.0f}]")
        out = lidar(lp); b, s = decode(lidar, out)
        rp = torch.tensor(f["radar_pts"]).float(); rout = radar(rp); rb, rs = decode(radar, rout)
        print(f"   lidar head {tuple(out.shape)} -> {len(b)} boxes ; radar head {tuple(rout.shape)} -> {len(rb)} boxes (untrained)")
