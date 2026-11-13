"""BEV cross-attention fusion prototype (merged attn_*). Subcommands: proto | rasterize | real"""
import sys

# ======== src/fusion/attn_fusion_proto.py ========
import torch, torch.nn as nn, torch.nn.functional as F
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
class GateThreshold(nn.Module):
    """BASELINE (no learning): a fixed hand rule on reliability feats [lidar_density, radar_power, range].
    Open radar where lidar is sparse or far. No parameters -> isolates whether LEARNING the gate helps."""
    def forward(self, relfeat):
        ld, rng = relfeat[:, 0:1], relfeat[:, 2:3]
        return torch.sigmoid(6.0 * (0.10 - ld) + 2.0 * (rng - 0.6))
class GateShallow(nn.Module):
    """learned but SHALLOW: 1x1 conv (per-cell, no context) over hand-crafted relfeat = learned soft threshold."""
    def __init__(self, fin, hid=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(fin, hid, 1), nn.ReLU(), nn.Conv2d(hid, 1, 1))

    def forward(self, relfeat):
        return torch.sigmoid(self.net(relfeat))
class GateFeature(nn.Module):
    """LEARNED POLICY: gate from the ENCODER FEATURES (lidar+radar) with 3x3 spatial context.
    Learns what 'unreliable' looks like from data + neighborhood, not from 3 hand-picked scalars."""
    def __init__(self, c, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * c, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, 1, 1))

    def forward(self, l_feat, r_feat):
        return torch.sigmoid(self.net(torch.cat([l_feat, r_feat], 1)))
class GatedCrossAttn(nn.Module):
    """Soft per-cell modality SELECTION + attention refinement.
      base = (1-g)*lidar + g*radar         # direct gated path: radar reaches the head where lidar is blind
      query = lidar+radar (so blind cells still query via radar); keys/values = radar
      fused = base + g * crossattn(query, radar)
    """
    def __init__(self, c, heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm = nn.LayerNorm(c)

    def forward(self, l, r, gate):
        B, C, H, W = l.shape
        q = (l + r).flatten(2).transpose(1, 2)         # B, HW, C  (radar-aware query)
        rt = r.flatten(2).transpose(1, 2)
        att, _ = self.mha(self.norm(q), self.norm(rt), rt)
        att = att.transpose(1, 2).reshape(B, C, H, W)
        base = (1 - gate) * l + gate * r               # convex gate = soft modality selection
        return base + gate * att                        # attention refines, opened by the gate
class BEVAttnFusionNet(nn.Module):
    def __init__(self, c_l=2, c_r=3, c=32, relf=3, gate_kind="feature"):
        super().__init__()
        self.enc_l = nn.Sequential(nn.Conv2d(c_l, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU())
        self.enc_r = nn.Sequential(nn.Conv2d(c_r, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU())
        self.gate_kind = gate_kind
        self.gate = {"threshold": GateThreshold, "shallow": lambda: GateShallow(relf),
                     "feature": lambda: GateFeature(c)}[gate_kind]() if gate_kind != "threshold" else GateThreshold()
        self.fuse = GatedCrossAttn(c)
        self.head = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, 1, 1))

    def forward(self, lidar_bev, radar_bev, relfeat):
        l = self.enc_l(lidar_bev); r = self.enc_r(radar_bev)
        g = self.gate(l, r) if self.gate_kind == "feature" else self.gate(relfeat)
        f = self.fuse(l, r, g)
        return self.head(f), g                         # heatmap logits, gate map
def toy_batch(B, H, W, n_obj=5, device=DEV, gen=None):
    lidar = torch.zeros(B, 2, H, W); radar = torch.zeros(B, 3, H, W); tgt = torch.zeros(B, 1, H, W)
    dropout = torch.zeros(B, 1, H, W)                  # where lidar is blind (bookkeeping only)
    ri = torch.randint
    for b in range(B):
        # lidar-dropout box (the glare/far-range analog): lidar sees nothing inside it
        y0, x0 = int(ri(0, H // 2, (1,), generator=gen)), int(ri(0, W // 2, (1,), generator=gen))
        y1, x1 = y0 + H // 3, x0 + W // 3
        dropout[b, 0, y0:y1, x0:x1] = 1
        for _ in range(n_obj):
            oy, ox = int(ri(2, H - 2, (1,), generator=gen)), int(ri(2, W - 2, (1,), generator=gen))
            tgt[b, 0, oy, ox] = 1
            # radar ALWAYS sees the object but with COARSE localization (jitter +-2 cells = angular res)
            jy, jx = int(ri(-2, 3, (1,), generator=gen)), int(ri(-2, 3, (1,), generator=gen))
            ry, rx = min(max(oy + jy, 0), H - 1), min(max(ox + jx, 0), W - 1)
            radar[b, 0, ry, rx] = 1; radar[b, 2, ry, rx] = 1 + torch.rand(1, generator=gen).item()
            if not dropout[b, 0, oy, ox]:                # lidar sees it (EXACT) only outside dropout
                lidar[b, 0, oy, ox] = 1; lidar[b, 1, oy, ox] = 1
    tgt = F.max_pool2d(tgt, 3, 1, 1)                     # fatten targets to 3x3 blobs
    lidar += 0.02 * torch.randn_like(lidar); radar += 0.02 * torch.randn_like(radar)
    # reliability feats: lidar density, radar power, normalized range (dist from bottom-center ego)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    rng = torch.hypot(yy, xx)[None, None].expand(B, 1, H, W)
    ldens = F.avg_pool2d(lidar[:, :1], 5, 1, 2)
    relf = torch.cat([ldens, radar[:, 2:3], rng], 1)
    return (lidar.to(device), radar.to(device), relf.to(device), tgt.to(device), dropout.to(device))
def run_gate(gate_kind, H=48, W=48, steps=400):
    torch.manual_seed(0)                                   # same init-seed & data stream per variant
    net = BEVAttnFusionNet(gate_kind=gate_kind).to(DEV)
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if any(p.requires_grad for p in net.parameters()):
        opt = torch.optim.Adam(net.parameters(), 2e-3)
    for step in range(steps):
        lidar, radar, relf, tgt, drop = toy_batch(16, H, W)
        hm, g = net(lidar, radar, relf)
        loss = F.binary_cross_entropy_with_logits(hm, tgt, pos_weight=torch.tensor(20.0, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():                                  # fixed eval set
        g_ = torch.Generator().manual_seed(123)
        lidar, radar, relf, tgt, drop = toy_batch(64, H, W, gen=g_)
        hm, g = net(lidar, radar, relf)
        pred = (hm.sigmoid() > 0.5).float(); obj = (tgt > 0.5)
        rec_in = ((pred.bool() & obj & drop.bool()).sum() / (obj & drop.bool()).sum().clamp(min=1)).item()
        rec_out = ((pred.bool() & obj & ~drop.bool()).sum() / (obj & ~drop.bool()).sum().clamp(min=1)).item()
        g_in = ((g * drop).sum() / drop.sum()).item(); g_out = ((g * (1 - drop)).sum() / (1 - drop).sum()).item()
    return dict(params=params, loss=loss.item(), rec_in=rec_in, rec_out=rec_out, g_in=g_in, g_out=g_out)
def demo():
    print(f"device={DEV}   toy: lidar zeroed inside a random dropout box (glare/range analog); radar coarse\n")
    print(f"{'gate':10s} | {'params':>7s} | {'recall-in':>9s} | {'recall-out':>10s} | {'gate sel (in-out)':>17s}")
    print("-" * 70)
    for kind in ["threshold", "shallow", "feature"]:
        m = run_gate(kind)
        sel = m["g_in"] - m["g_out"]
        print(f"{kind:10s} | {m['params']:7d} | {m['rec_in']:9.2f} | {m['rec_out']:10.2f} | "
              f"{m['g_in']:.3f}-{m['g_out']:.3f}={sel:+.3f}")
    print("\nrecall-in = objects detected in the lidar-BLIND region (radar must rescue them).")
    print("gate sel  = gate[in-dropout] - gate[outside]; >0 means it leans on radar where lidar is blind.")
    print("threshold = non-learnable baseline; feature = learned gate from encoder features + 3x3 context.")

def proto_main():
    demo()

# ======== src/fusion/attn_rasterize.py ========
import os, sys, json
import numpy as np
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FWD, LAT, RES = 45.0, 24.0, 1.5
H, W = int(FWD / RES), int(2 * LAT / RES)                 # 30 x 32
def _cell(x_fwd, y_lat):
    r = np.clip((x_fwd / RES).astype(int), 0, H - 1)
    c = np.clip(((y_lat + LAT) / RES).astype(int), 0, W - 1)
    return r, c
def rasterize_lidar(pts):
    g = np.zeros((3, H, W), np.float32)
    m = (pts[:, 0] >= 0) & (pts[:, 0] < FWD) & (np.abs(pts[:, 1]) < LAT)
    p = pts[m]
    if len(p):
        r, c = _cell(p[:, 0], p[:, 1])
        np.add.at(g[1], (r, c), 1.0)                      # density (count)
        np.maximum.at(g[2], (r, c), np.clip((p[:, 2] + 0.5) / 2.5, 0, 1))  # max height
        g[0, r, c] = 1.0                                   # occupancy
    g[1] = np.clip(g[1] / 8.0, 0, 1)
    return g
def rasterize_radar(xy, vel, power):
    g = np.zeros((3, H, W), np.float32)
    m = (xy[:, 0] >= 0) & (xy[:, 0] < FWD) & (np.abs(xy[:, 1]) < LAT)
    xy, vel, power = xy[m], vel[m], power[m]
    if len(xy):
        r, c = _cell(xy[:, 0], xy[:, 1])
        g[0, r, c] = 1.0
        g[1, r, c] = np.clip(vel / 5.0, -1, 1)             # doppler (normalized ~max_doppler)
        pn = (power - power.min()) / (np.ptp(power) + 1e-9)
        g[2, r, c] = pn
    return g
def relfeat(lidar_bev, radar_bev):
    import torch.nn.functional as F
    ld = F.avg_pool2d(torch.tensor(lidar_bev[1:2])[None], 5, 1, 2)[0].numpy()
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(-1, 1, W), indexing="ij")
    rng = np.hypot(yy, xx)[None].astype(np.float32)
    return np.concatenate([ld, radar_bev[2:3], rng], 0)
def rasterize_target(boxes):
    import torch.nn.functional as F
    g = np.zeros((1, H, W), np.float32)
    for b in boxes:
        if b.get("label") != "car":
            continue
        xf = 0.5 * (b["forward"][0] + b["forward"][1]); yl = 0.5 * (b["lateral"][0] + b["lateral"][1])
        if not (0 <= xf < FWD and abs(yl) < LAT):
            continue
        r, c = _cell(np.array([xf]), np.array([yl]))
        g[0, r[0], c[0]] = 1.0
    return F.max_pool2d(torch.tensor(g)[None], 3, 1, 1)[0].numpy()
def build_cache(split, frames, out):
    from dataset import Benchmark
    from detector import RadarDetector
    ds = Benchmark(os.path.join(REPO, "sync/res/benchmark_v1"), split)
    by_id = {s["sample_id"]: s for s in ds.samples}
    L, Rr, Rf, T = [], [], [], []
    for n, f in enumerate(frames):
        sid = f"0511_{f:06d}"
        if sid not in by_id:
            continue
        s = ds[[i for i, x in enumerate(ds.samples) if x["sample_id"] == sid][0]]
        R2L = np.array(s["calib"]["radar_to_lidar_auto"])
        pts = ds.lidar(s)
        d = ds.radar(s); Xd, Yd = RadarDetector.to_xy(d)
        rf = R2L[0, 0] * Xd + R2L[0, 1] * Yd + R2L[0, 3]; rl = R2L[1, 0] * Xd + R2L[1, 1] * Yd + R2L[1, 3]
        lb = rasterize_lidar(pts)
        rb = rasterize_radar(np.c_[rf, rl], np.asarray(d["vel"], float), np.asarray(d["power"], float))
        L.append(lb); Rr.append(rb); Rf.append(relfeat(lb, rb)); T.append(rasterize_target(s["boxes"]))
        print(f"  {n+1}/{len(frames)} {sid}: lidar_occ {lb[0].sum():.0f}  radar_occ {rb[0].sum():.0f}  gt_cars {int(T[-1].sum()>0)}", flush=True)
    torch.save({k: torch.tensor(np.stack(v)) for k, v in [("lidar", L), ("radar", Rr), ("relf", Rf), ("target", T)]}, out)
    print(f"wrote {out}  ({len(L)} frames, grid {H}x{W})")

def rasterize_main():
    sys.path.insert(0, os.path.join(REPO, "sync/res/benchmark_v1"))
    sys.path.insert(0, os.path.join(REPO, "sync/src/fusion/radar_first"))
    split = sys.argv[1]; frames = [int(x) for x in sys.argv[2:]]
    build_cache(split, frames, os.path.join(REPO, "sync/res/benchmark_v1/bev_cache.pt"))

# ======== src/fusion/attn_real.py ========
import os, sys
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
def obj_recall_prec(pred, tgt):
    tp_r = n_gt = tp_p = n_pr = 0
    for b in range(pred.shape[0]):
        gl, ng = ndimage.label(tgt[b, 0] > 0.5); pl, npr = ndimage.label(pred[b, 0] > 0.5)
        n_gt += ng; n_pr += npr
        for g in range(1, ng + 1):
            if (pred[b, 0][gl == g] > 0.5).any(): tp_r += 1
        for p in range(1, npr + 1):
            if (tgt[b, 0][pl == p] > 0.5).any(): tp_p += 1
    return tp_r / max(n_gt, 1), tp_p / max(n_pr, 1)
def train_eval(data, gate_kind, seed, steps=400):
    lidar, radar, relf, tgt = data
    n = len(lidar)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, va = perm[:int(n * 0.8)], perm[int(n * 0.8):]
    to = lambda t, idx: t[idx].to(DEV)
    Ltr, Rtr, Ftr, Ttr = to(lidar, tr), to(radar, tr), to(relf, tr), to(tgt, tr)
    Lva, Rva, Fva, Tva = to(lidar, va), to(radar, va), to(relf, va), to(tgt, va)
    torch.manual_seed(seed)
    net = BEVAttnFusionNet(c_l=3, c_r=3, relf=3, gate_kind=gate_kind).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
    best = (-1, 0, 0)  # recall, prec, mean_gate  (early stopping on val recall)
    for step in range(steps):
        net.train(); hm, g = net(Ltr, Rtr, Ftr)
        loss = F.binary_cross_entropy_with_logits(hm, Ttr, pos_weight=torch.tensor(40.0, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            net.eval()
            with torch.no_grad():
                hv, gv = net(Lva, Rva, Fva)
                rec, prec = obj_recall_prec((hv.sigmoid() > 0.5).float().cpu().numpy(), Tva.cpu().numpy())
            if rec > best[0]:
                best = (rec, prec, gv.mean().item())
    return best
def main():
    d = torch.load(os.path.join(REPO, "sync/res/benchmark_v1/bev_cache.pt"))
    data = (d["lidar"].float(), d["radar"].float(), d["relf"].float(), d["target"].float())
    print(f"device={DEV}  {len(data[0])} real clear frames  grid {tuple(data[0].shape[2:])}  "
          f"(early-stop best val recall, 3 seeds)\n")
    print(f"{'gate':10s} | {'best val recall':>16s} | {'prec@best':>10s} | {'mean gate':>9s}")
    print("-" * 56)
    for kind in ["threshold", "shallow", "feature"]:
        res = [train_eval(data, kind, s) for s in (0, 1, 2)]
        rec = np.array([r[0] for r in res]); prec = np.array([r[1] for r in res]); mg = np.array([r[2] for r in res])
        print(f"{kind:10s} | {rec.mean():6.2f} +/- {rec.std():4.2f}   | {prec.mean():10.2f} | {mg.mean():9.3f}")
    print("\nbest val recall = object recall at the early-stopping peak (avg of 3 seeds).")
    print("Clear-condition control: if gate variants tie here, that's EXPECTED (no lidar-blind cells to exploit);")
    print("the discriminating test is test_glare (lidar-blind) with verified GT.")

def real_main():
    sys.path.insert(0, os.path.join(REPO, "sync/src/fusion"))
    main()

if __name__ == '__main__':
    _m = {'proto': proto_main, 'rasterize': rasterize_main, 'real': real_main}
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
