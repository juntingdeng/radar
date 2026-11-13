"""Reliability-gated fusion head over cross-modal detection groups.

Per group, feature (17-d) =
  [s_cam, s_lid, s_rad | present(3) | pair_close(3) | nmod/3]                   base 10-d: detector CONFIDENCES
                                                                                 + DETECTION-level spatial agreement
  [pair_size(3)]                                                                 extent agreement per present pair
  [lid_occ, lid_cnt, rad_mean, rad_max]                                          RAW-EVIDENCE agreement: does the raw
                                                                                 lidar/radar actually light up inside
                                                                                 the group box (from the ev rasters)?
The raw-evidence dims let the head veto a detector that fires where its own sensor has nothing (hallucinated boxes)
and trust a lone camera box that radar power quietly corroborates. The head -> one fused logit; detect iff > tau.
Trained on a pooled soft-F1 over train-scene groups (a group is a positive if its box centre-matches a union-GT
object), so it learns to KEEP corroborated groups and REJECT uncorroborated ones (the camera's lonely false positives).

NOTE on cost: the weighting-policy's lambda*cost ("which sensor to run") has no clean per-group meaning here -- all
three detectors run to form groups. So this detection-level head optimises F1 directly. (A compute-aware two-stage
variant -- cheap probes decide what to run, then fuse -- is future work.)"""
import numpy as np, torch, torch.nn as nn
import associate as AS
from voxel_detector import GRID, CELL

BASE_DIM = 10            # the pre-evidence feature prefix (scores+present+pair_close+nmod) -- kept first for ablation
FEAT_DIM = BASE_DIM + 3 + 4

def _cells(box):
    """BEV box -> (slice_fwd, slice_lat) into the 64x64 ev rasters (>=1 cell each way)."""
    fmin, fmax, lmin, lmax = GRID
    H = int(round((fmax - fmin) / CELL)); W = int(round((lmax - lmin) / CELL))
    x0 = int(np.clip((box[0] - fmin) / CELL, 0, H - 1)); x1 = int(np.clip((box[2] - fmin) / CELL, 0, H - 1))
    y0 = int(np.clip((box[1] - lmin) / CELL, 0, W - 1)); y1 = int(np.clip((box[3] - lmin) / CELL, 0, W - 1))
    return slice(x0, x1 + 1), slice(y0, y1 + 1)

def evidence_feat(box, ev):
    """[lid_occ, lid_cnt, rad_mean, rad_max]: raw lidar occupancy / point count and z-scored radar power in the box."""
    if ev is None: return [0.0, 0.0, 0.0, 0.0]
    sx, sy = _cells(box)
    lid = np.asarray(ev["lid"], float)[sx, sy]
    lid_occ = float((lid > 0).mean()); lid_cnt = float(np.log1p(lid.sum()) / 8.0)
    rad = np.asarray(ev["rad"], float); mu, sd = rad.mean(), rad.std() + 1e-6
    z = (rad[sx, sy] - mu) / sd
    rad_mean = float(np.clip(z.mean(), -2, 5) / 5.0); rad_max = float(np.clip(z.max(), -2, 5) / 5.0)
    return [lid_occ, lid_cnt, rad_mean, rad_max]

def group_feat(g, ev=None):
    gf = AS.group_features(g)
    x = np.array(gf["scores"] + gf["present"] + gf["pair_iou"] + [gf["nmod"] / 3.0]
                 + gf["pair_size"] + evidence_feat(gf["box"], ev), float)
    return gf, x

class FusionHead(nn.Module):
    def __init__(self, d=FEAT_DIM):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_fusion(X, y, epochs=400, lr=5e-3, seed=0):
    """X:(N,d) features, y:(N,) 1=group matches a union-GT object. Pooled soft-F1. -> trained FusionHead.
    d is taken from X, so the same trainer serves the full 17-d head and the base-10 ablation (X[:, :BASE_DIM])."""
    torch.manual_seed(seed)
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32); yt = torch.tensor(np.asarray(y), dtype=torch.float32)
    head = FusionHead(d=Xt.shape[1]); opt = torch.optim.Adam(head.parameters(), lr); eps = 1e-6
    pos = yt > 0.5; neg = ~pos
    for _ in range(epochs):
        p = torch.sigmoid(head(Xt))
        stp = (p * pos).sum(); sfp = (p * neg).sum(); sfn = ((1 - p) * pos).sum()
        soft_f1 = 2 * stp / (2 * stp + sfp + sfn + eps)
        loss = -soft_f1
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval(); return head

@torch.no_grad()
def score_groups(head, groups, ev=None, dims=None):
    """-> (scores logits, boxes) for a frame's groups. dims: optional feature-prefix width (ablation)."""
    if not groups: return np.zeros((0,)), np.zeros((0, 4))
    feats = []; boxes = []
    for g in groups:
        gf, x = group_feat(g, ev); feats.append(x if dims is None else x[:dims]); boxes.append(gf["box"])
    s = head(torch.tensor(np.stack(feats), dtype=torch.float32)).numpy()
    return s, np.stack(boxes)
