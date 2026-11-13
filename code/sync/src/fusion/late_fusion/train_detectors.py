"""Train the small voxel detectors center-head style on the UNION GT, SCENE-split.

Modalities: lidar (raw scan), radar (strong BEV cells). The trained CAMERA detector lives in camera_detector.py
(2D crops + monocular range regression -- a different architecture, same train split).
Target: each GT object's centre cell -> objectness 1 (classification) + regression (centre offset dx,dy within the
cell, extents w,l). Loss: BCE(objectness, pos-weighted) + L1(box reg on positive cells). Trained ONLY on the train
scenes so the Level-2 eval on test scenes is honest. Each modality only learns objects it has signal for (empty
pillars -> 0 feature -> background), so e.g. lidar naturally gets high miss rate on camera-only objects.

Saves/updates detectors_l2.pt (only the requested modalities are retrained; the rest keep their weights).
Run: python train_detectors.py [lidar radar] [packextra]     (default: both, L2 train scenes only)
  packextra: BALANCED data-scale training = L2 train frames x3 (keeps glare/night/smoke weight) + ~350 pack day
  frames (stride 20) + ~380 pack night frames (stride 30), epochs=15. The day-only variant (697 day frames)
  closed the day train/test gaps (lidar 0.22->0.08, radar ->~0) but DROWNED night (lidar night miss 0.20->0.35,
  glare 0.42->0.66) -- condition balance is mandatory, not optional.

Models are tiny prototypes on ~120 frames -> weak absolute numbers; the deliverable is the working detector + the
per-modality miss-rate / recoverability it enables.
"""
import os, sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import data_l2
from voxel_detector import VoxelDetector, decode
import boxes as BX
HERE = os.path.dirname(os.path.abspath(__file__)); torch.manual_seed(0); np.random.seed(0)
IN_CH = {"lidar": 4, "radar": 3}

def build_target(model, gt_boxes):
    H, W = model.H, model.W
    obj = torch.zeros(H, W); reg = torch.zeros(4, H, W); mask = torch.zeros(H, W)
    fs, ls = model.cell_centers()
    for b in gt_boxes:
        cx = (b[0] + b[2]) / 2; cy = (b[1] + b[3]) / 2; w = b[2] - b[0]; l = b[3] - b[1]
        if not (model.fmin <= cx < model.fmax and model.lmin <= cy < model.lmax): continue
        px = min(max(int((cx - model.fmin) / model.cell), 0), H - 1)
        py = min(max(int((cy - model.lmin) / model.cell), 0), W - 1)
        obj[px, py] = 1; mask[px, py] = 1
        reg[0, px, py] = cx - fs[px]; reg[1, px, py] = cy - ls[py]; reg[2, px, py] = w; reg[3, px, py] = l
    return obj, reg, mask

def frame_pts(f, modality):
    return torch.tensor(f["lidar_pts"] if modality == "lidar" else f["radar_pts"]).float()

def train_one(modality, in_ch, frames, epochs=80, lr=3e-3):
    model = VoxelDetector(in_ch=in_ch); opt = torch.optim.Adam(model.parameters(), lr)
    # precompute targets
    data = []
    for f in frames:
        pts = frame_pts(f, modality)
        if len(pts) == 0: continue
        gt = [u["box"] for u in f["union_gt"]]
        data.append((pts, *build_target(model, gt)))
    print(f"  [{modality}] {len(data)} train frames")
    for ep in range(epochs):
        model.train(); tot = 0.0; np.random.shuffle(data)
        for pts, obj_t, reg_t, mask in data:
            out = model(pts); ol = out[0]; reg = out[1:]
            npos = max(1.0, mask.sum().item())
            pw = torch.tensor((model.H * model.W) / npos).clamp(5, 200)
            lobj = F.binary_cross_entropy_with_logits(ol, obj_t, pos_weight=pw)
            lreg = (F.l1_loss(reg, reg_t, reduction="none") * mask).sum() / (npos * 4)
            loss = lobj + lreg
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"    ep {ep:3d} loss {tot/len(data):.3f}")
    model.eval(); return model

def quick_recall(model, modality, frames, iou_thr=0.3):
    """fraction of union-GT objects with a decoded proposal IoU>=thr (sanity of the trained detector)."""
    tp = tot = 0
    for f in frames:
        pts = frame_pts(f, modality)
        if len(pts) == 0: continue
        b, s = decode(model, model(pts), score_thr=0.3, topk=20)
        G = np.stack([u["box"] for u in f["union_gt"]]) if f["union_gt"] else np.zeros((0, 4))
        tot += len(G)
        if len(G) and len(b):
            M = BX.iou_matrix(G, b)
            tp += int((M.max(1) >= iou_thr).sum())
    return tp / max(1, tot)

if __name__ == "__main__":
    which = [m for m in sys.argv[1:] if m in IN_CH] or ["lidar", "radar"]
    frames = data_l2.load_frames()
    if os.path.exists(f"{HERE}/split.json"):
        trp = data_l2.load_split()                          # canonical split (split.json) -- never regenerate silently
    else:
        pids = sorted({f["pid"] for f in frames}); rng = np.random.default_rng(0); rng.shuffle(pids)
        trp = set(pids[:int(0.7 * len(pids))])
        json.dump(dict(note="canonical scene split -- single source of truth", seed=0, train_frac=0.7,
                       n_scenes=len(pids), train_pids=sorted(trp),
                       test_pids=sorted(set(pids) - trp)), open(f"{HERE}/split.json", "w"), indent=1)
        print(f"created {HERE}/split.json")
    tr = [f for f in frames if f["pid"] in trp]; te = [f for f in frames if f["pid"] not in trp]
    nsc = len({f["pid"] for f in frames})
    extra = []
    if "packextra" in sys.argv:
        import pack_day
        extra = pack_day.pack_frames("2026.05.08", stride=20) + pack_day.pack_frames("2026.05.11", stride=30)
        tr = tr * 3                                     # oversample L2 frames: glare/smoke exist ONLY there
    epochs = 15 if extra else 80                        # data-rich -> fewer epochs
    print(f"{len(frames)} frames -> train {len(tr)}+{len(extra)} extra / test {len(te)}  "
          f"({nsc} scenes); training {which} for {epochs} epochs")
    ck = torch.load(f"{HERE}/detectors_l2.pt", map_location="cpu") if os.path.exists(f"{HERE}/detectors_l2.pt") else {}
    models = {m: train_one(m, IN_CH[m], tr + extra, epochs=epochs) for m in which}
    for m, mod in models.items():
        print(f"union-GT recall@0.3 [{m}]:  test {quick_recall(mod, m, te):.2f}  train {quick_recall(mod, m, tr):.2f}")
        ck[m] = mod.state_dict()
    ck["train_pids"] = sorted(trp)
    torch.save(ck, f"{HERE}/detectors_l2.pt")
    print(f"saved {HERE}/detectors_l2.pt (modalities: {[k for k in ck if k != 'train_pids']})")
