"""CAMERA detector, DETR ARCHITECTURE, pretrained on the train scenes. One of the three frozen detectors
(cam=this, lidar/radar=voxel center-heads in detectors_l2.pt) behind every baseline and the RL loop.

Backbone+transformer = pretrained DETR-ResNet-50, FROZEN (121 train frames cannot support a full fine-tune).
Trained task heads sit on the decoder QUERY EMBEDDINGS (256-d, one per detection slot), so the whole inference
path stays a single DETR pass: image -> queries -> heads -> BEV detections.
  classification  cls head: query embedding + 2D geometry -> object logit for OUR scenes      (BCE, pos-weighted)
  localisation    bearing is ANALYTIC from the query box's pixel column (intrinsics+extrinsic rotation, nothing to
                  learn); reg head predicts the hard part: log RANGE  -> ATE                  (L1 on log)
                  range features = mono cues (apparent size, rows, analytic ground/height ranges) + the D435's OWN
                  box depth with a validity flag. Pure mono is ill-posed on this rig (camera ~0.37 m above ground
                  -> ground-contact ray near-parallel to ground; 480p apparent height ~8 m median) -- the trained
                  head uses measured depth when valid and falls back to mono cues when not (night blobs, far, smoke).
  box             reg head also predicts log BEV extents (w, l)                -> ASE        (L1 on log)

Supervision: union GT projected into the image through the per-capture CALIB_KEY extrinsic (BEV box + 0..1.8 m
nominal height -> 2D box); a query candidate is a positive iff its 2D centre falls inside a projected GT box.
Out-of-FOV GT never projects -> the camera only learns what it can physically see (same principle as the voxel
detectors only learning objects they have signal for). No depth anywhere.

Train: python camera_detector.py [mixed|dayfull] [stride]
  mixed   (default) the 121 L2 benchmark train scenes, all four conditions -> camera_detr.pt (PRODUCTION:
          this is what propose()/precompute/eval/RL load; it must see night+smoke or it goes blind there)
  dayfull ALL labelled 0508 day frames at the given stride (default 5 -> ~1.4k frames) -> camera_detr_dayfull.pt
          (EXPERIMENT: the data-scale ceiling probe; day-only, never deploy it). Guards: test_clear sids excluded,
          +/-50-pid band around L2 day test pids, z0 reused from camera_detr.pt (no lidar needed).
Both modes share the identical architecture, losses, candidate matching, and held-out diagnostics.
Use:   LF_CAM_BACKEND=mono python precompute_proposals.py
"""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import camera_proposals as CP
HERE = os.path.dirname(os.path.abspath(__file__)); torch.manual_seed(0)
PERSONISH = {1, 2, 4}        # COCO ids person/bicycle/motorcycle: different size prior than cars
CAND_THR = 0.25              # DETR query candidate threshold (low on purpose -- OUR cls head sets the final score)
EMIT_THR = 0.5               # trained score needed to emit a detection

# ---------------- geometry ----------------
_geo = {}
def geo_calib(seq):
    """Calib for PROJECTION/BEARING: always the auto (ICP/PnP) extrinsic. The 05.11 manual key in CP.CALIB_KEY is
    right for the depth-lift (it was solved THROUGH the D435 depth, absorbing its bias) but geometrically wrong as
    a projector (frustum A/B: 7% GT hit vs 65% with auto) -- training labels made with it go blind at night."""
    if seq not in _geo:
        import json
        cb = dict(json.load(open(CP.REAL_CALIB[seq])))
        cb["camera_to_lidar"] = cb["camera_to_lidar_auto"]
        _geo[seq] = cb
    return _geo[seq]

def project_gt_box(u, cb, img_hw=(480, 640), zs=(0.0, 1.8)):
    """union-GT dict (box=[f0,l0,f1,l1]) -> 2D [x0,y0,x1,y1] in the colour image, or None if it doesn't project."""
    f0, l0, f1, l1 = u["box"]; ci = cb["color_intrinsics"]
    Tinv = np.linalg.inv(np.array(cb["camera_to_lidar"], float))
    corners = np.array([[f, l, z, 1.0] for f in (f0, f1) for l in (l0, l1) for z in zs])
    Pc = (Tinv @ corners.T).T[:, :3]
    if (Pc[:, 2] <= 0.5).any(): return None                                  # behind / grazing the camera
    px = ci["fx"] * Pc[:, 0] / Pc[:, 2] + ci["cx"]; py = ci["fy"] * Pc[:, 1] / Pc[:, 2] + ci["cy"]
    H, W = img_hw
    x0, x1 = px.min(), px.max(); y0, y1 = py.min(), py.max()
    ix0, ix1 = max(0.0, x0), min(float(W), x1); iy0, iy1 = max(0.0, y0), min(float(H), y1)
    if ix1 - ix0 < 8 or iy1 - iy0 < 8: return None                           # out of frame / sliver
    if (ix1 - ix0) * (iy1 - iy0) < 0.25 * max(1.0, (x1 - x0) * (y1 - y0)): return None   # mostly outside
    return [ix0, iy0, ix1, iy1]

def bearing(u_px, cb):
    """pixel column -> BEV bearing (rad) in the lidar frame, through intrinsics + extrinsic rotation."""
    ci = cb["color_intrinsics"]; R = np.array(cb["camera_to_lidar"], float)[:3, :3]
    d = R @ np.array([(u_px - ci["cx"]) / ci["fx"], 0.0, 1.0])
    return float(np.arctan2(d[1], d[0]))

def ground_range(u, v, cb, z0):
    """Range where the pixel ray meets the ground plane z=z0 (lidar frame). The classic monocular localisation cue
    computed EXACTLY through the calibrated extrinsic instead of hoping an MLP rediscovers it. 60 m if no hit."""
    ci = cb["color_intrinsics"]; T = np.array(cb["camera_to_lidar"], float)
    d = T[:3, :3] @ np.array([(u - ci["cx"]) / ci["fx"], (v - ci["cy"]) / ci["fy"], 1.0]); C = T[:3, 3]
    if d[2] >= -1e-6: return 60.0
    t = (z0 - C[2]) / d[2]
    if t <= 0: return 60.0
    p = C + t * d
    return float(np.clip(np.hypot(p[0], p[1]), 1.0, 60.0))

def box_depth(bbox2d, depth, cb):
    """Median valid D435 depth (m) in the central half of the 2D box, or 0.0 if unavailable. The camera's OWN range
    sensor (same device as the RGB) as a regression FEATURE -- on this rig the mono cues are near-degenerate: the
    camera sits only ~0.37 m above ground, so the ground-contact ray is almost parallel to the ground (an object at
    8 m and one at 30 m differ by a few pixels of box bottom), and apparent-height at 480p is worth ~8 m median."""
    if depth is None: return 0.0
    ci = cb["color_intrinsics"]; di = cb["intrinsics"]; sc = cb.get("depth_scale", 0.001)
    x0, y0, x1, y1 = bbox2d
    qx, qy = (x1 - x0) / 4, (y1 - y0) / 4
    def to_d(u, v):
        return (int(round((u - ci["cx"]) / ci["fx"] * di["fx"] + di["cx"])),
                int(round((v - ci["cy"]) / ci["fy"] * di["fy"] + di["cy"])))
    u0, v0 = to_d(x0 + qx, y0 + qy); u1, v1 = to_d(x1 - qx, y1 - qy)
    H, W = depth.shape
    u0, u1 = sorted((max(0, min(W - 1, u0)), max(0, min(W - 1, u1))))
    v0, v1 = sorted((max(0, min(H - 1, v0)), max(0, min(H - 1, v1))))
    patch = depth[v0:v1 + 1, u0:u1 + 1].astype(float) * sc
    valid = patch[(patch > 0.3) & (patch < 40)]
    if valid.size < 5: return 0.0
    # NEAREST-SURFACE statistic, not median: the object is the closest thing inside its own box; the median of a
    # near car's box reads the BACKGROUND through the windows (measured: 6.3 m car -> 25.6 m median).
    return float(np.percentile(valid, 25))

def geom_feat(bbox2d, score, label, img_hw, cb, z0, depth=None):
    """apparent size, rows, class, conf | analytic mono ranges (log) | D435 box depth (log, + valid flag).
    The reg head learns to trust the measured depth when valid and fall back to the mono cues when not."""
    H, W = img_hw; x0, y0, x1, y1 = bbox2d
    per = float(label in PERSONISH)
    rg = ground_range((x0 + x1) / 2, y1, cb, z0)
    rh = float(np.clip(cb["color_intrinsics"]["fy"] * (1.7 if per else 1.5) / max(8.0, y1 - y0), 1.0, 60.0))
    d = box_depth(bbox2d, depth, cb)
    return [(y1 - y0) / H, (x1 - x0) / W, y1 / H, (y0 + y1) / 2 / H, per, float(score),
            float(np.log(rg)), float(np.log(rh)), float(np.log(d)) if d > 0 else 0.0, float(d > 0)]

# ---------------- DETR forward: candidates WITH their query embeddings ----------------
def detr_queries(img, thr=CAND_THR):
    """One DETR pass -> [dict(q (256,), score, label, bbox2d xyxy)] for vehicle/person queries above thr."""
    import torch as T
    proc, model = _detr()
    with T.no_grad():
        out = model(**proc(images=img, return_tensors="pt"))
    probs = out.logits.softmax(-1)[0, :, :-1]                                # (100, 91), drop no-object
    emb = out.last_hidden_state[0]                                           # (100, 256) decoder queries
    H, W = img.shape[:2]; res = []
    for qi in range(probs.shape[0]):
        p, cls = probs[qi].max(0)
        if int(cls) not in CP.VEHICLE_PERSON_COCO or float(p) < thr: continue
        cx, cy, w, h = out.pred_boxes[0, qi].numpy()
        res.append(dict(q=emb[qi].numpy(), score=float(p), label=int(cls),
                        bbox2d=[(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]))
    return res

_d = {}
def _detr():
    if not _d:
        from transformers import DetrForObjectDetection, DetrImageProcessor
        _d["proc"] = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        _d["model"] = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()
    return _d["proc"], _d["model"]

# ---------------- trained heads ----------------
class DetrHeads(nn.Module):
    """cls: our-scene objectness. reg: [log r_mono, log fwd-extent, log lat-extent, gate logit].
    Range = sigmoid(gate)*depth + (1-sigmoid(gate))*r_mono when depth is valid, else r_mono -- a learned gate with
    the right inductive bias: near-field the measurement wins, far/night the mono regression takes over."""
    def __init__(self, d=256, g=10):
        super().__init__()
        self.cls = nn.Sequential(nn.Linear(d + g, 64), nn.ReLU(), nn.Linear(64, 1))
        self.reg = nn.Sequential(nn.Linear(d + g, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))
    def forward(self, q, g):
        x = torch.cat([q, g], -1)
        return self.cls(x).squeeze(-1), self.reg(x)

def gated_range(reg, geom):
    """reg (N,4), geom (N,10) -> range (N,). Blends measured box depth (geom[:,8:10] = log d, valid) with mono."""
    r_mono = torch.exp(reg[:, 0].clamp(-1, 4.5))
    d = torch.exp(geom[:, 8]) * geom[:, 9]
    gate = torch.sigmoid(reg[:, 3]) * geom[:, 9]
    return gate * d + (1 - gate) * r_mono

# ---------------- dataset: label DETR candidates against projected GT ----------------
def ground_z(frames):
    """Per-sequence ground height in the lidar frame (rig constant, estimated once from train lidar near-field)."""
    z0 = {}
    for seq in {f["seq"] for f in frames}:
        zs = np.concatenate([f["lidar_pts"][np.hypot(f["lidar_pts"][:, 0], f["lidar_pts"][:, 1]) < 20, 2]
                             for f in frames if f["seq"] == seq][:10])
        z0[seq] = float(np.percentile(zs, 5))
    return z0

def build_dataset(frames, z0, total=None):
    """-> Q (N,256), G (N,10), y (N,), reg mask (N,), reg targets (N,3). One DETR forward per frame.
    `frames` may be a list or a lazy generator of dicts with keys cam (RGB), cam_idx, seq, union_gt."""
    total = total if total is not None else (len(frames) if hasattr(frames, "__len__") else "?")
    Q, Gf, y, rm, rt = [], [], [], [], []
    for n, f in enumerate(frames):
        cb = geo_calib(f["seq"]); H, W = f["cam"].shape[:2]
        depth = CP.aligned_depth(f["seq"], f["cam_idx"])
        gts = []
        for u in f["union_gt"]:
            # 2D box for candidate matching: NATIVE bbox_xyxy when a member has one (tight, labelling-time truth);
            # BEV-hull reprojection only as the fallback for lidar/radar-raised objects with no image box.
            b2 = u.get("bbox2d") or project_gt_box(u, cb, (H, W))
            if b2 is None: continue
            fc = (u["box"][0] + u["box"][2]) / 2; lc = (u["box"][1] + u["box"][3]) / 2
            gts.append((b2, np.hypot(fc, lc), u["box"][2] - u["box"][0], u["box"][3] - u["box"][1]))
        for c in detr_queries(f["cam"]):
            cx = (c["bbox2d"][0] + c["bbox2d"][2]) / 2; cy = (c["bbox2d"][1] + c["bbox2d"][3]) / 2
            # match by 2D IoU: a range target only when the candidate box IS this object (IoU>=0.25). A merged
            # multi-object box (centre merely inside some GT) stays a cls positive but must NOT supervise range --
            # arbitrary candidate<->GT pairing taught the regressor the dataset-mean range and killed the gate.
            iou, g = 0.0, None
            for gg in gts:
                i2 = _iou2d(c["bbox2d"], gg[0])
                if i2 > iou: iou, g = i2, gg
            inside = any(gg[0][0] <= cx <= gg[0][2] and gg[0][1] <= cy <= gg[0][3] for gg in gts)
            Q.append(c["q"]); Gf.append(geom_feat(c["bbox2d"], c["score"], c["label"], (H, W), cb, z0[f["seq"]], depth))
            y.append(float(iou >= 0.25 or inside)); rm.append(float(iou >= 0.25))
            rt.append([np.log(max(1.0, g[1])), np.log(max(0.5, g[2])), np.log(max(0.5, g[3]))]
                      if iou >= 0.25 else [0.0, 0.0, 0.0])
        if n % 50 == 0: print(f"  dataset {n+1}/{total}: {len(y)} candidates ({int(sum(y))} pos)", flush=True)
    if not Q:
        return (np.zeros((0, 256), np.float32), np.zeros((0, 10), np.float32), np.zeros(0, np.float32),
                np.zeros(0, np.float32), np.zeros((0, 3), np.float32))
    return (np.stack(Q).astype(np.float32), np.array(Gf, np.float32), np.array(y, np.float32),
            np.array(rm, np.float32), np.array(rt, np.float32))

SEQCFG = {
    "2026.05.08": dict(res=("2026.05.08", "17-34-27"),
                       cam=os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.08/camera_extracted"),
                       excl=("test_clear",)),
    "2026.05.11": dict(res=("2026.05.11", "21.37.07"),
                       cam=os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.11/camera_extracted"),
                       excl=("test_glare", "test_glare_verified", "test_clear")),
}

def labeled_frames(seq, stride=5, guard=50):
    """Lazy frame dicts for ALL labelled pids of `seq` (train+val labels), images from the extracted jpgs.
    Excludes the seq's held-out label splits, ALL L2 benchmark pids (those come from data_l2 separately),
    and a +/-guard band around L2 TEST pids. -> (generator, count)."""
    import csv, json, pickle
    CODE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
    cfg = SEQCFG[seq]
    RES = os.path.join(CODE, "sync", "res", *cfg["res"])
    LAB = os.path.join(CODE, "sync", "res", "benchmark_v1", "labels")
    import data_l2
    P = pickle.load(open(f"{HERE}/proposals_l2.pkl", "rb")); trp = data_l2.load_split()
    pref = seq[5:7] + seq[8:10]
    l2_pids = {f["pid"] for f in P["frames"] if f["seq"] == seq}
    test_pids = [p for p in l2_pids if p not in trp]
    lab = {}
    for fn in ("train", "val"):
        for k, v in json.load(open(os.path.join(LAB, fn + ".json"))).items():
            if k.startswith(pref): lab.setdefault(k, []).extend(v)
    excl = set()
    for fn in cfg["excl"]:
        fp = os.path.join(LAB, fn + ".json")
        if os.path.exists(fp):
            excl |= {k for k in json.load(open(fp)) if k.startswith(pref)}
    pairs = {int(r["pair_idx"]): int(r["camera_color_idx"])
             for r in csv.DictReader(open(os.path.join(RES, "camera_sync_pairs.csv")))}
    pids = [int(s.split("_")[1]) for s in sorted(lab)
            if s not in excl and int(s.split("_")[1]) in pairs
            and int(s.split("_")[1]) not in l2_pids
            and not any(abs(int(s.split("_")[1]) - t) <= guard for t in test_pids)][::stride]
    def gen():
        import cv2
        for p in pids:
            fp = os.path.join(cfg["cam"], "color", f"{pairs[p]:06d}.jpg")
            if not os.path.exists(fp): fp = fp[:-3] + "png"
            img = cv2.imread(fp)
            if img is None: continue
            yield dict(cam=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cam_idx=pairs[p], seq=seq,
                       union_gt=data_l2._union(data_l2._gt_for(f"{pref}_{p:06d}", lab)))
    return gen(), len(pids)

def dayfull_frames(stride=5, guard=50):                   # back-compat alias
    return labeled_frames("2026.05.08", stride, guard)

def train(tr_frames, z0, epochs=200, lr=1e-3, total=None):
    Q, G, y, rm, rt = build_dataset(tr_frames, z0, total)
    print(f"train: {len(y)} DETR candidates, {int(y.sum())} positive")
    heads = DetrHeads(); opt = torch.optim.Adam(heads.parameters(), lr)
    Qt, Gt, yt = torch.tensor(Q), torch.tensor(G), torch.tensor(y)
    rmt, rtt = torch.tensor(rm), torch.tensor(rt)
    pw = torch.tensor((len(y) - y.sum()) / max(1.0, y.sum()))
    r_true = torch.exp(rtt[:, 0])
    d_meas = torch.exp(Gt[:, 8]) * Gt[:, 9]
    gate_target = ((d_meas - r_true).abs() < 1.5).float()          # depth is RIGHT here -> gate should open
    gmask = rmt * Gt[:, 9]                                          # supervise gate only on matched+valid-depth
    for ep in range(epochs):
        cl, rg = heads(Qt, Gt)
        lcls = F.binary_cross_entropy_with_logits(cl, yt, pos_weight=pw)
        # range loss in METRES: the eval radius is absolute (2 m / 1 m) -- log-space L1 penalises RELATIVE error,
        # which let the net compress far ranges toward the dataset prior (measured -7..-21 m at 20-37 m true).
        lrng = (torch.abs(gated_range(rg, Gt) - r_true) * rmt).sum() / rmt.sum().clamp(1)
        lext = (F.l1_loss(rg[:, 1:3], rtt[:, 1:3], reduction="none").mean(1) * rmt).sum() / rmt.sum().clamp(1)
        lgate = (F.binary_cross_entropy_with_logits(rg[:, 3], gate_target, reduction="none") * gmask
                 ).sum() / gmask.sum().clamp(1)
        loss = lcls + 0.5 * lrng + lext + 0.3 * lgate
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  ep {ep:3d} cls {lcls.item():.3f} rng(m) {lrng.item():.2f} ext {lext.item():.3f} "
                  f"gate {lgate.item():.3f}", flush=True)
    heads.eval(); return heads

# ---------------- inference ----------------
def load_nets(path=None):
    ck = torch.load(path or f"{HERE}/camera_detr.pt", map_location="cpu")
    heads = DetrHeads(); heads.load_state_dict(ck["heads"]); heads.eval()
    return heads, ck["z0"]

@torch.no_grad()
def propose(frame, nets):
    """frame -> [dict(bev_box, score, label, bbox2d)]: DETR queries scored by OUR cls head, localised by OUR
    range/extent regression + analytic bearing, BEV-NMS'd (duplicate DETR queries on one object)."""
    heads, z0 = nets
    cb = geo_calib(frame["seq"]); H, W = frame["cam"].shape[:2]; out = []
    depth = CP.aligned_depth(frame["seq"], frame["cam_idx"])
    # dedupe duplicate DETR queries in 2D (same object -> near-identical 2D box; distinct neighbours don't overlap)
    cands = sorted(detr_queries(frame["cam"]), key=lambda c: -c["score"])
    kept = []
    for c in cands:
        if all(_iou2d(c["bbox2d"], k["bbox2d"]) < 0.7 for k in kept): kept.append(c)
    for c in kept:
        gt_ = torch.tensor(np.array([geom_feat(c["bbox2d"], c["score"], c["label"], (H, W),
                                               cb, z0[frame["seq"]], depth)], np.float32))
        cl, rg = heads(torch.tensor(c["q"][None]), gt_)
        p = float(torch.sigmoid(cl))
        if p < EMIT_THR: continue
        r = float(gated_range(rg, gt_)[0])
        wf, wl = np.exp(rg.numpy()[0][1:3])
        az = bearing((c["bbox2d"][0] + c["bbox2d"][2]) / 2, cb)
        fc, lc = r * np.cos(az), r * np.sin(az)
        wf = float(np.clip(wf, 0.5, 12)); wl = float(np.clip(wl, 0.5, 12))
        out.append(dict(bev_box=[fc - wf / 2, lc - wl / 2, fc + wf / 2, lc + wl / 2],
                        score=p, label=c["label"], bbox2d=c["bbox2d"]))
    return out

def _iou2d(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0

if __name__ == "__main__":
    import sys, data_l2
    mode = sys.argv[1] if len(sys.argv) > 1 else "mixed"
    frames = data_l2.load_frames()
    trp = data_l2.load_split()                          # canonical split (split.json)
    te = [f for f in frames if f["pid"] not in trp]
    if mode == "mixed":
        tr = [f for f in frames if f["pid"] in trp]
        print(f"MIXED: {len(tr)} train / {len(te)} test frames (scene split shared with the voxel detectors)")
        z0 = ground_z(tr)
        print(f"ground z (lidar frame, rig constant): {z0}")
        heads = train(tr, z0)
        out = f"{HERE}/camera_detr.pt"
    elif mode == "dayfull":
        stride = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        z0 = load_nets()[1]                               # rig constant from the production checkpoint
        gen, n = dayfull_frames(stride=stride)
        print(f"DAYFULL: {n} day frames (stride {stride}); held-out diagnostics on day_clean L2 test scenes")
        heads = train(gen, z0, epochs=300, total=n)
        te = [f for f in te if f["cls"] == "day_clean"]   # day-only model -> day-only diagnostics
        out = f"{HERE}/camera_detr_dayfull.pt"
    elif mode == "fullmix":
        # camera analog of the voxels' packextra: L2 train x3 (all conditions; glare/smoke live ONLY there)
        # + labelled day (stride 20) + labelled night (stride 30), clean native matching, NEW checkpoint.
        import itertools
        tr = [f for f in frames if f["pid"] in trp]
        z0 = ground_z(tr)
        g8, n8 = labeled_frames("2026.05.08", stride=20)
        g11, n11 = labeled_frames("2026.05.11", stride=30)
        total = 3 * len(tr) + n8 + n11
        print(f"FULLMIX: {len(tr)}x3 L2 + {n8} day + {n11} night = {total} frames; "
              f"held-out diagnostics on ALL L2 test scenes")
        heads = train(itertools.chain(tr * 3, g8, g11), z0, epochs=300, total=total)
        out = f"{HERE}/camera_detr_fullmix.pt"
    else:
        raise SystemExit(f"unknown mode {mode!r}: use mixed|dayfull")
    torch.save({"heads": heads.state_dict(), "z0": z0}, out)
    print(f"saved {out}")
    # held-out diagnostics on the detector's own terms: candidate AUC, range error (-> ATE), extent error (-> ASE)
    Q, G, y, rm, rt = build_dataset(te, z0)
    with torch.no_grad():
        Gt_ = torch.tensor(G)
        cl, rg_t = heads(torch.tensor(Q), Gt_)
        p = torch.sigmoid(cl).numpy(); r_hat = gated_range(rg_t, Gt_).numpy(); rg = rg_t.numpy()
    from sklearn.metrics import roc_auc_score
    m = rm > 0.5
    rerr = np.abs(r_hat[m] - np.exp(rt[m, 0]))
    wp, lp = np.exp(rg[m, 1]), np.exp(rg[m, 2]); wg, lg = np.exp(rt[m, 1]), np.exp(rt[m, 2])
    inter = np.minimum(wp, wg) * np.minimum(lp, lg)
    ase = 1 - inter / (wp * lp + wg * lg - inter)
    print(f"TEST: cls AUC {roc_auc_score(y, p):.3f} on {len(y)} candidates ({int(y.sum())} pos)")
    print(f"TEST: range |err| median {np.median(rerr):.2f} m / mean {rerr.mean():.2f} m "
          f"(true range median {np.median(np.exp(rt[m, 0])):.1f} m)  [-> ATE]")
    print(f"TEST: extent ASE (1 - centre-aligned IoU) median {np.median(ase):.2f}  [-> ASE]")
