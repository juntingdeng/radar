"""Level-2 frame loader + UNION GT, fully offline (reuses the persistent env npz + lidar caches + benchmark labels).

Per frame it assembles the three modality inputs and the ground truth:
  cam        (480,640,3) RGB            (from the env npz; depth loaded on demand from the extracted dir)
  lidar_pts  (N,4) x,y,z,intensity      (from ~/.cache lidar scan cache by lidar_idx)
  radar_bev  (301,601) + la,fa          (from the env npz) ; radar_pts = strong BEV cells as (fwd,lat,power)
  day_smoke frames are the day_clean frames degraded with the REAL measured 10.15 smoke layers (apply_smoke below);
  cam+lidar scatter, radar barely changes, GT is the clean frame's GT -- the regime the generator was built for.
  gt         full benchmark GT w/ PROVENANCE  (cam-raised vs lidar/radar-raised vs hand-verified)
  union_gt   provenance-deduped real objects (an object = any GT, merged across provenance by BEV IoU)

Provenance (reviewer: report stratified; miss rate needs a complete union answer key):
  cam        -> yolo_camera_projection, yolo_lidar_frustum          (needed the camera to find it)
  lidarradar -> lidar_radar_proposal, glare_lidar(_dense), radar_doppler_rescue, glare_moving, glare_glare_dense
  verified   -> hand-checked subset (test_glare_verified)
"""
import os, json, glob, numpy as np
import boxes as BX
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))      # late_fusion/fusion/src/sync -> .../code
RL_ENV = os.path.expanduser("~/.cache/ti_setup_frames/rl_env")
NCACHE = os.path.expanduser("~/.cache/ti_setup_frames/lidar_scan_cache")
DCACHE = os.path.expanduser("~/.cache/ti_setup_frames/day08_lidar")
LABELS = os.path.join(CODE, "sync", "res", "benchmark_v1", "labels")
CAM_EXTRACT = {  # extracted camera dirs (for depth-lift later); local copies
    "2026.05.11": os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.11/camera_extracted"),
    "2026.05.08": os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.08/camera_extracted"),
}
CAM_PROV = {"yolo_camera_projection", "yolo_lidar_frustum"}
LR_PROV = {"lidar_radar_proposal", "glare_lidar", "glare_lidar_dense", "radar_doppler_rescue",
           "glare_moving", "glare_glare_dense"}

def _provenance(src):
    if src in CAM_PROV: return "cam"
    if src in LR_PROV:  return "lidarradar"
    return "verified"                                                    # test_glare_verified / unknown

def load_split():
    """Canonical scene split -- the SINGLE source of truth (late_fusion/split.json). Returns the set of TRAIN pids;
    a scene (pid) not in it is TEST. Every trainer and eval must read this, never a checkpoint/pkl copy."""
    return set(json.load(open(os.path.join(HERE, "split.json")))["train_pids"])

def _load_labels():
    lab = {}
    for f in ["train", "val", "test_clear", "test_glare", "test_glare_verified"]:
        p = os.path.join(LABELS, f + ".json")
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                lab.setdefault(k, []).extend(v)
    return lab

def radar_points(bev, la, fa, excess_db=6.0, cap=3200):
    """Strong radar cells by PER-RANGE excess over the local ring floor (CFAR-style), NOT a global percentile.
    Radar power falls steeply with range and the global top-4% is owned by a few huge scatterers (walls), which
    starved both far AND near objects (measured: 44% of day GT boxes received ZERO cells). A cell is kept iff its
    power exceeds the median of its own 2 m radial ring by excess_db; the point's power feature = that excess,
    comparable across range. Cap swept on day GT: signal-in-box 0.70/0.78/0.87/0.94 at 400/800/1600/3200
    -> 3200 (~3.5% of the wedge; the old global-p96/cap-400 gave 0.56). -> (M,3) (fwd, lat, excess_dB)."""
    b = np.nan_to_num(np.asarray(bev, float))
    FA, LA = np.meshgrid(np.asarray(fa, float), np.asarray(la, float), indexing="ij")
    rings = np.clip((np.hypot(FA, LA) / 2.0).astype(int), 0, 29)
    valid = b > 1e-6                                                     # inside the beamformed wedge
    floor = np.full_like(b, np.inf)
    gmed = np.median(b[valid]) if valid.any() else 0.0
    for r in range(30):
        m = valid & (rings == r)
        if m.any(): floor[rings == r] = np.median(b[m]) if m.sum() >= 20 else gmed
    exc = np.where(valid, b - floor, -np.inf)
    yy, xx = np.where(exc >= excess_db)
    if len(yy) == 0: return np.zeros((0, 3), np.float32)
    pw = exc[yy, xx]
    if len(yy) > cap:                                                    # keep the strongest excess
        k = np.argsort(pw)[::-1][:cap]; yy, xx, pw = yy[k], xx[k], pw[k]
    return np.stack([np.asarray(fa)[yy], np.asarray(la)[xx], pw], 1).astype(np.float32)

def _gt_for(sample_id, labels, fwd_lo=0.0, fwd_hi=50.0):
    """Full GT boxes for a frame -> list of dict(box, prov, fwd, lat, label). BEV box from forward/lateral."""
    out = []
    for b in labels.get(sample_id, []):
        if "forward" not in b or "lateral" not in b: continue
        fc = (b["forward"][0] + b["forward"][1]) / 2
        if not (fwd_lo <= fc <= fwd_hi): continue
        out.append(dict(box=BX.to_bev(b["forward"], b["lateral"]), prov=_provenance(b.get("source", "?")),
                        fwd=tuple(b["forward"]), lat=tuple(b["lateral"]), label=b.get("label", "car"),
                        bbox2d=b.get("bbox_xyxy")))                  # native 2D box (None for lidar/radar-raised)
    return out

def _union(gt, iou_thr=0.3):
    """Merge GT across provenance into real objects. -> list of dict(box, provs:set, members, label).
    label = majority class among the merged members (ties -> the seed/biggest member's class)."""
    if not gt: return []
    B = np.stack([g["box"] for g in gt]); used = np.zeros(len(gt), bool); groups = []
    order = np.argsort([(g["box"][2]-g["box"][0])*(g["box"][3]-g["box"][1]) for g in gt])[::-1]  # big first
    for i in order:
        if used[i]: continue
        ious = BX.iou_matrix(B[i:i+1], B)[0]; members = np.where((ious >= iou_thr) & (~used))[0]
        used[members] = True
        mb = B[members]; box = np.array([mb[:, 0].min(), mb[:, 1].min(), mb[:, 2].max(), mb[:, 3].max()])
        labs = [gt[m]["label"] for m in members]
        lab = max(set(labs), key=lambda l: (labs.count(l), l == gt[i]["label"]))
        nat = [gt[m]["bbox2d"] for m in members if gt[m].get("bbox2d")]
        bb2 = max(nat, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if nat else None   # native 2D box, if any member has one
        groups.append(dict(box=box, provs=set(gt[m]["prov"] for m in members), members=[int(m) for m in members],
                           label=lab, bbox2d=bb2))
    return groups

# ---------------- day_smoke: composite the corrected 10.15 smoke layers onto clean 05.08 frames ----------------
# Same calibrated recipe as rl_modality/build_env_multi.py (camera alpha/airlight matte, lidar beam-exact occlusion
# + transplanted backscatter via reconstruct_smoke._lidar_apply, radar BEV weighted delta), same deterministic layer
# pick (i*5 % len(lib)), so the two pipelines see the SAME degraded frames. Layers are real measured smoke.
_smoke = {}
def _smoke_lib():
    if not _smoke:
        import sys
        sys.path.insert(0, os.path.join(CODE, "sync", "res", "2026.10.15"))
        import reconstruct_smoke as RS
        L = np.load(os.path.join(RL_ENV, "smoke_library_corrected.npz"), allow_pickle=True)
        _smoke.update(RS=RS, lib=list(L["lib"]), clean_bev=np.nan_to_num(L["clean_bev"].astype(float)),
                      Blum=float(np.percentile(L["clean_cam"].astype(float).mean(2), 50)))
    return _smoke

def apply_smoke(i, cam, lid, bev, gt):
    """Degrade a clean day frame (cam uint8 (480,640,3), lid (N,3), bev (301,601)) with the i-th smoke layer.
    GT is only used to mirror the layer onto the object side (image-left objects <-> mean GT lateral > 0)."""
    S = _smoke_lib(); lay = S["lib"][(i * 5) % len(S["lib"])]
    ratio = float(np.percentile(cam.astype(float).mean(2), 50)) / S["Blum"]
    lats = [(g["lat"][0] + g["lat"][1]) / 2 for g in gt]
    flip = bool(lats and np.mean(lats) > 0)                          # lat>0 projects image-left (lat ~ -X_cam)
    a = lay["alpha"].astype(float); Af = lay["airlight"].astype(float) * ratio
    bs = lay["backscatter"].astype(float).copy()
    w = lay["bev_w"].astype(float); d = np.nan_to_num(lay["bev_smoke"].astype(float)) - S["clean_bev"]
    if flip:
        a = a[:, ::-1]; Af = Af[:, ::-1]; bs[:, 1] *= -1; w = w[:, ::-1]; d = d[:, ::-1]
    cam_s = np.clip(cam.astype(float) * (1 - a[..., None]) + Af * a[..., None], 0, 255).astype(np.uint8)
    lid_s = S["RS"]._lidar_apply(lid.astype(float), bs).astype(np.float32)
    bev_s = bev + w * np.clip(d, 0, None)
    return cam_s, lid_s, bev_s

def load_depth(seq, cam_idx):
    """On-demand depth frame (H,W) uint16 mm from the extracted dir, or None."""
    import cv2
    p = os.path.join(CAM_EXTRACT.get(seq, ""), "depth", f"{int(cam_idx):06d}.png")
    if not os.path.exists(p): return None
    return cv2.imread(p, cv2.IMREAD_UNCHANGED)

def load_frames(conditions=("night_clean", "night_glare", "day_clean", "day_smoke")):
    labels = _load_labels()
    seq_of = {"night_clean": "2026.05.11", "night_glare": "2026.05.11", "day_clean": "2026.05.08", "day_smoke": "2026.05.08"}
    srcs = {"night_clean": ("scenes50.npz", NCACHE), "night_glare": ("glare.npz", NCACHE),
            "day_clean": ("day.npz", DCACHE), "day_smoke": ("day.npz", DCACHE)}
    frames = []
    for cls in conditions:
        npz, lcache = srcs[cls]; seq = seq_of[cls]
        S = np.load(os.path.join(RL_ENV, npz), allow_pickle=True)
        meta = json.loads(str(S["meta"])); la, fa = S["la"], S["fa"]
        for i, mt in enumerate(meta):
            cam = S["cams"][i].astype(np.uint8)
            if cam.max() == 0: continue
            lp = os.path.join(lcache, f"l_{mt['lidar_idx']}.npy")
            if not os.path.exists(lp): continue
            lid = np.load(lp).astype(np.float32)
            bev = np.nan_to_num(S["bevs"][i].astype(float))
            sid = f"{seq[5:7]}{seq[8:10]}_{mt['pid']:06d}"               # 2026.05.11 -> 0511_013890
            gt = _gt_for(sid, labels)
            if cls == "day_smoke":
                cam, lid, bev = apply_smoke(i, cam, lid, bev, gt)
            if lid.shape[1] == 3:                                        # x,y,z -> add dummy intensity
                lid = np.concatenate([lid, np.ones((len(lid), 1), np.float32)], 1)
            frames.append(dict(fid=f"{cls}:{mt['pid']}", cls=cls, seq=seq, pid=int(mt["pid"]),
                               cam_idx=int(mt.get("cam_idx", -1)), cam=cam, lidar_pts=lid,
                               radar_bev=bev, radar_pts=radar_points(bev, la, fa), la=la, fa=fa,
                               gt=gt, union_gt=_union(gt)))
    return frames

if __name__ == "__main__":
    import collections
    fr = load_frames()
    print(f"frames: {len(fr)}  by condition: {dict(collections.Counter(f['cls'] for f in fr))}")
    prov = collections.Counter(g["prov"] for f in fr for g in f["gt"])
    print(f"GT boxes: {sum(len(f['gt']) for f in fr)}  by provenance: {dict(prov)}")
    print(f"union-GT objects: {sum(len(f['union_gt']) for f in fr)}")
    ex = fr[0]
    print(f"example {ex['fid']}: cam{ex['cam'].shape} lidar{ex['lidar_pts'].shape} radar_pts{ex['radar_pts'].shape} "
          f"gt={len(ex['gt'])} union={len(ex['union_gt'])}")
    # how often is an object confirmed by >1 provenance (sanity on the union)
    multi = sum(1 for f in fr for u in f["union_gt"] if len(u["provs"]) > 1)
    print(f"union objects confirmed by >1 provenance: {multi}")
