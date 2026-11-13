"""Doppler-gated night rescue (no LiDAR-motion tracking -- that was the bug).

Motion is judged by the RADAR DOPPLER (measured directly, reliable), NOT by tracking
LiDAR clusters (which merge/split at night and fabricate velocities for background).
  1. radar MOVING detections: CFAR + |Doppler| > v_min  -> real movers, static notched out
  2. snap raw LiDAR points at each detection's RANGE (precise) + azimuth (loose) -> car pts
  3. cluster -> box; size-sanity + NMS
Static background has no moving-Doppler return, so it is never proposed.
    python3 rescue_v3.py 14260 14270 14280 14290 14300
"""
import os, sys, json
import numpy as np
from sklearn.cluster import DBSCAN
from common import load_pairs, load_calib, LidarSource, RES
from detector import RadarDetector
from ego_doppler import fit_ego_velocity
sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _ground_level

TARGETS = [int(a) for a in sys.argv[1:]] or [14280]
ROI = (0.0, 30.0, -16.0, 16.0); GPCT = 10.0; HBAND = (0.3, 2.2)
MOVER_MARGIN = 0.8         # a true mover sits this far (m/s) off the ego-Doppler static line
RANGE_TOL = 1.5; AZ_TOL = 12.0; MIN_PTS = 12; MIN_EGO_INLIERS = 6

pairs = load_pairs(); cal = load_calib()
ci = cal["color_intrinsics"]; L2C = np.linalg.inv(np.array(cal["camera_to_lidar_auto"]))
R2L = np.array(cal["radar_to_lidar_auto"]); A = R2L[:2, :2]; t = R2L[:2, 3]; A_inv = np.linalg.inv(A)
det = RadarDetector(processing="legacy")   # all returns; ego-Doppler gate removes static below
src = LidarSource()


def corners(b):
    dx = np.array([b["L"]/2]*2 + [-b["L"]/2]*2); dy = np.array([b["W"]/2, -b["W"]/2, -b["W"]/2, b["W"]/2])
    cs, sn = np.cos(b["yaw"]), np.sin(b["yaw"]); fx = b["cx"]+dx*cs-dy*sn; fy = b["cy"]+dx*sn+dy*cs
    return np.c_[fx, fy, np.full(4, b["z0"])], np.c_[fx, fy, np.full(4, b["z1"])]

def to_px(xyz):
    opt = (L2C @ np.r_[xyz, 1])[:3]
    return None if opt[2] <= 0.2 else [float(ci["fx"]*opt[0]/opt[2]+ci["cx"]), float(ci["fy"]*opt[1]/opt[2]+ci["cy"])]

def box_from(cl):
    xy = cl[:, :2]; cen = xy.mean(0); X = xy - cen
    _, _, vt = np.linalg.svd(X, full_matrices=False); yaw = float(np.arctan2(vt[0, 1], vt[0, 0]))
    proj = X @ vt.T; L = max(float(np.ptp(proj[:, 0])), 1.5); W = max(float(np.ptp(proj[:, 1])), 1.6)
    if L < 3.0: L = 4.3; yaw = float(np.arctan2(cen[1], cen[0]))
    z0 = float(cl[:, 2].min())
    return dict(cx=float(cen[0]), cy=float(cen[1]), yaw=yaw, L=min(L, 6.0), W=min(W, 2.2), z0=z0, z1=z0+1.5, n=int(cl.shape[0]))


def process(T):
    d, _ = det.detect(pairs[T]["radar_idx"])                     # all returns
    # ego-Doppler gate: drop detections whose Doppler matches ego-induced static motion
    if len(d) >= MIN_EGO_INLIERS:
        ab, inl = fit_ego_velocity(d["az"], d["vel"])
        if ab is not None and inl.sum() >= MIN_EGO_INLIERS:
            azr = np.radians(-d["az"]); pred = np.cos(azr)*ab[0] + np.sin(azr)*ab[1]
            d = d[(~inl) & (np.abs(d["vel"] - pred) > MOVER_MARGIN)]
        else:
            d = d[:0]
    else:
        d = d[:0]
    MOVERS[str(T)] = int(len(d))                                  # ego-Doppler mover count = the regime signal
    if not len(d):
        print(f"pair {T}: 0 true movers after ego-Doppler gate -> 0 boxes", flush=True); return []
    Xd, Yd = RadarDetector.to_xy(d); drng = np.hypot(Xd, Yd); dang = np.degrees(np.arctan2(Yd, Xd)); dpow = d["power"]
    P = src.xyz(pairs[T]["lidar_idx"])
    x0, x1, y0, y1 = ROI
    reg = P[(P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)]
    gz = _ground_level(reg, GPCT)
    slab = reg[(reg[:, 2] > gz + HBAND[0]) & (reg[:, 2] < gz + HBAND[1])]
    # lidar slab -> radar frame, then range+az
    rr = (A_inv @ (slab[:, :2] - t).T).T
    prng = np.hypot(rr[:, 0], rr[:, 1]); pang = np.degrees(np.arctan2(rr[:, 1], rr[:, 0]))
    backed = np.zeros(len(slab), bool)
    for dr, da in zip(drng, dang):
        backed |= (np.abs(prng - dr) < RANGE_TOL) & (np.abs(((pang - da + 180) % 360) - 180) < AZ_TOL)
    rp = slab[backed]
    boxes = []
    if len(rp) >= MIN_PTS:
        lab = DBSCAN(eps=1.0, min_samples=6).fit_predict(rp[:, :2])
        for c in set(lab.tolist()):
            if c == -1: continue
            cl = rp[lab == c]
            if cl.shape[0] >= MIN_PTS:
                b = box_from(cl)
                # radar strength backing this box = max power of moving dets at its range/azimuth
                rc = A_inv @ (np.array([b["cx"], b["cy"]]) - t)
                brng = np.hypot(*rc); bang = np.degrees(np.arctan2(rc[1], rc[0]))
                near = (np.abs(drng-brng) < RANGE_TOL+1.0) & (np.abs(((dang-bang+180) % 360)-180) < AZ_TOL)
                b["pow"] = float(dpow[near].max()) if near.any() else -1e9
                boxes.append(b)
    # min-range 8m drops the persistent near-range static (curb/pole ~5.6m) that kept stealing the pick
    boxes = [b for b in boxes if 1.5 <= b["L"] <= 6.0 and b["W"] <= 3.0 and np.hypot(b["cx"], b["cy"]) >= 8.0]
    for b in sorted(boxes, key=lambda z: -z["pow"]):
        print(f"      cand rng {np.hypot(b['cx'],b['cy']):5.1f} cen=({b['cx']:5.1f},{b['cy']:6.1f}) n{b['n']:3d} pow{b['pow']:6.1f}", flush=True)
    # 1) POWER GATE qualifies the car region (rejects weak-radar background) -- the car's radar is strong
    if boxes:
        pmax = max(b["pow"] for b in boxes)
        boxes = [b for b in boxes if b["pow"] >= pmax - 2.5]
    # 2) NMS keeps the DENSEST box among overlaps (best car geometry; corners are sparse)
    boxes.sort(key=lambda b: -b["n"]); kept = []
    for b in boxes:
        if any(np.hypot(b["cx"]-k["cx"], b["cy"]-k["cy"]) < 4.0 for k in kept): continue  # merge one-car splits
        kept.append(b)
    print(f"pair {T}: {len(d)} movers(gated), {backed.sum()} backed pts -> {len(kept)} boxes", flush=True)
    for b in kept:
        print(f"    rng {np.hypot(b['cx'],b['cy']):5.1f} cen=({b['cx']:5.1f},{b['cy']:6.1f}) n{b['n']} pow{b['pow']:.1f}", flush=True)
    return kept


allobj = {}
MOVERS = {}
for T in TARGETS:
    lst = []
    for b in process(T):
        bo, tp = corners(b); px = [to_px(x) for x in list(bo)+list(tp)]; px = [p for p in px if p]
        bbox = None
        if len(px) >= 4:
            a = np.array(px); bbox = [float(a[:,0].min()), float(a[:,1].min()), float(a[:,0].max()), float(a[:,1].max())]
        box3d = dict(cx=b["cx"], cy=b["cy"], cz=(b["z0"]+b["z1"])/2, yaw=b["yaw"], length=b["L"], width=b["W"],
                     height=b["z1"]-b["z0"], reliable=True, size_plausible=True, length_fit=0.0, n_points=b["n"])
        lst.append(dict(label="car", source="radar_doppler_rescue", score=0.6, color="lime",
                        bbox_xyxy=bbox, box3d=box3d, radar_confirmed=True,
                        forward=[b["cx"]-b["L"]/2, b["cx"]+b["L"]/2], lateral=[b["cy"]-b["W"]/2, b["cy"]+b["W"]/2]))
    allobj[str(T)] = lst
ANN = f"{RES}/annotations/rescue_v3.json"
json.dump(dict(objects=allobj, movers=MOVERS), open(ANN, "w"), indent=1)
print(f"wrote {ANN} ({sum(len(v) for v in allobj.values())} boxes / {len(allobj)} pairs)", flush=True)
