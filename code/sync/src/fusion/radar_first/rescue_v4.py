"""Night moving-car rescue with TEMPORAL STATIC-REJECTION.

Builds on rescue_v3 (ego-Doppler gate + raw-LiDAR snap). The remaining failure was
static background that evades the ego-Doppler gate (spurious Doppler) and beats the car
on density/power. Discriminator: a real moving car occupies a WORLD location only briefly;
static clutter sits at the same world spot every frame. So:
  - compute candidate moving-backed boxes for every frame in a window (ego-compensated via odometry)
  - REJECT a target box if its world position is occupied by a candidate in many neighbor frames (static)
  - keep transient (moving) boxes -> the car
This is rejection via odometry, NOT the fragile per-cluster velocity tracking that failed.
    python3 rescue_v4.py 14260 14270 14280 14290 14300
"""
import os, sys, json
import numpy as np
from sklearn.cluster import DBSCAN
from common import load_pairs, load_calib, LidarSource, RES, pose_apply
from detector import RadarDetector
from ego_doppler import fit_ego_velocity
from odometry import odometry_scan2map
sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _ground_level

TARGETS = [int(a) for a in sys.argv[1:]] or [14280]
MARGIN = 4
ROI = (0.0, 30.0, -16.0, 16.0); GPCT = 10.0; HBAND = (0.3, 2.2)
MOVER_MARGIN = 0.8; RANGE_TOL = 1.5; AZ_TOL = 12.0; MIN_PTS = 12; MIN_EGO_INLIERS = 6
PERSIST_R = 1.6            # world radius to call two candidates "the same spot"
PERSIST_K = 3             # check only +/- this many neighbor frames (local window)
PERSIST_MAX = 3           # occupied in > this many of the 2*K neighbors -> static -> reject

pairs = load_pairs(); cal = load_calib()
ci = cal["color_intrinsics"]; L2C = np.linalg.inv(np.array(cal["camera_to_lidar_auto"]))
R2L = np.array(cal["radar_to_lidar_auto"]); A = R2L[:2, :2]; t = R2L[:2, 3]; A_inv = np.linalg.inv(A)
det = RadarDetector(processing="legacy")

W0, W1 = min(TARGETS) - MARGIN, max(TARGETS) + MARGIN
ids = [p for p in range(W0, W1 + 1) if p in pairs]
idx_of = {p: i for i, p in enumerate(ids)}
src = LidarSource()
scans = [src.xyz(pairs[p]["lidar_idx"]) for p in ids]
poses, _ = odometry_scan2map(scans)


def candidates(i):
    """Moving-radar-backed LiDAR clusters at frame index i -> boxes with world centroid."""
    P = scans[i]; T = ids[i]
    d, _ = det.detect(pairs[T]["radar_idx"])
    if len(d) < MIN_EGO_INLIERS: return []
    ab, inl = fit_ego_velocity(d["az"], d["vel"])
    if ab is None or inl.sum() < MIN_EGO_INLIERS: return []
    azr = np.radians(-d["az"]); pred = np.cos(azr)*ab[0] + np.sin(azr)*ab[1]
    d = d[(~inl) & (np.abs(d["vel"] - pred) > MOVER_MARGIN)]
    if not len(d): return []
    Xd, Yd = RadarDetector.to_xy(d); drng = np.hypot(Xd, Yd); dang = np.degrees(np.arctan2(Yd, Xd))
    x0, x1, y0, y1 = ROI
    reg = P[(P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)]
    gz = _ground_level(reg, GPCT)
    slab = reg[(reg[:, 2] > gz + HBAND[0]) & (reg[:, 2] < gz + HBAND[1])]
    rr = (A_inv @ (slab[:, :2] - t).T).T
    prng = np.hypot(rr[:, 0], rr[:, 1]); pang = np.degrees(np.arctan2(rr[:, 1], rr[:, 0]))
    backed = np.zeros(len(slab), bool)
    for dr, da in zip(drng, dang):
        backed |= (np.abs(prng - dr) < RANGE_TOL) & (np.abs(((pang - da + 180) % 360) - 180) < AZ_TOL)
    rp = slab[backed]; out = []
    if len(rp) >= MIN_PTS:
        lab = DBSCAN(eps=1.0, min_samples=6).fit_predict(rp[:, :2])
        for c in set(lab.tolist()):
            if c == -1: continue
            cl = rp[lab == c]
            if cl.shape[0] < MIN_PTS: continue
            cen = cl[:, :2].mean(0)
            if np.hypot(*cen) < 4.0: continue
            out.append(dict(pts=cl, cen=cen, cen_w=pose_apply(poses[i], cen[None, :])[0], n=int(cl.shape[0])))
    return out

cand_per = [candidates(i) for i in range(len(ids))]


def box_from(cl):
    xy = cl[:, :2]; cen = xy.mean(0); X = xy - cen
    _, _, vt = np.linalg.svd(X, full_matrices=False); yaw = float(np.arctan2(vt[0, 1], vt[0, 0]))
    proj = X @ vt.T; L = max(float(np.ptp(proj[:, 0])), 1.5); W = max(float(np.ptp(proj[:, 1])), 1.6)
    if L < 3.0: L = 4.3; yaw = float(np.arctan2(cen[1], cen[0]))
    z0 = float(cl[:, 2].min())
    return dict(cx=float(cen[0]), cy=float(cen[1]), yaw=yaw, L=min(L, 6.0), W=min(W, 2.2), z0=z0, z1=z0+1.5, n=int(cl.shape[0]))

def corners(b):
    dx = np.array([b["L"]/2]*2 + [-b["L"]/2]*2); dy = np.array([b["W"]/2, -b["W"]/2, -b["W"]/2, b["W"]/2])
    cs, sn = np.cos(b["yaw"]), np.sin(b["yaw"]); fx = b["cx"]+dx*cs-dy*sn; fy = b["cy"]+dx*sn+dy*cs
    return np.c_[fx, fy, np.full(4, b["z0"])], np.c_[fx, fy, np.full(4, b["z1"])]

def to_px(xyz):
    opt = (L2C @ np.r_[xyz, 1])[:3]
    return None if opt[2] <= 0.2 else [float(ci["fx"]*opt[0]/opt[2]+ci["cx"]), float(ci["fy"]*opt[1]/opt[2]+ci["cy"])]


def persistence(cen_w, ti):
    """# of LOCAL neighbor frames (+/-PERSIST_K) with a candidate near this world spot.
    A moving car leaves the spot within a few frames (low); static clutter stays (high)."""
    n = 0
    for j in range(max(0, ti-PERSIST_K), min(len(ids), ti+PERSIST_K+1)):
        if j == ti: continue
        if any(np.hypot(c["cen_w"][0]-cen_w[0], c["cen_w"][1]-cen_w[1]) < PERSIST_R for c in cand_per[j]):
            n += 1
    return n


allobj = {}
for T in TARGETS:
    ti = idx_of[T]
    cands = cand_per[ti]
    kept = []
    for c in cands:
        per = persistence(c["cen_w"], ti)
        tag = "STATIC" if per > PERSIST_MAX else "moving"
        print(f"  pair {T} cand cen=({c['cen'][0]:5.1f},{c['cen'][1]:6.1f}) n{c['n']:3d} persist{per} -> {tag}", flush=True)
        if per <= PERSIST_MAX:
            kept.append(c)
    # NMS + density among moving survivors
    kept.sort(key=lambda c: -c["n"]); ded = []
    for c in kept:
        if any(np.hypot(c["cen"][0]-k["cen"][0], c["cen"][1]-k["cen"][1]) < 2.5 for k in ded): continue
        ded.append(c)
    if ded:
        nmax = max(c["n"] for c in ded); ded = [c for c in ded if c["n"] >= max(60, 0.4*nmax)]
    print(f"pair {T}: {len(cands)} cand -> {len(ded)} boxes after static-rejection+density", flush=True)
    lst = []
    for c in ded:
        b = box_from(c["pts"]); bo, tp = corners(b); px = [to_px(x) for x in list(bo)+list(tp)]; px = [p for p in px if p]
        bbox = None
        if len(px) >= 4:
            a = np.array(px); bbox = [float(a[:,0].min()), float(a[:,1].min()), float(a[:,0].max()), float(a[:,1].max())]
        box3d = dict(cx=b["cx"], cy=b["cy"], cz=(b["z0"]+b["z1"])/2, yaw=b["yaw"], length=b["L"], width=b["W"],
                     height=b["z1"]-b["z0"], reliable=True, size_plausible=True, length_fit=0.0, n_points=b["n"])
        lst.append(dict(label="car", source="radar_motion_rescue", score=0.6, color="lime", bbox_xyxy=bbox,
                        box3d=box3d, radar_confirmed=True,
                        forward=[b["cx"]-b["L"]/2, b["cx"]+b["L"]/2], lateral=[b["cy"]-b["W"]/2, b["cy"]+b["W"]/2]))
    allobj[str(T)] = lst
ANN = f"{RES}/annotations/rescue_v4.json"
json.dump(dict(objects=allobj), open(ANN, "w"), indent=1)
print(f"wrote {ANN} ({sum(len(v) for v in allobj.values())} boxes / {len(allobj)} pairs)", flush=True)
