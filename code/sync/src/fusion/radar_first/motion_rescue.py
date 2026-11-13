"""Motion-matched night rescue: capture moving cars, reject static background.

For each LiDAR cluster at a target frame:
  - world velocity from cross-frame tracking (ego-compensated odometry)
  - FILTER: |world velocity| < MOVE_MIN  -> static background (wall/pole/ground) -> drop
  - CAPTURE: cluster near a MOVING radar return whose Doppler (alias-aware) matches the
    cluster's line-of-sight velocity -> it's the real moving car -> raise box
This resolves the ±azimuth arc ambiguity by MOTION, not flaky bearing.
    python3 motion_rescue.py 14260 14270 14280 14290 14300
"""
import os, sys, json
import numpy as np
from sklearn.cluster import DBSCAN
from common import load_pairs, load_calib, LidarSource, RES, pose_apply
from detector import RadarDetector
from odometry import odometry_scan2map
sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _ground_level

TARGETS = [int(a) for a in sys.argv[1:]] or [14280]
MARGIN = 5; DV = 3
ROI = (0.0, 30.0, -16.0, 16.0); GPCT = 10.0; HBAND = (0.3, 2.2)
MOVE_MIN = 0.6              # world speed below this = static background -> filtered
RANGE_TOL = 1.5; AZ_TOL = 16.0; VMAX = 2.44; VEL_TOL = 1.3
MIN_PTS = 12

pairs = load_pairs(); cal = load_calib()
ci = cal["color_intrinsics"]; L2C = np.linalg.inv(np.array(cal["camera_to_lidar_auto"]))
R2L = np.array(cal["radar_to_lidar_auto"])

W0, W1 = min(TARGETS) - MARGIN, max(TARGETS) + MARGIN
ids = [p for p in range(W0, W1 + 1) if p in pairs]
times = np.array([pairs[p]["radar_t"] for p in ids]); times -= times[0]
idx_of = {p: i for i, p in enumerate(ids)}

src = LidarSource()
scans = [src.xyz(pairs[p]["lidar_idx"]) for p in ids]
poses, _ = odometry_scan2map(scans)
traj = np.array([p[:3, 3][:2] for p in poses]); dtv = np.r_[0.1, np.diff(times)]; dtv[dtv <= 0] = 0.1
ego_v = np.zeros((len(poses), 2)); ego_v[1:] = np.diff(traj, axis=0) / dtv[1:, None]; ego_v[0] = ego_v[1]


def clusters_of(P, pose):
    x0, x1, y0, y1 = ROI
    reg = P[(P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)]
    gz = _ground_level(reg, GPCT)
    slab = reg[(reg[:, 2] > gz + HBAND[0]) & (reg[:, 2] < gz + HBAND[1])]
    if len(slab) < MIN_PTS: return []
    lab = DBSCAN(eps=1.0, min_samples=6).fit_predict(slab[:, :2]); out = []
    for c in set(lab.tolist()):
        if c == -1: continue
        cl = slab[lab == c]
        if cl.shape[0] < MIN_PTS: continue
        cen = cl[:, :2].mean(0)
        out.append(dict(pts=cl, cen=cen, cen_w=pose_apply(pose, cen[None, :])[0]))
    return out

cl_per = [clusters_of(scans[i], poses[i]) for i in range(len(ids))]


def world_vel(cen_w, i):
    def nearest(j):
        if j < 0 or j >= len(cl_per) or not cl_per[j]: return None
        cs = np.array([c["cen_w"] for c in cl_per[j]])
        d = np.hypot(cs[:, 0]-cen_w[0], cs[:, 1]-cen_w[1]); k = int(np.argmin(d))
        return cl_per[j][k]["cen_w"] if d[k] < 2.5 else None
    a, b = nearest(i-DV), nearest(i+DV)
    if a is None or b is None: return None
    dt = times[min(i+DV, len(times)-1)] - times[max(i-DV, 0)]
    return (b - a) / dt if dt > 0 else None


def alias_match(exp, measured):
    for k in range(-3, 4):
        for s in (1, -1):
            if abs(exp - s*(measured + k*2*VMAX)) < VEL_TOL:
                return True
    return False


det = RadarDetector(processing="ti")   # moving-target radar detections (range,az,vel)


def box_from(cl):
    xy = cl[:, :2]; cen = xy.mean(0); X = xy - cen
    _, _, vt = np.linalg.svd(X, full_matrices=False); yaw = float(np.arctan2(vt[0, 1], vt[0, 0]))
    proj = X @ vt.T; L = max(float(np.ptp(proj[:, 0])), 1.5); W = max(float(np.ptp(proj[:, 1])), 1.6)
    if L < 3.0: L = 4.3; yaw = float(np.arctan2(cen[1], cen[0]))
    z0 = float(cl[:, 2].min())
    return dict(cx=float(cen[0]), cy=float(cen[1]), yaw=yaw, L=min(L, 6.0), W=min(W, 2.2), z0=z0, z1=z0+1.5)


def corners(b):
    dx = np.array([b["L"]/2]*2 + [-b["L"]/2]*2); dy = np.array([b["W"]/2, -b["W"]/2, -b["W"]/2, b["W"]/2])
    cs, sn = np.cos(b["yaw"]), np.sin(b["yaw"]); fx = b["cx"]+dx*cs-dy*sn; fy = b["cy"]+dx*sn+dy*cs
    return np.c_[fx, fy, np.full(4, b["z0"])], np.c_[fx, fy, np.full(4, b["z1"])]

def to_px(xyz):
    opt = (L2C @ np.r_[xyz, 1])[:3]
    return None if opt[2] <= 0.2 else [float(ci["fx"]*opt[0]/opt[2]+ci["cx"]), float(ci["fy"]*opt[1]/opt[2]+ci["cy"])]


allobj = {}
for T in TARGETS:
    ti = idx_of[T]
    d, _ = det.detect(pairs[T]["radar_idx"], min_abs_vel=0.3)
    if len(d):
        Xd, Yd = RadarDetector.to_xy(d)
        lf = R2L[0,0]*Xd+R2L[0,1]*Yd+R2L[0,3]; ll = R2L[1,0]*Xd+R2L[1,1]*Yd+R2L[1,3]
        drng = np.hypot(lf, ll); dang = np.degrees(np.arctan2(ll, lf)); dvel = d["vel"]
    else:
        drng = dang = dvel = np.array([])
    boxes = []; n_static = n_nobk = n_nomatch = 0
    for cl in cl_per[ti]:
        v = world_vel(cl["cen_w"], ti)
        if v is None or np.hypot(*v) < MOVE_MIN:
            n_static += 1; continue                      # static background filtered
        rng = float(np.hypot(*cl["cen"])); ang = float(np.degrees(np.arctan2(cl["cen"][1], cl["cen"][0])))
        near = (np.abs(drng - rng) < RANGE_TOL) & (np.abs(((dang-ang+180) % 360)-180) < AZ_TOL) if len(drng) else np.array([], bool)
        if not near.any():
            n_nobk += 1; continue                        # no moving-radar backing
        egp = poses[ti][:3, 3][:2]; u = cl["cen_w"] - egp; u = u/(np.linalg.norm(u)+1e-6)
        exp = float((v - ego_v[ti]) @ u)
        if not any(alias_match(exp, mv) for mv in dvel[near]):
            n_nomatch += 1; continue                     # Doppler doesn't match cluster motion
        b = box_from(cl["pts"]); b["speed"] = float(np.hypot(*v)); boxes.append(b)
    # light NMS
    boxes.sort(key=lambda b: -b["speed"]); kept = []
    for b in boxes:
        if any(np.hypot(b["cx"]-k["cx"], b["cy"]-k["cy"]) < 2.5 for k in kept): continue
        kept.append(b)
    print(f"pair {T}: {len(cl_per[ti])} clusters -> static-filtered {n_static}, no-radar {n_nobk}, "
          f"no-velmatch {n_nomatch} -> {len(kept)} boxes", flush=True)
    for b in kept:
        print(f"    rng {np.hypot(b['cx'],b['cy']):5.1f} cen=({b['cx']:5.1f},{b['cy']:6.1f}) spd {b['speed']:.1f} m/s", flush=True)
    lst = []
    for b in kept:
        bo, tp = corners(b); px = [to_px(x) for x in list(bo)+list(tp)]; px = [p for p in px if p]
        bbox = None
        if len(px) >= 4:
            a = np.array(px); bbox = [float(a[:,0].min()), float(a[:,1].min()), float(a[:,0].max()), float(a[:,1].max())]
        box3d = dict(cx=b["cx"], cy=b["cy"], cz=(b["z0"]+b["z1"])/2, yaw=b["yaw"], length=b["L"], width=b["W"],
                     height=b["z1"]-b["z0"], reliable=True, size_plausible=True, length_fit=0.0, n_points=0)
        lst.append(dict(label="car", source="radar_motion_rescue", score=0.6, color="lime",
                        bbox_xyxy=bbox, box3d=box3d, radar_confirmed=True,
                        forward=[b["cx"]-b["L"]/2, b["cx"]+b["L"]/2], lateral=[b["cy"]-b["W"]/2, b["cy"]+b["W"]/2]))
    allobj[str(T)] = lst
ANN = f"{RES}/annotations/motion_rescue.json"
json.dump(dict(objects=allobj), open(ANN, "w"), indent=1)
print(f"wrote {ANN} ({sum(len(v) for v in allobj.values())} boxes / {len(allobj)} pairs)", flush=True)
