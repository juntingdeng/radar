"""Night rescue: box moving cars from STRONG MOVING-Doppler radar + raw LiDAR.

Recipe (validated on 14280):
  1. strong MOVING-Doppler radar returns -> the approaching car, static bg notched out
  2. raw LiDAR points at that range/azimuth -> bypasses the cluster/mega-blob merge
  3. size-sanity + NMS + adaptive density gate -> keep only the dominant solid car
Runs on one or more pairs (argv), emits a combined annotations JSON for viz_labels.
    python3 rescue_v2.py 14270 14280 14290 14300
"""
import os, sys, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.cluster import DBSCAN
from common import load_pairs, load_calib, LidarSource, RES, H5, CFG

sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _ground_level, _radar_setup, _radar_panel, _peaks_from_bev

TARGETS = [int(a) for a in sys.argv[1:]] or [14280]
ROI = (0.0, 30.0, -16.0, 16.0); GPCT = 10.0; HBAND = (0.3, 2.2)
STRONG_PCT = 99.6; RANGE_TOL = 1.2; AZ_TOL = 12.0; MIN_PTS = 10

pairs = load_pairs(); cal = load_calib()
ci = cal["color_intrinsics"]; L2C = np.linalg.inv(np.array(cal["camera_to_lidar_auto"]))
args = SimpleNamespace(radar_h5=H5, cfg_file=CFG, sync_csv=f"{RES}/sync_pairs.csv",
                       sync_summary=f"{RES}/sync_summary.json", lidar_pcap=None, lidar_metadata=None, radar_bev_res=0.1)
loader, Rmat = _radar_setup(args, cal, doppler_mode="moving")
src = LidarSource()


def corners(b):
    dx = np.array([b["L"]/2]*2 + [-b["L"]/2]*2); dy = np.array([b["W"]/2, -b["W"]/2, -b["W"]/2, b["W"]/2])
    cs, sn = np.cos(b["yaw"]), np.sin(b["yaw"]); fx = b["cx"]+dx*cs-dy*sn; fy = b["cy"]+dx*sn+dy*cs
    return np.c_[fx, fy, np.full(4, b["z0"])], np.c_[fx, fy, np.full(4, b["z1"])]

def to_px(xyz):
    opt = (L2C @ np.r_[xyz, 1])[:3]
    return None if opt[2] <= 0.2 else [float(ci["fx"]*opt[0]/opt[2]+ci["cx"]), float(ci["fy"]*opt[1]/opt[2]+ci["cy"])]


def process(TARGET):
    bev, la, fa = _radar_panel(loader, TARGET)
    pf, pl = _peaks_from_bev(bev, la, fa, Rmat, STRONG_PCT)
    peak_rng = np.hypot(pf, pl); peak_az = np.degrees(np.arctan2(pl, pf))
    P = src.xyz(pairs[TARGET]["lidar_idx"])
    x0, x1, y0, y1 = ROI
    reg = P[(P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)]
    gz = _ground_level(reg, GPCT)
    slab = reg[(reg[:, 2] > gz + HBAND[0]) & (reg[:, 2] < gz + HBAND[1])]
    rng = np.hypot(slab[:, 0], slab[:, 1]); az = np.degrees(np.arctan2(slab[:, 1], slab[:, 0]))
    backed = np.zeros(len(slab), bool)
    for pr, pa in zip(peak_rng, peak_az):
        backed |= (np.abs(rng - pr) < RANGE_TOL) & (np.abs(((az - pa + 180) % 360) - 180) < AZ_TOL)
    rp = slab[backed]
    boxes = []
    if len(rp) >= MIN_PTS:
        lab = DBSCAN(eps=1.0, min_samples=6).fit_predict(rp[:, :2])
        for c in set(lab.tolist()):
            if c == -1: continue
            cl = rp[lab == c]
            if cl.shape[0] < MIN_PTS: continue
            xy = cl[:, :2]; cen = xy.mean(0); X = xy - cen
            _, _, vt = np.linalg.svd(X, full_matrices=False); yaw = float(np.arctan2(vt[0, 1], vt[0, 0]))
            proj = X @ vt.T; L = max(float(np.ptp(proj[:, 0])), 1.5); W = max(float(np.ptp(proj[:, 1])), 1.6)
            if L < 3.0: L = 4.3; yaw = float(np.arctan2(cen[1], cen[0]))
            z0 = float(cl[:, 2].min())
            boxes.append(dict(cx=float(cen[0]), cy=float(cen[1]), yaw=yaw, L=L, W=min(W, 2.2),
                              z0=z0, z1=z0+1.5, n=int(cl.shape[0]), rng=float(np.hypot(*cen))))
    raw = len(boxes)
    boxes = [b for b in boxes if 1.5 <= b["L"] <= 6.0 and b["W"] <= 3.0 and b["rng"] >= 4.0 and b["n"] >= 20]
    boxes.sort(key=lambda b: -b["n"]); kept = []
    for b in boxes:
        if any(np.hypot(b["cx"]-k["cx"], b["cy"]-k["cy"]) < 2.5 for k in kept): continue
        kept.append(b)
    if kept:
        nmax = max(b["n"] for b in kept)
        kept = [b for b in kept if b["n"] >= max(60, 0.4*nmax)]
    kept.sort(key=lambda b: b["rng"])
    print(f"pair {TARGET}: strong-moving {len(pf)}, backed-pts {len(rp)}, {raw} raw -> {len(kept)} boxes", flush=True)
    for b in kept:
        print(f"    rng {b['rng']:5.1f} cen=({b['cx']:5.1f},{b['cy']:6.1f}) L{b['L']:.1f} W{b['W']:.1f} n{b['n']}", flush=True)
    return kept


allobj = {}
for T in TARGETS:
    boxes = process(T); lst = []
    for b in boxes:
        bo, tp = corners(b); px = [to_px(x) for x in list(bo)+list(tp)]; px = [p for p in px if p]
        bbox = None
        if len(px) >= 4:
            a = np.array(px); bbox = [float(a[:,0].min()), float(a[:,1].min()), float(a[:,0].max()), float(a[:,1].max())]
        box3d = dict(cx=b["cx"], cy=b["cy"], cz=(b["z0"]+b["z1"])/2, yaw=b["yaw"], length=b["L"], width=b["W"],
                     height=b["z1"]-b["z0"], reliable=True, size_plausible=True, length_fit=0.0, n_points=b["n"])
        lst.append(dict(label="car", source="radar_strong_rescue", score=0.5, color="lime",
                        bbox_xyxy=bbox, box3d=box3d, radar_confirmed=True,
                        forward=[b["cx"]-b["L"]/2, b["cx"]+b["L"]/2], lateral=[b["cy"]-b["W"]/2, b["cy"]+b["W"]/2]))
    allobj[str(T)] = lst
ANN = f"{RES}/annotations/rescue_neighbors.json"
json.dump(dict(objects=allobj), open(ANN, "w"), indent=1)
print(f"wrote {ANN} ({sum(len(v) for v in allobj.values())} boxes over {len(allobj)} pairs)", flush=True)
