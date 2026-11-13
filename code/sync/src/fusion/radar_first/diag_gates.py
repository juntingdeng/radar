"""Per-cluster verdict of each generalizable gate (model/plane/tall/isolation) for a pair.
    python3 diag_gates.py 14630 14520
"""
import os, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
from common import load_pairs, load_calib, LidarSource, LOCAL, RES
import glare_bearing as G   # reuse constants + fit_ground_plane + ray + car_glare_blobs
sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC

PAIRS = [int(a) for a in sys.argv[1:]] or [14630]
CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
bag_ts = np.load(CIDX)["color_bag_ts_ns"]; src = LidarSource()
imgs = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[p]["color_idx"] for p in PAIRS], Path(CACHE), "color", bag_ts)

for PAIR in PAIRS:
    img = imgs[pairs[PAIR]["color_idx"]]
    if img.ndim == 1: img = img.reshape(H, W, 3)
    blobs = G.car_glare_blobs(img)
    P = src.xyz(pairs[PAIR]["lidar_idx"]); P = P[np.linalg.norm(P[:, :2], axis=1) < G.RMAX]
    coef = G.fit_ground_plane(P); Pha = G.height_above(P, coef)
    slab = P[(Pha > 0.3) & (Pha < 2.2)]; sb = np.degrees(np.arctan2(slab[:, 1], slab[:, 0]))
    print(f"\npair {PAIR}: plane coef a={coef[0]:+.3f} b={coef[1]:+.3f} c={coef[2]:+.2f}, {len(blobs)} blobs", flush=True)
    for b in blobs:
        b_lo, b_hi = sorted((G.ray(b["u0"], b["vb"])[0], G.ray(b["u1"], b["vb"])[0]))
        onb = (sb >= b_lo-G.BEARING_TOL) & (sb <= b_hi+G.BEARING_TOL); pts = slab[onb]
        if len(pts) < G.CAR_MIN_PTS: continue
        lb = DBSCAN(eps=1.0, min_samples=4).fit_predict(pts[:, :2])
        for c in set(lb.tolist()):
            if c == -1: continue
            cl = pts[lb == c]
            if len(cl) < G.CAR_MIN_PTS: continue
            cen = cl[:, :2].mean(0); X = cl[:, :2]-cen
            _, _, vt = np.linalg.svd(X, full_matrices=False); proj = X @ vt.T
            mj, mnr = float(np.ptp(proj[:, 0])), float(np.ptp(proj[:, 1]))
            hmin = float(G.height_above(cl, coef).min())
            x0, x1 = cl[:, 0].min()-0.4, cl[:, 0].max()+0.4; y0, y1 = cl[:, 1].min()-0.4, cl[:, 1].max()+0.4
            box = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
            tall = int((box & (Pha > G.TALL_H)).sum())
            m = 1.2; big = (P[:, 0] >= x0-m) & (P[:, 0] <= x1+m) & (P[:, 1] >= y0-m) & (P[:, 1] <= y1+m)
            ring = big & ~box & (Pha > 0.3) & (Pha < 2.2)
            a_cl = max((x1-x0)*(y1-y0), 1.0); a_ring = max((x1-x0+2*m)*(y1-y0+2*m)-a_cl, 1.0)
            iso = (ring.sum()/a_ring) / (len(cl)/a_cl)
            r = float(np.hypot(*cen))
            v = []
            if not (G.CAR_LEN[0] <= mj <= G.CAR_LEN[1] and G.CAR_WID[0] <= mnr <= G.CAR_WID[1]): v.append("MODEL")
            if hmin > G.GROUND_CLEAR: v.append("PLANE")
            if tall >= 8: v.append("TALL")
            if iso > G.ISO_RATIO: v.append("ISO")
            print(f"  r{r:5.1f} ({cen[0]:5.1f},{cen[1]:5.1f}) {mj:.1f}x{mnr:.1f} n{len(cl):3d} hmin{hmin:+.2f} tall{tall:3d} iso{iso:.2f} -> {'KEEP' if not v else 'reject:'+','.join(v)}", flush=True)
