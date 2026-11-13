"""Diagnostic: do RAW radar detections land on the LiDAR car clusters?

Single frame, LiDAR frame, NO odometry / NO tracking -- isolates whether radar
proposals actually coincide with the cars (then the fix is association+boxing) or
not (then it's calibration/detection). This is the honest test of radar-first.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from common import load_pairs, load_calib, LidarSource, RES
from detector import RadarDetector
from ego_doppler import fit_ego_velocity

PAIR = 13890
pairs = load_pairs()
R2L = np.array(load_calib()["radar_to_lidar_auto"])

# --- radar: all detections + ego-Doppler movers, lifted to LiDAR frame ---
det = RadarDetector(processing="legacy")
d, _ = det.detect(pairs[PAIR]["radar_idx"])
azr = np.radians(-d["az"]); Xr = d["range"]*np.cos(azr); Yr = d["range"]*np.sin(azr)
lf = R2L[0,0]*Xr + R2L[0,1]*Yr + R2L[0,3]
ll = R2L[1,0]*Xr + R2L[1,1]*Yr + R2L[1,3]
ab, inl = fit_ego_velocity(d["az"], d["vel"])
if ab is not None:
    pred = np.cos(azr)*ab[0] + np.sin(azr)*ab[1]
    mover = (~inl) & (np.abs(d["vel"] - pred) > 0.8)
else:
    mover = np.zeros(len(d), bool)

# also the production-style MOVING detector ('ti' = zero-Doppler notch) for comparison
det_ti = RadarDetector(processing="ti")
dt, _ = det_ti.detect(pairs[PAIR]["radar_idx"], min_abs_vel=0.3)
azt = np.radians(-dt["az"]); Xt = dt["range"]*np.cos(azt); Yt = dt["range"]*np.sin(azt)
lft = R2L[0,0]*Xt + R2L[0,1]*Yt + R2L[0,3]
llt = R2L[1,0]*Xt + R2L[1,1]*Yt + R2L[1,3]

# --- lidar: near-range, above-ground, car-sized clusters ---
src = LidarSource()
P = src.xyz(pairs[PAIR]["lidar_idx"])
near = (np.linalg.norm(P[:, :2], axis=1) < 30)
gz = np.percentile(P[near][:, 2], 3)
obj = P[near & (P[:, 2] > gz + 0.3) & (P[:, 2] < gz + 2.2)]
clusters = []
if len(obj) > 30:
    lab = DBSCAN(eps=0.7, min_samples=8).fit_predict(obj[:, :2])
    for c in set(lab):
        if c == -1: continue
        pts = obj[lab == c][:, :2]
        if not (15 <= len(pts) <= 3000): continue
        fp = (pts.max(0) - pts.min(0))
        if 0.8 <= max(fp) <= 7 and min(fp) <= 4:         # car / object sized
            clusters.append(pts)

fig, ax = plt.subplots(figsize=(11, 11))
m = np.linalg.norm(P[:, :2], axis=1) < 30
ax.scatter(P[m, 1], P[m, 0], s=0.4, c="0.8", label="lidar")
for i, pts in enumerate(clusters):
    ax.scatter(pts[:, 1], pts[:, 0], s=6, label="lidar car-cluster" if i == 0 else None)
    cx, cy = pts.mean(0)
    ax.add_patch(plt.Rectangle((pts[:,1].min(), pts[:,0].min()),
                 np.ptp(pts[:,1]), np.ptp(pts[:,0]), fill=False, ec="k", lw=1.2))
ax.scatter(ll, lf, s=18, c="silver", edgecolors="gray", label="radar det (all)", zorder=3)
ax.scatter(llt, lft, s=55, marker="s", facecolors="none", edgecolors="blue",
           label="radar MOVING ('ti' notch)", zorder=4)
if mover.any():
    ax.scatter(ll[mover], lf[mover], s=90, facecolors="none", edgecolors="red",
               linewidths=2, label="radar mover (ego-Doppler)", zorder=5)
ax.set_aspect("equal"); ax.set_xlim(15, -15); ax.set_ylim(0, 28)
ax.set_xlabel("lateral Y (m)"); ax.set_ylabel("forward X (m)")
ax.set_title(f"pair {PAIR}: do radar detections land on the LiDAR cars?\n"
             f"{len(clusters)} lidar clusters · {len(d)} radar det · {int(mover.sum())} ego-movers · {len(dt)} 'ti'-moving")
ax.legend(loc="upper right", fontsize=8)
out = f"{RES}/fusion/diag_radar_on_lidar.png"
plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"clusters {len(clusters)}; radar det {len(d)}; ego-movers {int(mover.sum())}; ti-moving {len(dt)}", flush=True)
print(f"saved {out}", flush=True)
