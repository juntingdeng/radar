"""Diagnose WHERE the process loses the car at 14260 (data is fine per user).

Shows, in the LiDAR frame:
  - radar detections colored by the EGO-DOPPLER decision: kept-as-mover vs dropped-as-static
  - strong radar BEV returns (the actual strong areas)
  - LiDAR points
so we can see if the car's strong radar is being dropped by the ego-Doppler gate or
lost in clustering/density.
"""
import os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_pairs, load_calib, LidarSource, RES, H5, CFG
from detector import RadarDetector
from ego_doppler import fit_ego_velocity
sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _radar_setup, _radar_panel, _peaks_from_bev, _extract_frames, DEFAULT_COLOR_TOPIC

PAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 14260
cal = load_calib(); pairs = load_pairs()
ci = cal["color_intrinsics"]; R2L = np.array(cal["radar_to_lidar_auto"])
CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM = "/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11/camera/20260511_213724.bag"
CIDX = f"{RES}/camera_index.npz"

det = RadarDetector(processing="legacy")
d, _ = det.detect(pairs[PAIR]["radar_idx"])
Xd, Yd = RadarDetector.to_xy(d)
lf = R2L[0,0]*Xd+R2L[0,1]*Yd+R2L[0,3]; ll = R2L[1,0]*Xd+R2L[1,1]*Yd+R2L[1,3]   # lidar frame
ab, inl = fit_ego_velocity(d["az"], d["vel"])
azr = np.radians(-d["az"]); pred = np.cos(azr)*ab[0] + np.sin(azr)*ab[1]
mover = (~inl) & (np.abs(d["vel"] - pred) > 0.8)
print(f"detections {len(d)}  ego-inliers(static,dropped) {inl.sum()}  movers(kept) {mover.sum()}", flush=True)

# strong radar BEV returns
args = SimpleNamespace(radar_h5=H5, cfg_file=CFG, sync_csv=f"{RES}/sync_pairs.csv",
                       sync_summary=f"{RES}/sync_summary.json", lidar_pcap=None, lidar_metadata=None, radar_bev_res=0.1)
loader, Rmat = _radar_setup(args, cal, doppler_mode="all")
bev, la, fa = _radar_panel(loader, PAIR); pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.5)

src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); ml = np.linalg.norm(P[:, :2], axis=1) < 28
bag_ts = np.load(CIDX)["color_bag_ts_ns"]
img = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[PAIR]["color_idx"]], Path(CACHE), "color", bag_ts).get(pairs[PAIR]["color_idx"])
if img is not None and img.ndim == 1: img = img.reshape(ci["height"], ci["width"], 3)

fig, axs = plt.subplots(1, 2, figsize=(18, 8))
if img is not None: axs[0].imshow(img)
axs[0].axis("off"); axs[0].set_title(f"CAMERA {PAIR} (where is the car?)")

axs[1].scatter(P[ml, 1], P[ml, 0], s=0.4, c="0.85")
axs[1].scatter(pl, pf, s=70, marker="*", c="gold", edgecolors="orange", label="radar STRONG return", zorder=4)
axs[1].scatter(ll[inl], lf[inl], s=40, marker="x", c="silver", label="radar det: DROPPED as static (ego-inlier)", zorder=3)
axs[1].scatter(ll[mover], lf[mover], s=90, facecolors="none", edgecolors="red", linewidths=2,
               label="radar det: KEPT as mover", zorder=5)
# detections that are outliers but below margin (neither)
other = (~inl) & (~mover)
axs[1].scatter(ll[other], lf[other], s=40, marker="+", c="steelblue", label="outlier<margin (dropped)", zorder=3)
axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 26)
axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
axs[1].set_title(f"{PAIR}: radar strong returns + ego-Doppler keep/drop\n(is the car's strong return kept or dropped?)")
axs[1].legend(loc="upper right", fontsize=8)
out = f"{RES}/fusion/diag_" + str(PAIR) + "_proc.png"
plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}", flush=True)
