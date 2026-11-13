"""Diagnose pair 14280: glare-blind camera, LiDAR sees a dense FRONT face (no L-shape
-> fails sizeok), radar has a strong arc. Show exactly why the car is missed and what
would recover it (relaxed size + radar-range support)."""
import os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_pairs, load_calib, LidarSource, LOCAL, RES, H5, CFG

sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import (_cluster_bev, _drop_ground, _ground_level,
                           _radar_setup, _radar_panel, _peaks_from_bev,
                           _extract_frames, DEFAULT_COLOR_TOPIC)

PAIR = 14280
CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAMERA_BAG = f"{LOCAL}/camera/20260511_213724.bag"; CAMERA_INDEX = f"{RES}/camera_index.npz"
ROI = (0.0, 30.0, -16.0, 16.0); GPCT, MINH = 10.0, 0.3
# current pipeline sizeok gate
MAJ = (2.0, 6.5); MIN = (1.2, 3.5); HGT = (0.5, 2.4)

pairs = load_pairs(); cal = load_calib()
C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]

# --- lidar clusters ---
src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"])
x0, x1, y0, y1 = ROI
reg = P[(P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)]
gz = _ground_level(reg, GPCT)
obj = _drop_ground(reg, GPCT, MINH)
labels = _cluster_bev(obj[:, :2], ROI)

# --- radar: strong-return peaks (arc) in lidar frame ---
args = SimpleNamespace(radar_h5=H5, cfg_file=CFG, sync_csv=f"{RES}/sync_pairs.csv",
                       sync_summary=f"{RES}/sync_summary.json", lidar_pcap=None, lidar_metadata=None,
                       radar_bev_res=0.1)
loader, Rmat = _radar_setup(args, cal, doppler_mode="all", processing="legacy")
bev, la, fa = _radar_panel(loader, PAIR)
pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.0)

rows = []
clusters = []
for lb in np.unique(labels):
    cl = obj[labels == lb]
    if cl.shape[0] < 8:
        continue
    L = float(np.ptp(cl[:, 0])); W = float(np.ptp(cl[:, 1])); H = float(np.ptp(cl[:, 2]))
    mx, mn = max(L, W), min(L, W)
    sizeok = (MAJ[0] <= mx <= MAJ[1] and MIN[0] <= mn <= MIN[1] and HGT[0] <= H <= HGT[1])
    cen = cl[:, :2].mean(0)
    rng = float(np.hypot(*cen))
    radar_xy = bool(len(pf) and (np.hypot(pf - cen[0], pl - cen[1]) < 2.5 + mx/2).any())
    radar_rng = bool(len(pf) and (np.abs(np.hypot(pf, pl) - rng) < 1.5).any())  # radar at same RANGE
    clusters.append((cl, sizeok, cen, L, W, H, radar_xy, radar_rng))
    if cl.shape[0] >= 12:
        rows.append((int(cl.shape[0]), cen[0], cen[1], L, W, H, sizeok, radar_xy, radar_rng))

print(f"pair {PAIR}: {len(clusters)} clusters; radar peaks {len(pf)}", flush=True)
print(" n   fwd    lat    L    W    H   sizeok radarXY radarRNG", flush=True)
for r in sorted(rows, key=lambda z: z[1]):
    print(f"{r[0]:4d} {r[1]:5.1f} {r[2]:6.1f} {r[3]:4.1f} {r[4]:4.1f} {r[5]:4.1f}   "
          f"{str(r[6]):5s}  {str(r[7]):5s}  {str(r[8]):5s}", flush=True)

# camera frame
bag_ts = np.load(CAMERA_INDEX)["color_bag_ts_ns"]; cidx = pairs[PAIR]["color_idx"]
img = _extract_frames(CAMERA_BAG, DEFAULT_COLOR_TOPIC, [cidx], Path(CACHE), "color", bag_ts).get(cidx)
if img is not None and img.ndim == 1:
    img = img.reshape(ci["height"], ci["width"], 3)

fig, axs = plt.subplots(1, 2, figsize=(19, 8))
if img is not None:
    axs[0].imshow(img)
axs[0].axis("off"); axs[0].set_title(f"CAMERA pair {PAIR} (glare / night)")

ml = np.linalg.norm(P[:, :2], axis=1) < 30
axs[1].scatter(P[ml, 1], P[ml, 0], s=0.4, c="0.85")
for cl, sizeok, cen, L, W, H, rxy, rrng in clusters:
    if cl.shape[0] < 12:
        continue
    col = "limegreen" if sizeok else "red"
    axs[1].scatter(cl[:, 1], cl[:, 0], s=6, c=col)
    tag = f"L{L:.1f} W{W:.1f}" + ("  R-arc" if rrng else "")
    axs[1].text(cen[1], cen[0] + 0.4, tag, fontsize=7, color=col)
axs[1].scatter(pl, pf, s=22, marker="x", c="purple", label="radar peak (arc)")
axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 28)
axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
axs[1].set_title("LiDAR clusters: green=sizeok PASS, red=FAIL (missed)\n+ radar peaks (x)")
axs[1].legend(loc="upper right", fontsize=8)
plt.tight_layout()
out = f"{RES}/fusion/diag_14280.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}", flush=True)
