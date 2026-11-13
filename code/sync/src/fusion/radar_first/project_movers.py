"""Project the radar-first movers onto the CAMERA image + LiDAR BEV.

Mirror of the old fusion_step13 figure, but driven by the validated pipeline
(odometry -> ego-Doppler -> Kalman track -> LiDAR-dynamic + range-rate gates).
Green = radar-raised mover proposals; lifted radar(range,az) -> LiDAR -> camera.
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_pairs, load_calib, LOCAL, RES
from tracker import track_movers

sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC

CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAMERA_BAG = f"{LOCAL}/camera/20260511_213724.bag"
CAMERA_INDEX = f"{RES}/camera_index.npz"
P0, NP = 13850, 100

pairs = load_pairs()
ids = [p for p in range(P0, P0 + NP) if p in pairs]
R = track_movers(ids, pairs)
movers, poses, scans = R["movers"], R["poses"], R["scans"]
idx_of = {p: i for i, p in enumerate(ids)}
cal = load_calib()
C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]


def mover_lidar_xy(t, fi):
    h = np.array(t.hist); j = int(np.argmin(np.abs(h[:, 0] - fi)))
    if abs(h[j, 0] - fi) > 2:
        return None
    return (np.linalg.inv(poses[fi]) @ np.r_[h[j, 1], h[j, 2], 0, 1])[:2]


def in_fov(lf, ll):
    return lf > 0.5 and abs(np.degrees(np.arctan2(ll, lf))) < 28


# pick the frame with the most movers inside the camera FOV
best = (ids[len(ids)//2], -1, len(ids)//2)
for p in ids:
    fi = idx_of[p]
    cnt = sum(1 for t in movers if (xy := mover_lidar_xy(t, fi)) is not None and in_fov(*xy))
    if cnt > best[1]:
        best = (p, cnt, fi)
FP, cnt, fi = best
print(f"chosen frame: pair {FP} (index {fi}), {cnt} movers in camera FOV", flush=True)

# extract that single camera frame (seek via camera_index)
bag_ts = np.load(CAMERA_INDEX)["color_bag_ts_ns"]
cidx = pairs[FP]["color_idx"]
cmap = _extract_frames(CAMERA_BAG, DEFAULT_COLOR_TOPIC, [cidx], Path(CACHE), "color", bag_ts)
img = cmap.get(cidx)
if img is not None and img.ndim == 1:
    img = img.reshape(ci["height"], ci["width"], 3)

Pl = scans[fi]; tab = plt.cm.tab10
fig, axs = plt.subplots(1, 2, figsize=(19, 7.5))

# --- camera panel ---
if img is not None:
    axs[0].imshow(img)
for i, t in enumerate(movers):
    xy = mover_lidar_xy(t, fi)
    if xy is None or not in_fov(*xy):
        continue
    d = np.hypot(Pl[:, 0]-xy[0], Pl[:, 1]-xy[1]); near = Pl[d < 1.8]
    z = float(np.median(near[:, 2])) if len(near) >= 3 else -0.5
    opt = (L2C @ np.r_[xy, z, 1])[:3]; Z = opt[2]
    if Z <= 0.3:
        continue
    u = ci["fx"]*opt[0]/Z + ci["cx"]; v = ci["fy"]*opt[1]/Z + ci["cy"]
    axs[0].scatter([u], [v], s=320, facecolors="none", edgecolors="lime", linewidths=3)
    axs[0].text(u+10, v, f"{t.speed:.1f} m/s", color="lime", fontsize=11, fontweight="bold")
axs[0].set_xlim(0, ci["width"]); axs[0].set_ylim(ci["height"], 0); axs[0].axis("off")
axs[0].set_title(f"Radar-first movers on CAMERA (pair {FP})\ngreen = radar-raised proposal")

# --- lidar BEV panel ---
ml = np.linalg.norm(Pl[:, :2], axis=1) < 30
axs[1].scatter(Pl[ml, 1], Pl[ml, 0], s=0.5, c="0.75")
invp = np.linalg.inv(poses[fi])
for i, t in enumerate(movers):
    h = np.array(t.hist)
    LL = np.array([(invp @ np.r_[x, y, 0, 1])[:2] for _, x, y in h[:, :3]])
    c = tab(i % 10)
    axs[1].plot(LL[:, 1], LL[:, 0], "-", color=c, lw=2.2)
    axs[1].annotate("", xy=(LL[-1, 1], LL[-1, 0]), xytext=(LL[0, 1], LL[0, 0]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2))
    axs[1].text(LL[-1, 1], LL[-1, 0]+0.4, f"{t.speed:.1f}", fontsize=9, color=c, fontweight="bold")
axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 28)
axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
axs[1].set_title(f"LiDAR BEV + radar-first mover tracks\n{len(movers)} movers (all gates passed)")

plt.tight_layout()
os.makedirs(f"{RES}/fusion", exist_ok=True)
out = f"{RES}/fusion/radar_first_project_cam.png"
plt.savefig(out, dpi=115, bbox_inches="tight")
print(f"saved {out}", flush=True)
