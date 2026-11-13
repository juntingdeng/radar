"""Where is the car's LiDAR along the glare bearing? (why the glare+radar box is empty)

Shows the LiDAR car-height clusters along the glare bearing with their size/density, so
we can see if the car has a dense cluster (and at what range) vs the box's radar range.
"""
import os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_pairs, load_calib, LidarSource, LOCAL, RES, H5, CFG
sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _radar_setup, _radar_panel, _peaks_from_bev, _extract_frames, DEFAULT_COLOR_TOPIC

PAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 14560
CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
BRIGHT = 205; VBAND = (0.30, 0.78); MIN_BLOB = 60

pairs = load_pairs(); cal = load_calib()
ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; Hh, Wd = ci["height"], ci["width"]
R2L = np.array(cal["radar_to_lidar_auto"])

# glare bearing
bag_ts = np.load(CIDX)["color_bag_ts_ns"]
img = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[PAIR]["color_idx"]], Path(CACHE), "color", bag_ts).get(pairs[PAIR]["color_idx"])
if img.ndim == 1: img = img.reshape(Hh, Wd, 3)
bright = (img.min(2) > BRIGHT); bright[:int(Hh*VBAND[0])] = False; bright[int(Hh*VBAND[1]):] = False
lab, nl = ndimage.label(bright)
cols = [np.where(lab == i)[1] for i in range(1, nl+1)]; cols = [c for c in cols if len(c) >= MIN_BLOB]
allc = np.concatenate(cols) if cols else np.array([cx])
umin, umax, vrow = allc.min(), allc.max(), Hh*0.55
def bof(u): d = np.array([(u-cx)/fx, 0.0, 1.0]); d /= np.linalg.norm(d); dl = Rc2l @ d; return np.degrees(np.arctan2(dl[1], dl[0]))
b_lo, b_hi = sorted((bof(umin), bof(umax))); bearing = 0.5*(b_lo+b_hi)
print(f"pair {PAIR}: glare cols [{umin},{umax}] -> bearing window [{b_lo:+.1f},{b_hi:+.1f}] deg", flush=True)

# radar strong on bearing
args = SimpleNamespace(radar_h5=H5, cfg_file=CFG, sync_csv=f"{RES}/sync_pairs.csv",
                       sync_summary=f"{RES}/sync_summary.json", lidar_pcap=None, lidar_metadata=None, radar_bev_res=0.1)
loader, Rmat = _radar_setup(args, cal, doppler_mode="all"); bev, la, fa = _radar_panel(loader, PAIR)
pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.0); pbear = np.degrees(np.arctan2(pl, pf)); prng = np.hypot(pf, pl)
onr = (pbear >= b_lo-6) & (pbear <= b_hi+6)
rr = np.sort(prng[onr]) if onr.any() else np.array([])
print(f"  radar ranges in window: {np.round(rr, 1)[:20]}", flush=True)

# lidar clusters in the glare WINDOW, out to 45m (far cars)
RMAX = 45.0
src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < RMAX
gz = np.percentile(P[nm][:, 2], 10)
slab = P[nm & (P[:, 2] > gz+0.3) & (P[:, 2] < gz+2.2)]
sb = np.degrees(np.arctan2(slab[:, 1], slab[:, 0])); onb = (sb >= b_lo-6) & (sb <= b_hi+6)
lb = slab[onb]
print(f"  lidar car-height pts in window (<{RMAX:.0f}m): {len(lb)}", flush=True)
clusters = []
if len(lb) >= 6:
    L = DBSCAN(eps=1.2, min_samples=4).fit_predict(lb[:, :2])
    for c in set(L.tolist()):
        if c == -1: continue
        cl = lb[L == c]; cen = cl[:, :2].mean(0); r = float(np.hypot(*cen))
        fpr = np.ptp(cl[:, :2], axis=0); mx, mn = max(fpr), min(fpr)
        carish = 1.5 <= mx <= 6.5 and mn <= 3.5
        clusters.append((r, len(cl), cen, fpr))
        print(f"    lidar cluster r={r:4.1f}m n={len(cl):3d} footprint {fpr[0]:.1f}x{fpr[1]:.1f} {'CAR' if carish else ''}", flush=True)

fig, axs = plt.subplots(1, 2, figsize=(18, 8))
axs[0].imshow(img); axs[0].axvline(umin, color="lime", lw=1); axs[0].axvline(umax, color="lime", lw=1)
axs[0].axis("off"); axs[0].set_title(f"CAMERA {PAIR}: glare window (lime)")
axs[1].scatter(P[nm, 1], P[nm, 0], s=0.5, c="0.8")
axs[1].scatter(pl, pf, s=12, marker="x", c="purple", label="radar strong")
axs[1].scatter(lb[:, 1], lb[:, 0], s=10, c="orange", label="lidar car-ht in window")
for bb in (b_lo, b_hi):
    r = np.linspace(0, RMAX, 2); axs[1].plot(r*np.sin(np.radians(bb)), r*np.cos(np.radians(bb)), "lime", lw=1, ls="--")
for rr_, n, cen, fpr in clusters:
    axs[1].text(cen[1], cen[0], f"r{rr_:.0f} n{n}", color="red", fontsize=9)
axs[1].set_aspect("equal"); axs[1].set_xlim(22, -8); axs[1].set_ylim(0, RMAX)
axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)"); axs[1].legend(loc="upper right", fontsize=8)
axs[1].set_title(f"{PAIR}: lidar clusters in glare window (how many cars?)")
out = f"{RES}/fusion/glare/diag_glare_lidar_{PAIR}.png"
plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); print(f"saved {out}", flush=True)
