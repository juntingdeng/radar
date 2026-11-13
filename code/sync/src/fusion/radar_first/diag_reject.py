"""Why is the clear L-cluster rejected? For a pair, show lidar car-height clusters in a WIDE cone
around the glare bearing and each gate's verdict (footprint / ground-touch / tall-above / range).
    python3 diag_reject.py 14610
"""
import os, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_pairs, load_calib, LidarSource, LOCAL, RES
sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC

PAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 14610
CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
BRIGHT = 205; VBAND = (0.20, 0.85); MIN_BLOB = 60; RMAX = 45.0; CONE = 15.0
cal = load_calib(); ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; H, W = ci["height"], ci["width"]
pairs = load_pairs(); bag_ts = np.load(CIDX)["color_bag_ts_ns"]
img = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[PAIR]["color_idx"]], Path(CACHE), "color", bag_ts)[pairs[PAIR]["color_idx"]]
if img.ndim == 1: img = img.reshape(H, W, 3)

bright = (img.min(2) > BRIGHT); bright[:int(H*VBAND[0])] = False; bright[int(H*VBAND[1]):] = False
lab, n = ndimage.label(bright); bl = [np.where(lab == i)[1] for i in range(1, n+1)]
bl = [c for c in bl if len(c) >= MIN_BLOB]
u = float(np.concatenate(bl).mean()) if bl else cx
d = np.array([(u-cx)/fx, 0.0, 1.0]); d /= np.linalg.norm(d); dl = Rc2l @ d
bearing = np.degrees(np.arctan2(dl[1], dl[0]))
print(f"pair {PAIR}: glare bearing {bearing:+.1f} deg (search cone +-{CONE})", flush=True)

src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < RMAX
gz = float(np.percentile(P[nm][:, 2], 10))
slab = P[nm & (P[:, 2] > gz+0.3) & (P[:, 2] < gz+2.2)]
sb = np.degrees(np.arctan2(slab[:, 1], slab[:, 0]))
onc = np.abs(((sb - bearing + 180) % 360) - 180) < CONE
pts = slab[onc]
print(f"  car-height pts in cone: {len(pts)}", flush=True)
clusters = []
if len(pts) >= 6:
    L = DBSCAN(eps=1.0, min_samples=4).fit_predict(pts[:, :2])
    for c in set(L.tolist()):
        if c == -1: continue
        cl = pts[L == c]
        if len(cl) < 6: continue
        cen = cl[:, :2].mean(0); X = cl[:, :2]-cen
        _, _, vt = np.linalg.svd(X, full_matrices=False); proj = X @ vt.T
        mj, mnr = float(np.ptp(proj[:, 0])), float(np.ptp(proj[:, 1]))
        r = float(np.hypot(*cen)); be = np.degrees(np.arctan2(cen[1], cen[0]))
        x0, x1 = cl[:, 0].min()-0.4, cl[:, 0].max()+0.4; y0, y1 = cl[:, 1].min()-0.4, cl[:, 1].max()+0.4
        above = int(((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1) & (P[:, 2] > gz+2.6)).sum())
        shape_ok = (mnr <= 3.5) and (1.0 <= mj <= 7.5)
        grnd_ok = float(cl[:, 2].min()) <= gz+0.8
        tall_ok = above < 8
        inwin = abs(((be-bearing+180) % 360)-180) < 5.0
        verdict = "KEPT" if (shape_ok and grnd_ok and tall_ok and inwin) else "reject:" + ",".join(
            ([] if shape_ok else ["shape"]) + ([] if grnd_ok else ["ground"]) + ([] if tall_ok else ["tall"]) + ([] if inwin else ["offbearing(>5deg)"]))
        clusters.append((cen, r, be, cl, verdict))
        print(f"    cl r{r:5.1f} bearing{be:+6.1f} n{len(cl):3d} fp {mj:.1f}x{mnr:.1f} z0{cl[:,2].min()-gz:+.2f} above{above:3d} -> {verdict}", flush=True)

fig, axs = plt.subplots(1, 2, figsize=(17, 8))
axs[0].imshow(img); axs[0].axvline(u, color="lime"); axs[0].axis("off"); axs[0].set_title(f"{PAIR} glare u={u:.0f}")
axs[1].scatter(P[nm, 1], P[nm, 0], s=0.5, c="0.8")
for bb, st in ((bearing-5, "lime"), (bearing+5, "lime"), (bearing-CONE, "orange"), (bearing+CONE, "orange")):
    rr = np.linspace(0, RMAX, 2); axs[1].plot(rr*np.sin(np.radians(bb)), rr*np.cos(np.radians(bb)), st, lw=0.8, ls="--")
for cen, r, be, cl, verdict in clusters:
    col = "green" if verdict == "KEPT" else "red"
    axs[1].scatter(cl[:, 1], cl[:, 0], s=8, c=col)
    axs[1].text(cen[1], cen[0], verdict.split(":")[0], color=col, fontsize=8)
axs[1].set_aspect("equal"); axs[1].set_xlim(24, -8); axs[1].set_ylim(0, RMAX)
axs[1].set_xlabel("lateral Y"); axs[1].set_ylabel("forward X")
axs[1].set_title(f"{PAIR}: clusters in cone (green=kept, red=rejected); lime=+-5 window, orange=+-{CONE}")
out = f"{RES}/fusion/night_label/diag_reject_{PAIR}.png"
plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); print(f"saved {out}", flush=True)
