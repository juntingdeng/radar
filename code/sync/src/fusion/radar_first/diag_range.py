"""Along the glare bearing: is there a CAR in lidar, and at what range? Or is the car beyond lidar range
(only near foliage returns)? Scatter of (range, height-above-plane) for points on the glare bearing.
    python3 diag_range.py 15500 15480
"""
import os, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_pairs, load_calib, LidarSource, LOCAL, RES
import glare_bearing as G
sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC

PAIRS = [int(a) for a in sys.argv[1:]] or [15500]
CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
bag_ts = np.load(CIDX)["color_bag_ts_ns"]; src = LidarSource()
imgs = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[p]["color_idx"] for p in PAIRS], Path(CACHE), "color", bag_ts)

for PAIR in PAIRS:
    img = imgs[pairs[PAIR]["color_idx"]]
    if img.ndim == 1: img = img.reshape(H, W, 3)
    blobs = G.car_glare_blobs(img)
    P = src.xyz(pairs[PAIR]["lidar_idx"]); rng_all = np.linalg.norm(P[:, :2], axis=1)
    near = P[rng_all < G.RMAX]; coef = G.fit_ground_plane(near); ha = G.height_above(near, coef)
    nb = np.degrees(np.arctan2(near[:, 1], near[:, 0])); nr = np.hypot(near[:, 0], near[:, 1])
    print(f"\npair {PAIR}: max lidar range (any) {rng_all.max():.1f} m, {len(blobs)} glare blob(s)", flush=True)
    fig, axs = plt.subplots(1, len(blobs)+1, figsize=(7*(len(blobs)+1), 6))
    if len(blobs) == 1: axs = [axs[0], axs[1]] if hasattr(axs, '__len__') else axs
    axs = np.atleast_1d(axs)
    axs[0].imshow(img); axs[0].axis("off"); axs[0].set_title(f"{PAIR} camera")
    for i, b in enumerate(blobs, 1):
        bc = 0.5*(G.ray(b["u0"], b["vb"])[0] + G.ray(b["u1"], b["vb"])[0])
        onb = np.abs(((nb - bc + 180) % 360) - 180) < 5
        r = nr[onb]; h = ha[onb]
        carh = (h > 0.3) & (h < 2.0)
        print(f"  blob bearing {bc:+.1f}: {onb.sum()} pts on bearing; car-height(0.3-2.0m) {carh.sum()}; "
              f"range span {r.min():.1f}-{r.max():.1f}m", flush=True)
        for lo in range(0, 45, 5):
            m = (r >= lo) & (r < lo+5); ch = m & carh
            if m.sum(): print(f"     {lo:2d}-{lo+5:2d}m: {m.sum():4d} pts, car-ht {int(ch.sum()):3d}", flush=True)
        ax = axs[min(i, len(axs)-1)]
        ax.scatter(r, h, s=3, c=np.where(carh, "red", "0.6"))
        ax.axhline(0.3, color="g", ls=":"); ax.axhline(2.0, color="g", ls=":")
        ax.set_xlim(0, 45); ax.set_ylim(-1, 8); ax.set_xlabel("range (m)"); ax.set_ylabel("height above plane (m)")
        ax.set_title(f"{PAIR} bearing{bc:+.0f}: red=car-height")
    out = f"{RES}/fusion/night_label/diag_range_{PAIR}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close(); print(f"  saved {out}", flush=True)
