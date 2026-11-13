"""Show what the glare-blob detector sees on a frame: bright mask + every connected component
(size) + merged glare regions. Reveals which cars/glare are missed and why (threshold/VBAND/size).
    python3 diag_blobs.py 14570 14600
"""
import os, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import load_pairs, load_calib, LOCAL, RES
sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC

PAIRS = [int(a) for a in sys.argv[1:]] or [14570, 14600]
CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
BRIGHT = 205; VBAND = (0.20, 0.85); MIN_BLOB = 60
cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
pairs = load_pairs(); bag_ts = np.load(CIDX)["color_bag_ts_ns"]
imgs = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[p]["color_idx"] for p in PAIRS], Path(CACHE), "color", bag_ts)

for P in PAIRS:
    img = imgs[pairs[P]["color_idx"]]
    if img.ndim == 1: img = img.reshape(H, W, 3)
    # try several brightness thresholds to see what's recoverable
    print(f"\npair {P}:", flush=True)
    for TH in (180, 205, 225):
        bright = (img.min(2) > TH).copy()
        bright[:int(H*VBAND[0])] = False; bright[int(H*VBAND[1]):] = False
        lab, n = ndimage.label(bright)
        comps = []
        for i in range(1, n+1):
            ys, xs = np.where(lab == i)
            if len(xs) >= MIN_BLOB: comps.append((xs.min(), xs.max(), ys.mean(), len(xs)))
        print(f"  BRIGHT>{TH}: {len(comps)} blobs (>= {MIN_BLOB}px): " +
              ", ".join(f"[u{a}-{b} v{v:.0f} n{s}]" for a, b, v, s in sorted(comps)), flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    axs[0].imshow(img); axs[0].set_title(f"{P}: camera"); axs[0].axis("off")
    axs[1].imshow(img)
    bright = (img.min(2) > BRIGHT).copy(); bright[:int(H*VBAND[0])] = False; bright[int(H*VBAND[1]):] = False
    ov = np.zeros((*bright.shape, 4)); ov[bright] = [1, 0, 0, 0.5]; axs[1].imshow(ov)
    lab, n = ndimage.label(bright)
    for i in range(1, n+1):
        ys, xs = np.where(lab == i)
        if len(xs) < MIN_BLOB: continue
        axs[1].add_patch(Rectangle((xs.min(), ys.min()), xs.max()-xs.min(), ys.max()-ys.min(),
                         fill=False, ec="lime", lw=2))
        axs[1].text(xs.min(), ys.min()-4, f"n{len(xs)}", color="lime", fontsize=9)
    axs[1].axhline(H*VBAND[0], color="cyan", lw=0.8); axs[1].axhline(H*VBAND[1], color="cyan", lw=0.8)
    axs[1].set_title(f"{P}: BRIGHT>{BRIGHT} mask (red) + blobs (lime), VBAND (cyan)"); axs[1].axis("off")
    out = f"{RES}/fusion/night_label/diag_blobs_{P}.png"
    plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close(); print(f"  saved {out}", flush=True)
