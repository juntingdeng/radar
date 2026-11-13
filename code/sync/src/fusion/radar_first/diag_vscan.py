"""Lidar VERTICAL scan profile across a window: range (x) vs height-above-ground (y) for the forward
glare-side cone. Shows the beam fan -- how vertical sampling falls off with range, the overhanging
canopy, the road, and whether anything lands at CAR height at distance.
    python3 diag_vscan.py 15450 15460 ...
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_pairs, load_calib, LidarSource, RES
import glare_bearing as G

PAIRS = [int(a) for a in sys.argv[1:]] or [15450, 15480, 15500, 15520, 15560, 15600]
pairs = load_pairs(); src = LidarSource()
CONE = 25.0                                        # deg half-cone toward the forward/glare side (+bearing)
n = len(PAIRS); cols = 4; rows = int(np.ceil(n/cols))
fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows)); axs = np.atleast_1d(axs).ravel()
for i, P in enumerate(PAIRS):
    xyz = src.xyz(pairs[P]["lidar_idx"]); rng = np.hypot(xyz[:, 0], xyz[:, 1])
    near = xyz[rng < G.RMAX]; coef = G.fit_ground_plane(near)
    bearing = np.degrees(np.arctan2(near[:, 1], near[:, 0]))
    cone = (bearing > 0) & (bearing < CONE)        # forward + left (glare side in this window)
    r = np.hypot(near[cone, 0], near[cone, 1]); h = G.height_above(near[cone], coef)
    ax = axs[i]
    carh = (h > 0.3) & (h < 2.0)
    ax.scatter(r[~carh], h[~carh], s=2, c="0.65")
    ax.scatter(r[carh], h[carh], s=4, c="red")     # red = car-height band
    ax.axhline(0.3, color="g", ls=":", lw=0.8); ax.axhline(2.0, color="g", ls=":", lw=0.8)
    ax.set_xlim(0, 45); ax.set_ylim(-1, 9)
    ax.set_title(f"{P}  (car-ht band red: {int(carh.sum())} pts)", fontsize=10)
    ax.set_xlabel("range (m)"); ax.set_ylabel("height above ground (m)")
for j in range(n, len(axs)): axs[j].axis("off")
out = f"{RES}/fusion/night_label/diag_vscan.png"
plt.suptitle("W3 lidar vertical scan (forward glare-side cone): canopy above, road below, car-height band (red)", fontsize=12)
plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight"); print(f"saved {out}", flush=True)
