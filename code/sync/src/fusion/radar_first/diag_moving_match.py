"""WHY does moving-Doppler never corroborate the glare cars?

For each kept box, find the nearest MOVING radar detection and print d_range / d_bearing.
If movers sit at the right RANGE but wrong BEARING -> coarse radar azimuth (loosen bearing gate).
If no movers near in range at all -> the boxed cars are stopped, or ego-Doppler misses them.
    python3 diag_moving_match.py 15480 15580 15560
"""
import os, sys, json
from types import SimpleNamespace
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_pairs, load_calib, LidarSource, RES
from detector import RadarDetector
from ego_doppler import fit_ego_velocity

PAIRS = [int(a) for a in sys.argv[1:]] or [15480, 15580]
cal = load_calib(); pairs = load_pairs()
R2L = np.array(cal["radar_to_lidar_auto"])
det = RadarDetector(processing="legacy"); src = LidarSource()
tag = f"win{min(PAIRS)//10*10+0}" if False else None
# load whatever window annotation holds these pairs
ANN = None
for f in os.listdir(f"{RES}/annotations"):
    if f.startswith("glare_win"):
        o = json.load(open(f"{RES}/annotations/{f}"))["objects"]
        if all(str(p) in o for p in PAIRS): ANN = o; break
ANN = ANN or {}


def bear(a): return np.degrees(np.arctan2(a[1], a[0]))

for PAIR in PAIRS:
    d, _ = det.detect(pairs[PAIR]["radar_idx"]); Xd, Yd = RadarDetector.to_xy(d)
    rf = R2L[0,0]*Xd + R2L[0,1]*Yd + R2L[0,3]; rl = R2L[1,0]*Xd + R2L[1,1]*Yd + R2L[1,3]
    rrng = np.hypot(rf, rl); rbear = np.degrees(np.arctan2(rl, rf))
    ab, inl = fit_ego_velocity(d["az"], d["vel"])
    if ab is not None:
        pr = np.cos(np.radians(-d["az"]))*ab[0] + np.sin(np.radians(-d["az"]))*ab[1]
        resid = d["vel"] - pr; mover = (~inl) & (np.abs(resid) > 0.8)
        egospd = np.hypot(ab[0], ab[1])
    else:
        mover = np.zeros(len(d), bool); egospd = float("nan")
    print(f"\npair {PAIR}: ego-speed {egospd:.2f} m/s, {mover.sum()}/{len(d)} movers", flush=True)
    mv_r, mv_b, mv_v = rrng[mover], rbear[mover], d["vel"][mover]
    for i in np.argsort(mv_r):
        print(f"   mover: range {mv_r[i]:5.1f}  bearing {mv_b[i]:+6.1f}  vrad {mv_v[i]:+5.1f}", flush=True)
    for b in ANN.get(str(PAIR), []):
        o = b["box3d"]; br, bb = np.hypot(o["cx"], o["cy"]), bear([o["cx"], o["cy"]])
        if mover.any():
            dr = np.abs(mv_r - br); dbear = np.abs(((mv_b - bb + 180) % 360) - 180)
            j = np.argmin(dr + dbear/5.0)
            print(f"   BOX r{br:5.1f} b{bb:+6.1f} [{b['source']}] -> nearest mover dR={dr[j]:+.1f} dBear={dbear[j]:+.1f}", flush=True)
        else:
            print(f"   BOX r{br:5.1f} b{bb:+6.1f} [{b['source']}] -> no movers", flush=True)

    P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < 45
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(P[nm, 1], P[nm, 0], s=0.4, c="0.85")
    ax.scatter(rl[~mover], rf[~mover], s=25, marker="x", c="silver", label="radar static")
    ax.scatter(rl[mover], rf[mover], s=90, facecolors="none", edgecolors="red", linewidths=2, label="radar MOVING")
    for b in ANN.get(str(PAIR), []):
        o = b["box3d"]; col = "lime" if b["color"] == "lime" else "orange"
        ax.add_patch(plt.Rectangle((o["cy"]-o["width"]/2, o["cx"]-o["length"]/2), o["width"], o["length"], fill=False, ec=col, lw=2))
    ax.set_aspect("equal"); ax.set_xlim(24, -24); ax.set_ylim(0, 45)
    ax.set_xlabel("lateral Y"); ax.set_ylabel("forward X"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{PAIR}: MOVING radar (red) vs boxes (green/orange) -- do they coincide?")
    out = f"{RES}/fusion/glare/diag_mvmatch_{PAIR}.png"
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(); print(f"saved {out}", flush=True)
