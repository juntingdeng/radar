"""Lidar-on-camera overlay (calibration sanity + labeling aid).

Left: lidar points projected into the camera image, colored by range -- if they track the road/poles/car
bodies, camera<->lidar calib is good. Right: BEV (lidar gray) with GT-queue proposals (red=auto,
orange=temporal) + each proposal's camera-bearing ray -- shows which cars have a lidar cluster and which
are lidar-blind (camera-only / radar-matters).

  python3 sync/src/camera_projection/lidar_cam_overlay.py 17000 39015 ...   # run from repo root
Output PNGs -> sync/res/benchmark_v1/gt_queue/overlays/overlay_<pair>.png
"""
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "sync/src/camera_projection"); sys.path.insert(0, "sync/src/fusion/radar_first")
from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
from common import load_pairs, load_calib, LidarSource, LOCAL, RES

CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
OUTDIR = "sync/res/benchmark_v1/gt_queue/overlays"
PROPOSALS = "sync/res/benchmark_v1/gt_queue/proposals.json"

pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]
C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L)
fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; Hh, Ww = ci["height"], ci["width"]
bag_ts = np.load(CIDX)["color_bag_ts_ns"]
props_all = json.load(open(PROPOSALS))["objects"] if os.path.exists(PROPOSALS) else {}
src = LidarSource()


def overlay(F):
    img = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[F]["color_idx"]], Path(CACHE), "color", bag_ts).get(pairs[F]["color_idx"])
    if img is not None and img.ndim == 1: img = img.reshape(Hh, Ww, 3)
    P = src.xyz(pairs[F]["lidar_idx"]); P = P[(np.hypot(P[:, 0], P[:, 1]) > 1) & (np.hypot(P[:, 0], P[:, 1]) < 45)]
    Pc = (L2C @ np.c_[P, np.ones(len(P))].T).T[:, :3]; fr = Pc[:, 2] > 0.3
    u = fx * Pc[fr, 0] / Pc[fr, 2] + cx; v = fy * Pc[fr, 1] / Pc[fr, 2] + cy
    dep = np.hypot(P[fr, 0], P[fr, 1]); inim = (u >= 0) & (u < Ww) & (v >= 0) & (v < Hh)
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    if img is not None: ax[0].imshow(img)
    sc = ax[0].scatter(u[inim], v[inim], c=dep[inim], s=4, cmap="jet", alpha=0.6, vmin=0, vmax=40)
    ax[0].set_xlim(0, Ww); ax[0].set_ylim(Hh, 0); plt.colorbar(sc, ax=ax[0], fraction=0.03)
    ax[0].set_title(f"pair {F}: lidar projected onto camera (color=range)")
    ax[1].scatter(P[:, 1], P[:, 0], s=2, c="0.6")
    for b in props_all.get(str(F), []):
        xf = 0.5 * (b["forward"][0] + b["forward"][1]); yl = 0.5 * (b["lateral"][0] + b["lateral"][1])
        col = "orange" if b.get("proposal_source", "").startswith("temporal") else "red"
        ax[1].scatter([yl], [xf], marker="s", s=80, facecolors="none", edgecolors=col, linewidths=2)
        bb = b.get("bbox_xyxy")
        if bb:
            d = np.array([(0.5*(bb[0]+bb[2])-cx)/fx, (0.5*(bb[1]+bb[3])-cy)/fy, 1.0]); d /= np.linalg.norm(d)
            dl = C2L[:3, :3] @ d; be = np.arctan2(dl[1], dl[0])
            ax[1].plot([0, 45*np.sin(be)], [0, 45*np.cos(be)], col, lw=0.8, alpha=0.5)
    ax[1].set_xlim(24, -24); ax[1].set_ylim(0, 45); ax[1].set_aspect("equal")
    ax[1].set_title("BEV: lidar(gray) + proposals (red=auto, orange=temporal) + camera-bearing rays")
    ax[1].set_xlabel("lateral (m)"); ax[1].set_ylabel("forward (m)")
    os.makedirs(OUTDIR, exist_ok=True); out = f"{OUTDIR}/overlay_{F}.png"
    plt.tight_layout(); plt.savefig(out, dpi=90); plt.close(fig); print(f"saved {out}", flush=True)


if __name__ == "__main__":
    for F in [int(x) for x in sys.argv[1:]] or [17000]:
        overlay(F)
