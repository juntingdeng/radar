"""Radar-aware GT-queue overlay -- visualizes the v4 pipeline (lidar proposes / radar confirms / camera bonus).

LEFT : camera image + lidar projected (range-colored) + v4 proposal boxes drawn in the image (bbox for
       camera-seen ones; projected 3D footprint for lidar ones) colored by source.
RIGHT: BEV. lidar=gray. RADAR CFAR detections overlaid: static=silver x, MOVERS=red open circles (Doppler-
       flagged, |vrad| labeled). v4 proposals as rectangles colored by source:
         lime=lidar+radar (both active sensors agree)   gold=lidar+cam (radar silent)
         orange=lidar-only (geometric candidate)        magenta=cam-only (lidar-blind, size-range)
       This is the figure that shows radar corroborating cars the glare-blinded camera cannot see.

  python3 sync/src/camera_projection/radar_overlay.py 26874 19877 ...     # run from repo root
Output -> sync/res/benchmark_v1/gt_queue/overlays/radar_overlay_<pair>.png
"""
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "sync/src/camera_projection"); sys.path.insert(0, "sync/src/fusion/radar_first")
from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
from common import load_pairs, load_calib, LidarSource, RES, LOCAL
from detector import RadarDetector
from ego_doppler import fit_ego_velocity
from lidar_deskew import load_ego, load_xyz_motioncomp

CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
OUTDIR = "sync/res/benchmark_v1/gt_queue/overlays"
PROPOSALS = "sync/res/benchmark_v1/gt_queue/proposals.json"
SRC_COLOR = {"lidar+radar": "lime", "lidar+radar+cam": "lime", "lidar+cam": "gold",
             "lidar_only": "orange", "cam+radar_blind": "royalblue",
             "cam_only_guess": "magenta", "cam_only_blind": "magenta"}

pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]
C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L)
R2L = np.array(cal["radar_to_lidar_auto"])
fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; Hh, Ww = ci["height"], ci["width"]
bag_ts = np.load(CIDX)["color_bag_ts_ns"]
props_all = json.load(open(PROPOSALS))["objects"] if os.path.exists(PROPOSALS) else {}
src = LidarSource(); rdet = RadarDetector(processing="legacy"); EGO = load_ego()


def radar_bev(F):
    """radar CFAR dets in lidar frame -> (fwd, lat, vrad, mover_mask)."""
    try:
        d, _ = rdet.detect(pairs[F]["radar_idx"])
    except Exception as e:
        print(f"  radar detect failed on {F}: {e}", flush=True)
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, bool)
    Xd, Yd = RadarDetector.to_xy(d)
    rf = R2L[0, 0]*Xd + R2L[0, 1]*Yd + R2L[0, 3]; rl = R2L[1, 0]*Xd + R2L[1, 1]*Yd + R2L[1, 3]
    ab, inl = fit_ego_velocity(d["az"], d["vel"])
    if ab is not None:
        pr = np.cos(np.radians(-d["az"]))*ab[0] + np.sin(np.radians(-d["az"]))*ab[1]
        mover = (~inl) & (np.abs(d["vel"] - pr) > 0.8)
    else:
        mover = np.zeros(len(d), bool)
    return rf, rl, d["vel"], mover


BEV_ONLY = bool(os.environ.get("BEV_ONLY"))   # skip the slow camera Drive-seek; render BEV panel only


def overlay(F):
    if BEV_ONLY:                                          # fast: raw lidar context (proposals already motion-corrected)
        P = src.xyz(pairs[F]["lidar_idx"])
    else:
        P = load_xyz_motioncomp(src, pairs[F]["lidar_idx"], EGO.get(F), rdet, pairs[F]["radar_idx"],
                                R2L, max_xy=45.0)          # general ego+object motion compensation
    P = P[(np.hypot(P[:, 0], P[:, 1]) > 1) & (np.hypot(P[:, 0], P[:, 1]) < 45)]
    rf, rl, rvel, mover = radar_bev(F)

    if not BEV_ONLY:
        img = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[F]["color_idx"]], Path(CACHE), "color", bag_ts).get(pairs[F]["color_idx"])
        if img is not None and img.ndim == 1: img = img.reshape(Hh, Ww, 3)
        Pc = (L2C @ np.c_[P, np.ones(len(P))].T).T[:, :3]; fr = Pc[:, 2] > 0.3
        u = fx * Pc[fr, 0] / Pc[fr, 2] + cx; v = fy * Pc[fr, 1] / Pc[fr, 2] + cy
        dep = np.hypot(P[fr, 0], P[fr, 1]); inim = (u >= 0) & (u < Ww) & (v >= 0) & (v < Hh)
        fig, ax = plt.subplots(1, 2, figsize=(21, 8)); bev = ax[1]
        # ---- LEFT: camera + lidar + proposal boxes in image ----
        if img is not None: ax[0].imshow(img)
        sc = ax[0].scatter(u[inim], v[inim], c=dep[inim], s=4, cmap="jet", alpha=0.5, vmin=0, vmax=40)
        plt.colorbar(sc, ax=ax[0], fraction=0.03, label="lidar range (m)")
        for b in props_all.get(str(F), []):
            col = SRC_COLOR.get(b.get("proposal_source", ""), "white")
            bb = b.get("bbox_xyxy")
            if bb:                                           # camera-seen -> draw its 2D box
                ax[0].add_patch(Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], fill=False, edgecolor=col, lw=2))
            else:                                            # lidar-only -> project 3D center into image
                xf = 0.5*(b["forward"][0]+b["forward"][1]); yl = 0.5*(b["lateral"][0]+b["lateral"][1])
                zc = (b.get("box3d") or {}).get("cz", 0.0)
                pc = L2C @ np.array([xf, yl, zc, 1.0])
                if pc[2] > 0.3:
                    ax[0].plot(fx*pc[0]/pc[2]+cx, fy*pc[1]/pc[2]+cy, "o", mfc="none", mec=col, ms=14, mew=2)
        ax[0].set_xlim(0, Ww); ax[0].set_ylim(Hh, 0)
        ax[0].set_title(f"pair {F}: camera + lidar (range) + proposals")
    else:
        fig, bev = plt.subplots(figsize=(11, 10))

    # ---- BEV lidar + radar + proposals ----
    bev.scatter(P[:, 1], P[:, 0], s=2, c="0.72", label="lidar")
    if len(rf):
        st = ~mover
        bev.scatter(rl[st], rf[st], s=55, marker="x", c="dimgray", linewidths=1.6, label="radar static")
        if mover.any():
            bev.scatter(rl[mover], rf[mover], s=160, marker="o", facecolors="none",
                        edgecolors="red", linewidths=2.2, label="radar MOVING")
            for i in np.where(mover)[0]:
                bev.annotate(f"{rvel[i]:+.1f}", (rl[i], rf[i]), fontsize=8, color="red",
                             xytext=(4, 4), textcoords="offset points")
    seen = set()
    for b in props_all.get(str(F), []):
        s = b.get("proposal_source", ""); col = SRC_COLOR.get(s, "white")
        x0, x1 = b["forward"]; y0, y1 = b["lateral"]
        lab = s if s not in seen else None; seen.add(s)
        bev.add_patch(Rectangle((min(y0, y1), min(x0, x1)), abs(y1-y0), abs(x1-x0),
                                fill=False, edgecolor=col, lw=2.2, label=lab))
        if b.get("moving"):                              # ring the radar-moving boxes
            cxb, cyb = 0.5*(y0+y1), 0.5*(x0+x1)
            bev.plot(cxb, cyb, "o", mfc="none", mec="red", ms=20, mew=1.5)
    bev.set_xlim(40, -40); bev.set_ylim(0, 45); bev.set_aspect("equal")
    bev.set_title("BEV: lidar(gray) + radar(x=static, o=MOVING) + proposals\n"
                  "lime=lidar+radar  gold=lidar+cam  orange=lidar-only  blue=cam+radar  magenta=cam-only-guess")
    bev.set_xlabel("lateral (m)"); bev.set_ylabel("forward (m)")
    bev.legend(loc="upper right", fontsize=8, framealpha=0.9)
    os.makedirs(OUTDIR, exist_ok=True); out = f"{OUTDIR}/radar_overlay_{F}.png"
    plt.tight_layout(); plt.savefig(out, dpi=95); plt.close(fig); print(f"saved {out}", flush=True)


if __name__ == "__main__":
    for F in [int(x) for x in sys.argv[1:]] or [26874]:
        overlay(F)
