from __future__ import annotations
"""Sensor overlay / label viewers (merged). Subcommands: lidar_cam | radar | labels"""
import sys

# ======== src/camera_projection/lidar_cam_overlay.py ========
# Lidar-on-camera overlay (calibration sanity + labeling aid).
# 
# Left: lidar points projected into the camera image, colored by range -- if they track the road/poles/car
# bodies, camera<->lidar calib is good. Right: BEV (lidar gray) with GT-queue proposals (red=auto,
# orange=temporal) + each proposal's camera-bearing ray -- shows which cars have a lidar cluster and which
# are lidar-blind (camera-only / radar-matters).
# 
#   python3 sync/src/camera_projection/lidar_cam_overlay.py 17000 39015 ...   # run from repo root
# Output PNGs -> sync/res/benchmark_v1/gt_queue/overlays/overlay_<pair>.png
def lidar_cam_main():
    import os, sys, json
    from pathlib import Path
    import numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, "sync/src/camera_projection"); sys.path.insert(0, "sync/src/fusion/radar_first")
    sys.path.insert(0, "sync/src/camera_projection"); sys.path.insert(0, "sync/src/fusion/radar_first")
    from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
    from common import load_pairs, load_calib, LidarSource, LOCAL, RES
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    OUTDIR = "sync/res/benchmark_v1/gt_queue/overlays"
    PROPOSALS = "sync/res/benchmark_v1/gt_queue/proposals.json"
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L)
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L)
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; Hh, Ww = ci["height"], ci["width"]
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
    if True:
        for F in [int(x) for x in sys.argv[1:]] or [17000]:
            overlay(F)

# ======== src/camera_projection/radar_overlay.py ========
# Radar-aware GT-queue overlay -- visualizes the v4 pipeline (lidar proposes / radar confirms / camera bonus).
# 
# LEFT : camera image + lidar projected (range-colored) + v4 proposal boxes drawn in the image (bbox for
#        camera-seen ones; projected 3D footprint for lidar ones) colored by source.
# RIGHT: BEV. lidar=gray. RADAR CFAR detections overlaid: static=silver x, MOVERS=red open circles (Doppler-
#        flagged, |vrad| labeled). v4 proposals as rectangles colored by source:
#          lime=lidar+radar (both active sensors agree)   gold=lidar+cam (radar silent)
#          orange=lidar-only (geometric candidate)        magenta=cam-only (lidar-blind, size-range)
#        This is the figure that shows radar corroborating cars the glare-blinded camera cannot see.
# 
#   python3 sync/src/camera_projection/radar_overlay.py 26874 19877 ...     # run from repo root
# Output -> sync/res/benchmark_v1/gt_queue/overlays/radar_overlay_<pair>.png
def radar_main():
    import os, sys, json
    from pathlib import Path
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    sys.path.insert(0, "sync/src/camera_projection"); sys.path.insert(0, "sync/src/fusion/radar_first")
    sys.path.insert(0, "sync/src/camera_projection"); sys.path.insert(0, "sync/src/fusion/radar_first")
    from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
    from common import load_pairs, load_calib, LidarSource, RES, LOCAL
    from detector import RadarDetector
    from ego_doppler import fit_ego_velocity
    from lidar_deskew import load_ego, load_xyz_motioncomp
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    OUTDIR = "sync/res/benchmark_v1/gt_queue/overlays"
    PROPOSALS = "sync/res/benchmark_v1/gt_queue/proposals.json"
    SRC_COLOR = {"lidar+radar": "lime", "lidar+radar+cam": "lime", "lidar+cam": "gold",
                 "lidar_only": "orange", "cam+radar_blind": "royalblue",
                 "cam_only_guess": "magenta", "cam_only_blind": "magenta"}
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L)
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L)
    R2L = np.array(cal["radar_to_lidar_auto"])
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; Hh, Ww = ci["height"], ci["width"]
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; Hh, Ww = ci["height"], ci["width"]
    bag_ts = np.load(CIDX)["color_bag_ts_ns"]
    props_all = json.load(open(PROPOSALS))["objects"] if os.path.exists(PROPOSALS) else {}
    src = LidarSource(); rdet = RadarDetector(processing="legacy"); EGO = load_ego()
    src = LidarSource(); rdet = RadarDetector(processing="legacy"); EGO = load_ego()
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
    if True:
        for F in [int(x) for x in sys.argv[1:]] or [26874]:
            overlay(F)

# ======== src/camera_projection/viz_labels.py ========
# Single viewer for the labels: oriented 3D boxes (yaw + l/w/h) on the lidar BEV,
# the camera image with its 2D YOLO box, and (optionally) a radar cross-check panel.
# 
# This is the one label viewer -- it renders everything from the saved annotations plus,
# with ``--radar``, a live radar BEV panel (so it fully replaces the old generation-time
# ``yolo_to_lidar --overlay`` output). Boxes are colored by RELIABILITY; radar-confirmed
# boxes additionally get a magenta heading tick and an ``[R]`` tag.
# 
#   green = reliable (enough points AND plausible size)   orange = unreliable/implausible
#   [R]   = a radar strong-return lands within --radar_assoc_m of the box footprint
# 
#     # product view (camera + oriented box3d BEV)
#     python code/sync/src/camera_projection/viz_labels.py -d 2026.05.08/17-34-27 --pairs 4500 5000
#     # + radar cross-check panel (loads radar live)
#     python code/sync/src/camera_projection/viz_labels.py -d 2026.05.08/17-34-27 --pairs 5000 --radar
def labels_main():
    import argparse
    import json
    import os
    import sys
    from pathlib import Path
    import numpy as np
    os.environ.setdefault("MPLBACKEND", "Agg")
    _THIS_DIR = Path(__file__).resolve().parent
    _SYNC_DIR = _THIS_DIR.parent
    for _p in (_SYNC_DIR, _THIS_DIR, _SYNC_DIR / "calibration"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402
    from lib.camera_io import DEFAULT_COLOR_TOPIC  # noqa: E402
    from annotations import load_annotations  # noqa: E402
    from extrinsic_io import get_extrinsic  # noqa: E402
    from yolo_to_lidar import (  # noqa: E402
        _bev_window, _extract_frames, _frame_cache_dir, _lidar_points_at,
        _radar_setup, _radar_panel, _peaks_from_bev, _radar_hits,
    )
    def build_argparser():
        p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("--calibration_json", default=None)
        p.add_argument("--camera_bag", default=None)
        p.add_argument("--camera_index", default=None)
        p.add_argument("--camera_sync_csv", default=None)
        p.add_argument("--lidar_cache", default=None,
                       help="Dir of pre-extracted per-scan XYZ npy (l_<idx>.npy). If every "
                            "needed scan is present, the 70GB pcap is never opened -- enables "
                            "contention-free parallel rendering.")
        p.add_argument("--lidar_pcap", default=None)
        p.add_argument("--lidar_metadata", default=None)
        p.add_argument("--annotations", default=None)
        p.add_argument("--pairs", type=int, nargs="+", required=True)
        p.add_argument("--max_depth_m", type=float, default=25.0,
                       help="Auto-zoom BEV window cap (m) when boxes are present. Does NOT clip the "
                            "point scatter — points are always drawn and the zoom window sets the view.")
        p.add_argument("--frame_cache", default=None)
        p.add_argument("--out_dir", default=None)
        p.add_argument("--no_camera", action="store_true",
                       help="Drop the camera panel (box3d BEV [+ radar] only). Skips per-frame camera "
                            "fetch from the Drive bag (~42 s/uncached frame) -- use for fast overview "
                            "renders over many strided pairs whose frames aren't locally cached.")
        p.add_argument("--fixed_bev", action="store_true",
                       help="Use a FIXED BEV window for every frame (no per-frame auto-zoom), so objects "
                            "stay at a stable scale/position and are easy to track in a video. Extent set "
                            "by --bev_fwd_max and --bev_lat_half.")
        p.add_argument("--bev_fwd_max", type=float, default=25.0,
                       help="Fixed-BEV forward extent: 0..this m (with --fixed_bev).")
        p.add_argument("--bev_lat_half", type=float, default=15.0,
                       help="Fixed-BEV lateral half-width: +/-this m (with --fixed_bev).")
        p.add_argument("--no_fov_wedge", action="store_true",
                       help="Don't draw the camera horizontal-FOV wedge on the lidar BEV.")
        # --- radar cross-check panel (loaded live; ported from yolo_to_lidar --overlay) ---
        p.add_argument("--radar", action="store_true",
                       help="Add a radar BEV panel and mark radar-confirmed boxes ([R]). Loads radar live.")
        p.add_argument("--radar_h5", default=None)
        p.add_argument("--cfg_file", default=None)
        p.add_argument("--sync_csv", default=None)
        p.add_argument("--sync_summary", default=None)
        p.add_argument("--radar_percentile", type=float, default=99.9,
                       help="Percentile of the radar BEV kept as strong returns (higher = stricter).")
        p.add_argument("--radar_bev_res", type=float, default=0.1,
                       help="Radar BEV Cartesian grid resolution (m).")
        p.add_argument("--radar_assoc_m", type=float, default=1.5,
                       help="A box is radar-confirmed if a radar return lands within this margin (m).")
        p.add_argument("--radar_points", action="store_true",
                       help="Add a panel scatter-plotting the sparse radar strong-return POINTS (the "
                            "peaks used for [R] association) over faint lidar + the boxes -- a cleaner "
                            "'radar detections vs labels' view than the heatmap.")
        p.add_argument("--radar_doppler_mode", default="all", choices=["all", "moving", "static"],
                       help="Radar panel Doppler mode. Default 'all' shows static + moving returns "
                            "(so static structure like an indoor wall is visible); 'moving' drops "
                            "static clutter (better for isolating walkers, but hides static objects).")
        p.add_argument("--radar_processing", default="legacy", choices=["legacy", "ti"],
                       help="Radar panel chain. 'legacy' (default) keeps zero-Doppler energy so 'all' "
                            "shows static returns; 'ti' adds a zero-Doppler notch (re-hides static).")
        add_dataset_arguments(p)
        return p
    def _oriented_corners(o):
        """4 BEV corners (forward X, lateral Y) of an oriented box3d dict."""
        cx, cy, yaw = o["cx"], o["cy"], o["yaw"]
        hl, hw = o["length"] / 2.0, o["width"] / 2.0
        major = np.array([np.cos(yaw), np.sin(yaw)])       # (X fwd, Y lat)
        minor = np.array([-np.sin(yaw), np.cos(yaw)])
        c = np.array([cx, cy])
        return np.array([c + sl * hl * major + sw * hw * minor
                         for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    def _fov_wedge(calib_raw):
        """Camera apex + heading + half-FOV in the lidar frame, or None if unavailable."""
        try:
            cam2lid, _ = get_extrinsic(calib_raw, "camera_to_lidar")
        except KeyError:
            return None
        ci = calib_raw.get("color_intrinsics") or calib_raw.get("intrinsics")
        if not ci:
            return None
        apex_fwd, apex_lat = float(cam2lid[0, 3]), float(cam2lid[1, 3])
        fdir = cam2lid[:3, 2]  # camera +Z (optical forward) in lidar frame
        heading = np.arctan2(fdir[1], fdir[0])
        half_fov = np.arctan((float(ci.get("width", 640)) * 0.5) / float(ci["fx"]))
        return apex_fwd, apex_lat, heading, half_fov
    def main() -> int:
        args = apply_dataset_config(build_argparser().parse_args())
        if not args.annotations:
            args.annotations = str(Path(args.calibration_json).parent / "annotations" / "yolo.json")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Polygon

        objects = load_annotations(args.annotations)["objects"]
        with open(args.calibration_json, "r", encoding="utf-8") as f:
            calib_raw = json.load(f)
        wedge = None if args.no_fov_wedge else _fov_wedge(calib_raw)

        import csv as _csv
        rows = {}
        with open(args.camera_sync_csv, newline="") as f:
            for r in _csv.DictReader(f):
                rows[int(r["pair_idx"])] = r

        cache = _frame_cache_dir(args)
        bag_ts = None
        if args.camera_index and Path(args.camera_index).is_file():
            idx = np.load(args.camera_index)
            bag_ts = idx["color_bag_ts_ns"] if "color_bag_ts_ns" in idx else None
        color_idx = {int(rows[p]["camera_color_idx"]): p for p in args.pairs if p in rows}
        color_map = {} if args.no_camera else _extract_frames(
            args.camera_bag, DEFAULT_COLOR_TOPIC, sorted(color_idx), cache, "color", bag_ts)

        # Lidar source: prefer the pre-extracted scan cache. Open the 70GB pcap ONLY for
        # scans missing from the cache (cheap/none if fully pre-cached) -- this is what lets
        # many renderers run in parallel without thrashing the pcap.
        lcache_dir = args.lidar_cache
        def _cached_scan_path(idx):
            return os.path.join(lcache_dir, f"l_{int(idx)}.npy") if lcache_dir else None
        needed_lidx = [int(rows[p]["lidar_idx"]) for p in args.pairs if p in rows]
        missing_lidx = [i for i in needed_lidx if not (lcache_dir and os.path.exists(_cached_scan_path(i)))]
        lsrc = lmeta = None
        if missing_lidx:
            from lib.lidar_io import open_pcap_scan_source, sensor_info_from_source
            lsrc = open_pcap_scan_source(args.lidar_pcap, args.lidar_metadata, index=True)
            lmeta = sensor_info_from_source(lsrc)
            if lcache_dir:
                print(f"  {len(missing_lidx)} scan(s) missing from lidar cache -> opening pcap.")
        elif lcache_dir:
            print(f"  all {len(needed_lidx)} scans from lidar cache -> pcap NOT opened.")
        def _points_for(l_idx):
            cp = _cached_scan_path(l_idx)
            if cp and os.path.exists(cp):
                return np.load(cp).astype(np.float64)
            return _lidar_points_at(lsrc, lmeta, l_idx)

        # Optional radar loader (live, ported from yolo_to_lidar --overlay).
        radar_loader = radar_R = None
        if args.radar:
            radar_loader, radar_R = _radar_setup(
                args, calib_raw,
                doppler_mode=args.radar_doppler_mode, processing=args.radar_processing)
            if radar_loader is None:
                print("  --radar requested but radar unavailable; drawing without the radar panel.")

        out_dir = Path(args.out_dir or (Path(args.calibration_json).parent / "label_viz"))
        out_dir.mkdir(parents=True, exist_ok=True)

        for pid in args.pairs:
            r = rows.get(pid)
            if not r:
                continue
            c_idx, l_idx = int(r["camera_color_idx"]), int(r["lidar_idx"])
            boxes = objects.get(str(pid), [])
            pts = _points_for(l_idx)

            # Live radar cross-check: mark radar-confirmed boxes (adds [R]).
            bev = la = fa = rf = rl = None
            if radar_loader is not None:
                try:
                    bev, la, fa = _radar_panel(radar_loader, pid)
                    rf, rl = _peaks_from_bev(bev, la, fa, radar_R, args.radar_percentile)
                    for b in boxes:
                        b["radar_confirmed"] = bool(_radar_hits(b, rf, rl, args.radar_assoc_m) > 0)
                except Exception as exc:  # noqa: BLE001
                    print(f"  pair {pid}: radar load failed ({exc}).")
                    bev = rf = rl = None

            have_radar_panel = radar_loader is not None and bev is not None
            have_pts_panel = args.radar_points and rf is not None
            ncols = ((0 if args.no_camera else 1) + 1
                     + (1 if have_radar_panel else 0) + (1 if have_pts_panel else 0))
            fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7), squeeze=False)
            axes = axes[0]
            ai = 0
            axc = None if args.no_camera else axes[ai]
            if not args.no_camera:
                ai += 1
            axl = axes[ai]; ai += 1
            axrad = axpts = None
            if have_radar_panel:
                axrad = axes[ai]; ai += 1
            if have_pts_panel:
                axpts = axes[ai]; ai += 1

            # --- camera panel: 2D YOLO box, reliability color, conf + [R] tag ---
            if axc is not None:
                color = color_map.get(c_idx)
                if color is not None:
                    axc.imshow(color)
                for b in boxes:
                    xy = b.get("bbox_xyxy")
                    rel = (b.get("box3d") or {}).get("reliable", True)
                    ec = "lime" if rel else "orange"
                    if xy:
                        axc.add_patch(Rectangle((xy[0], xy[1]), xy[2] - xy[0], xy[3] - xy[1], fill=False, ec=ec, lw=2))
                        tag = b.get("label", "")
                        if b.get("score") is not None:
                            tag += f" {b['score']:.2f}"
                        if b.get("radar_confirmed"):
                            tag += " [R]"
                        axc.text(xy[0], max(0, xy[1] - 4), tag, color=ec, fontsize=9, clip_on=True)
                axc.set_title(f"pair {pid}: camera + YOLO (green=reliable)")
                axc.axis("off")

            # --- lidar BEV: oriented box3d, reliability color, FOV wedge, radar tick ---
            if pts.size:
                # No range clip on the point scatter — the BEV window (fixed_bev or auto-zoom)
                # already frames the view. Clipping points here made boxes drawn beyond the clip
                # render as empty phantoms; the zoom feature handles cropping instead.
                axl.scatter(pts[:, 1], pts[:, 0], s=2, c="0.6")
            if args.fixed_bev:
                # fixed window (no per-frame zoom): objects stay at stable scale/position.
                xhi, xlo, fy0, fy1 = args.bev_lat_half, -args.bev_lat_half, 0.0, args.bev_fwd_max
            else:
                (xhi, xlo), (fy0, fy1) = _bev_window(boxes, pad_m=2.0, min_half=4.0, cap=args.max_depth_m)
            if wedge is not None:
                apex_fwd, apex_lat, heading, half_fov = wedge
                for s in (+1, -1):
                    ang = heading + s * half_fov
                    axl.plot([apex_lat, apex_lat + (fy1 - apex_fwd) * np.tan(ang)],
                             [apex_fwd, fy1], "b--", lw=1, alpha=0.5)
            for b in boxes:
                o = b.get("box3d")
                if o is None:
                    continue
                rel = o.get("reliable", True)
                ec = "lime" if rel else "orange"
                cor = _oriented_corners(o)
                axl.add_patch(Polygon(cor[:, [1, 0]], closed=True, fill=False, ec=ec, lw=2.2))
                front = 0.5 * (cor[0] + cor[1])  # heading tick (front face midpoint)
                confirmed = bool(b.get("radar_confirmed"))
                axl.plot([o["cy"], front[1]], [o["cx"], front[0]], "-",
                         color="magenta" if confirmed else ec, lw=2.0 if confirmed else 1)
                tag = (f"{b.get('label','')} {o['length']:.1f}x{o['width']:.1f}x{o['height']:.1f} "
                       f"y{np.degrees(o['yaw']):+.0f} n{o.get('n_points','?')}"
                       + ("" if rel else " !") + (" [R]" if confirmed else ""))
                axl.text(o["cy"], o["cx"], tag, color=ec, fontsize=8, clip_on=True)
            axl.set_xlim(xhi, xlo)
            axl.set_ylim(fy0, fy1)
            axl.set_aspect("equal")
            axl.set_xlabel("Lateral Y (m)")
            axl.set_ylabel("Forward X (m)")
            axl.set_title(f"pair {pid}: oriented box3d (green=reliable, magenta tick=radar [R])")

            # --- radar BEV panel (warped into the lidar frame so distances match) ---
            if axrad is not None:
                RL, RF = np.meshgrid(la, fa)
                LF = radar_R[0, 0] * RF + radar_R[0, 1] * RL + radar_R[0, 3]
                LL = radar_R[1, 0] * RF + radar_R[1, 1] * RL + radar_R[1, 3]
                axrad.pcolormesh(LL, LF, bev, cmap="viridis", shading="gouraud")
                for b in boxes:
                    o = b.get("box3d")
                    if o is None:
                        continue
                    rel = o.get("reliable", True)
                    ec = "lime" if rel else "orange"           # reliability (same as lidar panel)
                    confirmed = bool(b.get("radar_confirmed"))
                    cor = _oriented_corners(o)
                    axrad.add_patch(Polygon(cor[:, [1, 0]], closed=True, fill=False, ec=ec, lw=2.0))
                    front = 0.5 * (cor[0] + cor[1])            # magenta heading tick = radar-confirmed
                    axrad.plot([o["cy"], front[1]], [o["cx"], front[0]], "-",
                               color="magenta" if confirmed else ec, lw=2.0 if confirmed else 1)
                axrad.set_xlim(xhi, xlo)
                axrad.set_ylim(fy0, fy1)
                axrad.set_aspect("equal")
                axrad.set_xlabel("Lateral Y (m)")
                axrad.set_ylabel("Forward X (m)")
                axrad.set_title(f"pair {pid}: radar BEV (lidar frame, {args.radar_doppler_mode}-Doppler) + boxes")

            # --- radar POINTS panel: highlight the radar returns AT each label ---
            # A labeled object may not be among the GLOBALLY strongest returns, so per box we
            # highlight (cyan) the strongest radar cells inside its footprint (within the assoc
            # margin) regardless of global rank; global top-percentile peaks are faint context.
            if axpts is not None:
                if pts.size:
                    axpts.scatter(pts[:, 1], pts[:, 0], s=1, c="0.85")  # faint lidar context (no range clip)
                axpts.scatter(rl, rf, s=16, c="0.35", marker="x", linewidths=1.0)  # global strong (context)
                # warp ALL radar cells to the lidar frame so we can pick returns inside each box.
                RL, RF = np.meshgrid(la, fa)
                LFc = radar_R[0, 0] * RF + radar_R[0, 1] * RL + radar_R[0, 3]
                LLc = radar_R[1, 0] * RF + radar_R[1, 1] * RL + radar_R[1, 3]
                finite = np.isfinite(bev)
                mA = args.radar_assoc_m
                for b in boxes:
                    o = b.get("box3d")
                    if o is None:
                        continue
                    rel = o.get("reliable", True)
                    ec = "lime" if rel else "orange"           # reliability (same as lidar panel)
                    confirmed = bool(b.get("radar_confirmed"))
                    cor = _oriented_corners(o)
                    axpts.add_patch(Polygon(cor[:, [1, 0]], closed=True, fill=False, ec=ec, lw=2.0))
                    front = 0.5 * (cor[0] + cor[1])            # magenta heading tick = radar-confirmed
                    axpts.plot([o["cy"], front[1]], [o["cx"], front[0]], "-",
                               color="magenta" if confirmed else ec, lw=2.0 if confirmed else 1)
                    # highlight the strongest radar cells inside this box's footprint (+assoc margin)
                    lat0, lat1 = b.get("lateral", [0, 0])
                    fwd0, fwd1 = b.get("forward", [0, 0])
                    inb = (finite & (LFc >= fwd0 - mA) & (LFc <= fwd1 + mA)
                           & (LLc >= lat0 - mA) & (LLc <= lat1 + mA))
                    if inb.any():
                        sel = inb & (bev >= bev[inb].max() - 6.0)   # peak returns at the label
                        axpts.scatter(LLc[sel], LFc[sel], s=16, c="red", marker="o",
                                      edgecolors="black", linewidths=0.4, zorder=5)
                axpts.set_xlim(xhi, xlo)
                axpts.set_ylim(fy0, fy1)
                axpts.set_aspect("equal")
                axpts.set_xlabel("Lateral Y (m)")
                axpts.set_ylabel("Forward X (m)")
                axpts.set_title(f"pair {pid}: radar points (red=in-box returns, gray=global strong)")

            out = out_dir / f"label_pair{pid}.png"
            fig.savefig(out, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")
        return 0
    if True:
        raise SystemExit(main())

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'lidar_cam': lidar_cam_main, 'radar': radar_main, 'labels': labels_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
