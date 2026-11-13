"""Single viewer for the labels: oriented 3D boxes (yaw + l/w/h) on the lidar BEV,
the camera image with its 2D YOLO box, and (optionally) a radar cross-check panel.

This is the one label viewer -- it renders everything from the saved annotations plus,
with ``--radar``, a live radar BEV panel (so it fully replaces the old generation-time
``yolo_to_lidar --overlay`` output). Boxes are colored by RELIABILITY; radar-confirmed
boxes additionally get a magenta heading tick and an ``[R]`` tag.

  green = reliable (enough points AND plausible size)   orange = unreliable/implausible
  [R]   = a radar strong-return lands within --radar_assoc_m of the box footprint

    # product view (camera + oriented box3d BEV)
    python code/sync/src/camera_projection/viz_labels.py -d 2026.05.08/17-34-27 --pairs 4500 5000
    # + radar cross-check panel (loads radar live)
    python code/sync/src/camera_projection/viz_labels.py -d 2026.05.08/17-34-27 --pairs 5000 --radar
"""
from __future__ import annotations

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


if __name__ == "__main__":
    raise SystemExit(main())
