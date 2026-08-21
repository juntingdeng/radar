"""Interactive radar->lidar extrinsic calibration via 2D top-down correspondences.

Radar has coarse azimuth resolution and no reliable elevation, so calibration is
done in the top-down (forward, lateral) plane: a planar rigid fit (yaw + XY
translation). The vertical offset between radar and lidar is not observable from
radar returns, so it is supplied via ``--z_offset_m`` (measure it once with a
tape) and carried straight into the 4x4.

Workflow
--------
1. Pick sync pairs where a compact strong reflector shows up in BOTH the radar
   BEV and the lidar BEV (a parked car, pole, corner, or a placed corner
   reflector). Wide, spread-out points give the most stable yaw estimate.
2. For each pair: click the reflector in the radar BEV (left), then the same
   object in the lidar BEV (right).
3. Press ``s`` to solve. Runs a 2D Umeyama fit, prints RMS, and writes
   ``radar_to_lidar`` into the calibration JSON.

Collect >= 3 well-separated correspondences (2 is the bare minimum).

Run::

    python code/sync/calibration/calibrate_radar_lidar.py \
        -d 2026.05.10/18-05-08 --pairs 500 1500 3000 --z_offset_m 0.0
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_CALIB_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _CALIB_DIR.parent
_CODE_ROOT = _SYNC_DIR.parent
for _p in (_SYNC_DIR, _CODE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.dataset_config import apply_dataset_config  # noqa: E402
from make_video import build_argparser as _video_argparser  # noqa: E402

from correspondences import CorrespondenceSet  # noqa: E402
from extrinsic_io import reuse_calibration, write_extrinsic  # noqa: E402
from rigid_solve import fit_with_rejection, planar_fit_to_4x4, umeyama_2d  # noqa: E402


def build_argparser():
    p = _video_argparser()  # inherits radar/lidar/scene/dataset flags + defaults
    p.description = "Radar->lidar extrinsic calibration (2D top-down)."
    g = p.add_argument_group("calibration")
    g.add_argument("--calibration_json", default=None, help="Calibration JSON to update")
    g.add_argument("--pairs", type=int, nargs="+", default=None, help="Sync pair indices.")
    g.add_argument("--correspondences", default=None, help="Correspondence JSON path.")
    g.add_argument("--solve_only", default=None, help="Solve from a correspondence JSON (no GUI).")
    g.add_argument(
        "--z_offset_m",
        type=float,
        default=0.0,
        help="Measured radar->lidar vertical offset (m); not observable from radar.",
    )
    g.add_argument(
        "--snap_radius_m",
        type=float,
        default=2.0,
        help="Max XY distance a lidar click may snap to a real return.",
    )
    g.add_argument(
        "--reuse_from",
        default=None,
        help="Same rig, already calibrated elsewhere: copy the solved radar_to_lidar "
        "(+ intrinsics) from this reference calibration JSON and exit (no picking).",
    )
    g.add_argument("--overwrite", action="store_true", help="With --reuse_from, replace an existing output.")
    g.add_argument(
        "--max_range_m",
        type=float,
        default=None,
        help="Upper bound for the BEV view (m). Default: the radar display range.",
    )
    g.add_argument(
        "--no_auto_range",
        action="store_true",
        help="Disable per-frame auto-zoom of both BEV panels (use the fixed radar range).",
    )
    g.add_argument(
        "--radar_bev_res",
        type=float,
        default=0.1,
        help="Radar BEV Cartesian grid resolution (m) for display. Finer = smoother, "
        "less blocky at short range (default 0.1; raise if rendering is slow).",
    )
    g.add_argument(
        "--no_flip_bev",
        action="store_true",
        help="Don't mirror the BEVs horizontally. By default BOTH panels are flipped so "
        "left/right matches the camera calibrator's physical orientation (display only).",
    )
    g.add_argument(
        "--reject_outliers_m",
        type=float,
        default=0.0,
        help="If >0, drop correspondences whose fit residual exceeds this (m) and refit.",
    )
    g.add_argument(
        "--show_camera",
        action="store_true",
        help="Add a camera RGB context panel (leftmost) so [camera | radar | lidar] are "
        "in a row — helps identify what a radar reflector actually is. Scans the bag once.",
    )
    g.add_argument(
        "--box",
        action="store_true",
        help="Box mode: DRAG a box around the reflector in each panel. Radar uses the "
        "intensity-weighted centroid (above background); lidar uses the point centroid.",
    )
    g.add_argument("--min_box_points", type=int, default=10, help="Min lidar points in a box (box mode).")
    g.add_argument(
        "--box_radar_pct",
        type=float,
        default=75.0,
        help="Box mode: radar centroid weights only cells above this percentile of the "
        "box (isolates the reflector peak from the diffuse envelope).",
    )
    return p


def _solve_and_write(cs: CorrespondenceSet, args, corr_path: Path) -> int:
    src, tgt = cs.as_arrays()
    if len(cs) < 2:
        print(f"Need >= 2 correspondences to solve; have {len(cs)}. Nothing written.")
        return 1
    if args.reject_outliers_m > 0:
        fit, keep = fit_with_rejection(
            umeyama_2d, src, tgt, max_residual=args.reject_outliers_m, min_points=3
        )
        dropped = [int(i) for i in np.where(~keep)[0]]
        if dropped:
            print(f"Robust fit dropped {len(dropped)} correspondence(s) "
                  f"> {args.reject_outliers_m} m: {dropped}")
    else:
        fit = umeyama_2d(src, tgt)
    yaw_deg = np.degrees(np.arctan2(fit.rotation[1, 0], fit.rotation[0, 0]))
    mat = planar_fit_to_4x4(fit, z_offset=args.z_offset_m)
    print("\n=== radar->lidar planar fit ===")
    print(fit.summary())
    print(f"yaw={yaw_deg:+.2f} deg, translation(forward,lateral)="
          f"({fit.translation[0]:+.3f}, {fit.translation[1]:+.3f}) m, z_offset={args.z_offset_m:+.3f} m")
    print("Transform (radar_sensor -> lidar_sensor):")
    print(np.array2string(mat, precision=5, suppress_small=True))
    if fit.rms > 0.5:
        print(f"WARNING: RMS {fit.rms:.2f} m is high; re-check picks or spread points wider.")
    cs.save(corr_path)
    write_extrinsic(
        args.calibration_json,
        key="radar_to_lidar",
        matrix=mat,
        n_points=fit.n_points,
        rms_m=fit.rms,
        max_error_m=fit.max_error,
        correspondences_path=corr_path,
        note=f"planar (forward,lateral) fit; z_offset={args.z_offset_m} m supplied manually",
    )
    return 0


def _load_pair_rows(csv_path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["pair_idx"])] = row
    return rows


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())

    # --- reuse path: same rig already calibrated elsewhere (no picking) ---
    if args.reuse_from:
        reuse_calibration(
            args.calibration_json,
            args.reuse_from,
            keys=("radar_to_lidar",),
            overwrite=args.overwrite,
        )
        return 0

    corr_path = Path(
        args.correspondences
        or (Path(args.calibration_json).parent / "radar_lidar_correspondences.json")
    )

    cs = CorrespondenceSet(
        kind="radar_lidar",
        dim=2,
        source_frame="radar_sensor_topdown",
        target_frame="lidar_sensor_topdown",
        dataset=getattr(args, "dataset", "") or "",
    )

    if args.solve_only:
        cs = CorrespondenceSet.load(args.solve_only)
        return _solve_and_write(cs, args, Path(args.solve_only))

    if not args.pairs:
        print("No --pairs given. Choose sync pair indices with a shared strong reflector.")
        return 2

    from interactive import bev_view_limits, ensure_interactive_backend

    ensure_interactive_backend()  # must run before pyplot is imported
    import matplotlib.pyplot as plt

    from lib.bev_render import LidarScanReader
    from make_video import RadarLoader

    radar = RadarLoader(args)
    # Render the radar BEV on a finer Cartesian grid so it isn't blocky when
    # zoomed into short range (the default 0.25 m grid looks like a few pixels).
    if hasattr(radar, "display_bev_kw"):
        radar.display_bev_kw["res"] = float(args.radar_bev_res)
    print(
        "Opening lidar PCAP (indexing ~12 GB; a few minutes over USB, no output "
        "during the scan — it is NOT stuck)...",
        flush=True,
    )
    lidar = LidarScanReader(args.lidar_pcap, args.lidar_metadata)
    lat_range = radar.lateral_range
    fwd_range = radar.forward_range

    frames: List[dict] = []
    for pid in sorted(args.pairs):
        if pid < 0 or pid >= radar.radar_idx.size:
            print(f"  pair {pid} out of range; skipping.")
            continue
        l_idx = int(radar.lidar_idx[pid])
        print(f"  loading pair {pid}: radar + lidar scan {l_idx} ...", flush=True)
        try:
            bev, _ = radar.load_radar_panel(pid)
            pts = lidar.points_from_scan_idx(l_idx)
        except Exception as exc:  # noqa: BLE001 - surface per-pair load issues, keep going
            print(f"  pair {pid}: load failed ({exc}); skipping.")
            continue
        frames.append({"pair_idx": pid, "radar_bev": bev, "lidar_xyz": pts})

    if not frames:
        print("No usable pairs loaded.")
        return 2

    # Optional camera RGB context panel -> [camera | radar | lidar] in a row.
    cam_loader = None
    if args.show_camera:
        cam_bag = getattr(args, "camera_bag", None)
        cam_csv = getattr(args, "camera_sync_csv", None)
        if cam_bag and cam_csv and Path(cam_bag).is_file() and Path(cam_csv).is_file():
            from make_video import CameraLoader

            cam_loader = CameraLoader(cam_bag, cam_csv)
            cam_loader.preload([f["pair_idx"] for f in frames])
        else:
            print("WARNING: --show_camera needs camera_bag + camera_sync_csv; skipping camera panel.")

    state = {"i": 0}

    def current_pair_idx() -> Optional[int]:
        return frames[state["i"]]["pair_idx"]

    def radar_pick(lateral_y: float, forward_x: float) -> Optional[np.ndarray]:
        if not (lat_range[0] <= lateral_y <= lat_range[1] and fwd_range[0] <= forward_x <= fwd_range[1]):
            return None
        return np.array([forward_x, lateral_y], dtype=np.float64)  # [forward, lateral]

    def lidar_snap(lateral_y: float, forward_x: float) -> Optional[np.ndarray]:
        pts = frames[state["i"]]["lidar_xyz"]
        if pts.size == 0:
            return None
        d2 = (pts[:, 1] - lateral_y) ** 2 + (pts[:, 0] - forward_x) ** 2
        j = int(np.argmin(d2))
        if np.sqrt(d2[j]) > args.snap_radius_m:
            return None
        return np.array([pts[j, 0], pts[j, 1]], dtype=np.float64)  # [forward, lateral]

    # --- box-mode extractors ---
    def radar_box(x0: float, x1: float, y0: float, y1: float) -> Optional[np.ndarray]:
        # x=lateral, y=forward. Intensity-weighted centroid of the reflector peak.
        bev = frames[state["i"]]["radar_bev"]  # (n_fwd, n_lat)
        n_fwd, n_lat = bev.shape
        lat_axis = np.linspace(lat_range[0], lat_range[1], n_lat)
        fwd_axis = np.linspace(fwd_range[0], fwd_range[1], n_fwd)
        latm = (lat_axis >= x0) & (lat_axis <= x1)
        fwdm = (fwd_axis >= y0) & (fwd_axis <= y1)
        if latm.sum() < 1 or fwdm.sum() < 1:
            return None
        sub = np.where(np.isfinite(bev[np.ix_(fwdm, latm)]), bev[np.ix_(fwdm, latm)], np.nan)
        if not np.isfinite(sub).any():
            return None
        thr = np.nanpercentile(sub, args.box_radar_pct)
        w = np.clip(np.nan_to_num(sub, nan=0.0) - thr, 0.0, None)  # keep only the peak above background
        if w.sum() <= 0:
            return None
        LW, FW = np.meshgrid(lat_axis[latm], fwd_axis[fwdm])
        lat_c = float((LW * w).sum() / w.sum())
        fwd_c = float((FW * w).sum() / w.sum())
        return np.array([fwd_c, lat_c], dtype=np.float64)  # [forward, lateral]

    def lidar_box(x0: float, x1: float, y0: float, y1: float) -> Optional[np.ndarray]:
        pts = frames[state["i"]]["lidar_xyz"]
        if pts.size == 0:
            return None
        m = (pts[:, 1] >= x0) & (pts[:, 1] <= x1) & (pts[:, 0] >= y0) & (pts[:, 0] <= y1)
        if int(m.sum()) < args.min_box_points:
            return None
        q = pts[m]
        return np.array([np.median(q[:, 0]), np.median(q[:, 1])], dtype=np.float64)  # [forward, lateral]

    if cam_loader is not None:
        fig, (ax_cam, ax_r, ax_l) = plt.subplots(1, 3, figsize=(19, 6))
    else:
        ax_cam = None
        fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(14, 6))
    extent = [lat_range[0], lat_range[1], fwd_range[0], fwd_range[1]]
    # Auto-fit stays within the radar display range unless overridden.
    range_cap = float(args.max_range_m) if args.max_range_m else float(
        max(abs(lat_range[0]), abs(lat_range[1]), fwd_range[1])
    )

    def draw_frame() -> None:
        f = frames[state["i"]]
        pts = f["lidar_xyz"]
        # Same limits on both panels so radar and lidar stay at matching scale.
        if args.no_auto_range:
            lat_lim, fwd_lim = tuple(lat_range), tuple(fwd_range)
        else:
            lat_lim, fwd_lim = bev_view_limits(pts, range_cap)

        if ax_cam is not None:
            ax_cam.clear()
            cam = cam_loader.get_frame(f["pair_idx"])
            if cam is not None:
                ax_cam.imshow(cam)
            else:
                ax_cam.text(0.5, 0.5, "no camera frame\nfor this pair", ha="center", va="center")
            ax_cam.set_title("Camera RGB (context only)")
            ax_cam.axis("off")

        ax_r.clear()
        ax_l.clear()
        ax_r.imshow(
            f["radar_bev"], origin="lower", extent=extent, aspect="equal",
            cmap="viridis", interpolation="bilinear",
        )
        # Flip both panels together so left/right matches the camera calibrator
        # (physical). Both stay mutually consistent; data coords are unchanged.
        xlim = (lat_lim[0], lat_lim[1]) if args.no_flip_bev else (lat_lim[1], lat_lim[0])
        ax_r.set_xlim(*xlim)
        ax_r.set_ylim(*fwd_lim)
        ax_r.set_xlabel("Lateral Y (m)  [L/R matches camera]")
        ax_r.set_ylabel("Forward X (m)")
        ax_r.set_title(f"Radar BEV (pair {f['pair_idx']}) — click reflector")
        if pts.size:
            ax_l.scatter(pts[:, 1], pts[:, 0], s=1, c=pts[:, 2], cmap="viridis")
        ax_l.set_xlim(*xlim)
        ax_l.set_ylim(*fwd_lim)
        ax_l.set_xlabel("Lateral Y (m)  [L/R matches camera]")
        ax_l.set_ylabel("Forward X (m)")
        span = f"{lat_lim[1] - lat_lim[0]:.0f}x{fwd_lim[1] - fwd_lim[0]:.0f} m"
        ax_l.set_title(f"Lidar BEV ({span}) — click same object")
        ax_l.set_aspect("equal")
        fig.canvas.draw_idle()

    _picker = {"p": None}  # holder so advance() can reset box selectors after redraw

    def advance() -> bool:
        if state["i"] + 1 < len(frames):
            state["i"] += 1
            draw_frame()
            p = _picker["p"]
            if p is not None and hasattr(p, "reset_selectors"):
                p.reset_selectors()  # RectangleSelectors are removed by ax.clear()
            return True
        print("Already at last preloaded pair.")
        return False

    def on_solve() -> None:
        _solve_and_write(cs, args, corr_path)

    draw_frame()

    # Keep a strong reference: matplotlib holds event callbacks weakly, so an
    # unreferenced picker gets garbage-collected and clicks silently stop working.
    if args.box:
        from interactive import BoxPairPicker

        picker = BoxPairPicker(
            fig, ax_r, ax_l,
            extract_source_box=radar_box,
            extract_target_box=lidar_box,
            corr_set=cs,
            current_pair_idx=current_pair_idx,
            advance_frame=advance,
            on_solve=on_solve,
        )
        mode_msg = "  DRAG a box around the reflector in the RADAR panel, then the same object in LIDAR."
    else:
        from interactive import DualPanelPicker

        picker = DualPanelPicker(
            fig, ax_r, ax_l,
            extract_source=radar_pick,
            extract_target=lidar_snap,
            corr_set=cs,
            current_pair_idx=current_pair_idx,
            advance_frame=advance,
            on_solve=on_solve,
        )
        mode_msg = "  Click the reflector in the RADAR panel, then the same object in the LIDAR panel."
    _picker["p"] = picker
    fig._calib_picker = picker  # extra safety: lives as long as the figure
    print(
        "Interactive calibration ready.\n"
        f"{mode_msg}\n"
        "  u=undo  n=next pair  s=solve+save  q=quit"
    )
    plt.show()

    if cs.points:
        cs.save(corr_path)
        print(f"Saved {len(cs)} correspondences to {corr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
