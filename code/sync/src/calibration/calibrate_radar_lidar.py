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

The radar panel is a top-down (forward, lateral) map either way; ``--radar_doppler_mode
moving`` rebuilds it from the NON-zero Doppler bins, so a walking person or a passing
car is what lights up instead of static clutter (pair it with ``--radar_aoa dbf_nci``,
which integrates power over those bins). ``static`` keeps only the near-zero-velocity
bins (walls, poles, parked cars); ``all`` (default) is every bin.

Every pick also snapshots the annotated figure to
``<calibration dir>/calibration_labels/radar_lidar/`` (one PNG per pair, boxes/marks
included) and those frames are stitched into a video on exit (``--no_label_frames``
to skip, ``--label_dir`` / ``--label_fps`` to control).

Collect >= 3 well-separated correspondences (2 is the bare minimum).

Run::

    python code/sync/calibration/calibrate_radar_lidar.py \
        -d 2026.05.10/18-05-08 --pairs 500 1500 3000 --z_offset_m 0.0

Auto mode (no GUI)
------------------
``--auto_movers`` builds correspondences from WALKING PEOPLE instead of hand picks: for
each pair it lifts the nearest YOLO ``person`` on the D435 color into the lidar frame
(color->depth-box->lidar-in-frustum), and matches it to the moving-target radar BEV
cluster CENTROID (SNR-gated, gated around the current ``radar_to_lidar`` seed). It then
solves yaw with EQUAL weighting — azimuth is exactly what constrains yaw, so ``--range_weighted``
/ ``--yaw_search`` are the wrong tool here and are ignored — and reports bootstrap /
leave-one-out spread so you can see whether yaw is actually pinned. Needs ``camera_to_lidar``
and a seed ``radar_to_lidar`` already in the calibration JSON. Give ``--pairs`` spanning
where people walk near (< ``--near_m``), ideally at a spread of bearings::

    python code/sync/src/calibration/calibrate_radar_lidar.py \
        -d 2026.05.08/17-34-27 --auto_movers --pairs 4000 4500 5490 5496 5508
"""

from __future__ import annotations

import argparse
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
from extrinsic_io import (  # noqa: E402
    get_extrinsic,
    has_extrinsic,
    reuse_calibration,
    write_extrinsic,
)
from frame_capture import LabeledFrameRecorder, add_label_arguments  # noqa: E402
from rigid_solve import (  # noqa: E402
    fit_with_rejection,
    fit_yaw_deg,
    planar_fit_to_4x4,
    umeyama_2d,
    weighted_umeyama_2d,
    yaw_bruteforce_2d,
    yaw_stability_2d,
)


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
    # g.add_argument(
    #     "--no_flip_bev",
    #     action="store_true",
    #     help="Don't mirror the BEVs horizontally. By default BOTH panels are flipped so "
    #     "left/right matches the camera calibrator's physical orientation (display only).",
    # )
    g.add_argument(
        "--reject_outliers_m",
        type=float,
        default=0.0,
        help="If >0, drop correspondences whose fit residual exceeds this (m) and refit.",
    )
    g.add_argument(
        "--range_weighted",
        action="store_true",
        help="Fit with radar RANGE trusted >> azimuth (mmWave has coarse cross-range). "
        "Down-weights each correspondence's tangential/azimuth error by --sigma_ratio.",
    )
    g.add_argument(
        "--sigma_ratio",
        type=float,
        default=10.0,
        help="Azimuth/range uncertainty ratio for --range_weighted (higher = trust "
        "range more). Default 10.",
    )
    g.add_argument(
        "--yaw_search",
        type=float,
        nargs=2,
        metavar=("LO_DEG", "HI_DEG"),
        default=None,
        help="Brute-force yaw over [LO,HI] deg (range-weighted translation solved per "
        "yaw). Bounds yaw to a physically-plausible range; overrides the free fit.",
    )
    g.add_argument("--yaw_step", type=float, default=0.25, help="Yaw brute-force step (deg).")
    # Canonical calibration radar chain (shared with the validation overlay and
    # auto-movers): TI clutter removal + windows + zero-Doppler notch, moving-target
    # Doppler bins, dbf_nci angle estimation. Override with --radar_doppler_mode all /
    # --radar_aoa capon to pick STATIC reflectors instead.
    p.set_defaults(radar_processing="ti", radar_doppler_mode="moving", radar_aoa="dbf_nci")
    g.add_argument(
        "--show_camera",
        action="store_true",
        help="Add a camera RGB context panel (leftmost) so [camera | radar | lidar] are "
        "in a row — helps identify what a radar reflector actually is. Scans the bag once.",
    )
    g.add_argument(
        "--box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Box mode (DEFAULT): DRAG a box around the reflector in each panel. Radar uses the "
        "intensity-weighted centroid (above background); lidar uses the point centroid. "
        "Pass --no-box for the old single-click (single-point) picking.",
    )
    a = p.add_argument_group("auto-movers (non-interactive pedestrian solve)")
    a.add_argument(
        "--auto_movers",
        action="store_true",
        help="Non-interactive: build correspondences from WALKING PEOPLE (YOLO person on "
        "D435 color -> lidar-in-frustum position) matched to the moving-target radar BEV "
        "cluster CENTROID (SNR-gated, gated around the current radar_to_lidar seed), then "
        "solve yaw with EQUAL weighting (azimuth is what fixes yaw, so do NOT range-weight "
        "it) and report bootstrap/leave-one-out spread. Needs camera_to_lidar + a seed "
        "radar_to_lidar in the calibration JSON.",
    )
    a.add_argument(
        "--pair_range", type=int, nargs="+", metavar="LO HI [STRIDE]", default=None,
        help="Sweep pairs LO..HI (inclusive) at STRIDE (default 1) instead of listing "
        "--pairs. Use this to run the auto solve over a whole capture; the near-range + "
        "SNR gates keep only frames with a usable near pedestrian.",
    )
    a.add_argument("--near_m", type=float, default=12.0,
                   help="Only use people within this range (m); mmWave sees pedestrians reliably "
                   "only in the near field.")
    a.add_argument("--gate_m", type=float, default=3.0,
                   help="Radius (m) around the seed-predicted person location to search the radar "
                   "BEV for its return cluster.")
    a.add_argument("--snr_db", type=float, default=4.0,
                   help="Reject a frame unless the gated radar cluster's peak exceeds the BEV "
                   "median by this many dB (drops frames with no real return).")
    a.add_argument(
        "--accumulate", action="store_true",
        help="Merge this run's correspondences with those already saved (union by pair), "
        "then solve on the pooled set. Lets you build up coverage over many runs / segments "
        "without re-processing, and pools bearing spread across walker passes.",
    )
    a.add_argument("--person_class", default="person", help="YOLO class name to track.")
    a.add_argument("--yolo_model", default="yolov8n.pt", help="Ultralytics model name/path.")
    a.add_argument("--frame_cache", default=None, help="Local dir for extracted camera frames.")
    a.add_argument("--auto_plot", default=None, help="Where to save the QC scatter (default: "
                   "<calib dir>/calibration_labels/radar_lidar/auto_movers.png).")
    add_label_arguments(p)
    g.add_argument("--min_box_points", type=int, default=10, help="Min lidar points in a box (box mode).")
    g.add_argument(
        "--box_radar_pct",
        type=float,
        default=75.0,
        help="Box mode: radar centroid weights only cells above this percentile of the "
        "box (isolates the reflector peak from the diffuse envelope).",
    )
    return p


def _bf_rot(src, tgt, lo, hi, args):
    """Yaw brute-force -> (rotation[1,0], rotation[0,0]) so arctan2(...) gives yaw."""
    f = yaw_bruteforce_2d(src, tgt, yaw_lo_deg=lo, yaw_hi_deg=hi,
                          step_deg=args.yaw_step, sigma_ratio=args.sigma_ratio)
    return f.rotation[1, 0], f.rotation[0, 0]


def _solve_and_write(cs: CorrespondenceSet, args, corr_path: Path,
                     *, write_key: str = "radar_to_lidar", method: Optional[str] = None) -> int:
    src, tgt = cs.as_arrays()
    if len(cs) < 2:
        print(f"Need >= 2 correspondences to solve; have {len(cs)}. Nothing written.")
        return 1
    if args.yaw_search:
        lo, hi = args.yaw_search

        def solver(s, t):
            return yaw_bruteforce_2d(s, t, yaw_lo_deg=lo, yaw_hi_deg=hi,
                                     step_deg=args.yaw_step, sigma_ratio=args.sigma_ratio)
        # Best yaw on the full data, plus a bootstrap estimate of how reliable it is:
        # resample correspondences with replacement, refit yaw, report the spread.
        best_yaw = np.degrees(np.arctan2(*_bf_rot(src, tgt, lo, hi, args)))
        n = len(src)
        boot = []
        if n >= 3:
            rgen = np.random.default_rng(0)
            for _ in range(300):
                idx = rgen.integers(0, n, n)
                boot.append(np.degrees(np.arctan2(*_bf_rot(src[idx], tgt[idx], lo, hi, args))))
            boot = np.array(boot)
            std = float(np.std(boot))
            p16, p84 = np.percentile(boot, [16, 84])
            print(f"Yaw brute-force [{lo:+.1f},{hi:+.1f}] deg, step {args.yaw_step}, "
                  f"sigma_ratio={args.sigma_ratio}: yaw={best_yaw:+.2f} deg "
                  f"(bootstrap +/-{std:.2f} deg, 16-84%=[{p16:+.2f},{p84:+.2f}])")
            if std > 2.0:
                print("  NOTE: large bootstrap spread -> yaw weakly constrained; "
                      "spread targets wider in azimuth / add more picks.")
        else:
            print(f"Yaw brute-force: yaw={best_yaw:+.2f} deg (need >=3 picks for bootstrap).")
    elif args.range_weighted:
        def solver(s, t):
            return weighted_umeyama_2d(s, t, sigma_ratio=args.sigma_ratio)
        print(f"Range-weighted fit (sigma_ratio={args.sigma_ratio}).")
    else:
        solver = umeyama_2d
    if args.reject_outliers_m > 0:
        fit, keep = fit_with_rejection(
            solver, src, tgt, max_residual=args.reject_outliers_m, min_points=3
        )
        dropped = [int(i) for i in np.where(~keep)[0]]
        if dropped:
            print(f"Robust fit dropped {len(dropped)} correspondence(s) "
                  f"> {args.reject_outliers_m} m: {dropped}")
    else:
        fit = solver(src, tgt)
    # For the free/range-weighted solvers, show how well yaw is actually pinned by the
    # data (yaw_search prints its own bootstrap above). A wide spread => yaw weakly
    # constrained (add picks with more bearing spread), not a trustworthy number.
    if not args.yaw_search and len(src) >= 3:
        st = yaw_stability_2d(solver, src, tgt)
        print(f"Yaw stability: {st['yaw_deg']:+.2f} deg (bootstrap +/-{st['boot_std']:.2f}, "
              f"16-84%=[{st['boot_p16']:+.2f},{st['boot_p84']:+.2f}]; "
              f"leave-one-out {st['loo_min']:+.2f}..{st['loo_max']:+.2f}, std {st['loo_std']:.2f})")
        if st["boot_std"] > 3.0:
            print("  NOTE: wide yaw spread -> weakly constrained; spread correspondences "
                  "over more bearings / add picks.")
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
    extra = {} if method is None else {"method": method}
    write_extrinsic(
        args.calibration_json,
        key=write_key,
        matrix=mat,
        n_points=fit.n_points,
        rms_m=fit.rms,
        max_error_m=fit.max_error,
        correspondences_path=corr_path,
        note=(
            f"planar (forward,lateral) fit"
            f"{' [range-weighted sigma_ratio=' + str(args.sigma_ratio) + ']' if args.range_weighted else ''}"
            f"; radar_processing={getattr(args, 'radar_processing', 'legacy')}"
            f"; doppler_mode={getattr(args, 'radar_doppler_mode', 'all')}"
            f"; aoa={getattr(args, 'radar_aoa', None) or 'capon'}"
            f"; z_offset={args.z_offset_m} m supplied manually"
        ),
        **extra,
    )
    return 0


def _load_pair_rows(csv_path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["pair_idx"])] = row
    return rows


def _radar_centroid_snr(bev, lat_axis, fwd_axis, pred, *, gate_m: float):
    """Intensity-weighted centroid of the radar return near ``pred`` + its SNR (dB).

    ``pred`` is the seed-predicted (forward, lateral) of the target in the radar frame.
    Returns ((forward, lateral) centroid or None, snr_db). The centroid uses only cells
    above the gate's 80th percentile (isolates the peak from the diffuse envelope); the
    SNR is the gate peak minus the whole-BEV median (the moving BEV is already in dB).
    """
    LAT, FWD = np.meshgrid(lat_axis, fwd_axis)
    fin = np.isfinite(bev)
    gate = fin & (np.hypot(FWD - pred[0], LAT - pred[1]) <= float(gate_m))
    if int(gate.sum()) < 5:
        return None, -np.inf
    med = float(np.nanmedian(bev[fin]))
    gmax = float(np.max(bev[gate]))
    thr = np.nanpercentile(bev[gate], 80)
    w = np.nan_to_num(np.where(gate, bev, np.nan) - thr, nan=0.0)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return None, -np.inf
    cen = np.array([(FWD * w).sum() / w.sum(), (LAT * w).sum() / w.sum()])
    return cen, gmax - med


def _auto_mover_correspondences(args, cs: CorrespondenceSet) -> CorrespondenceSet:
    """Fill ``cs`` with SNR-gated pedestrian centroid correspondences (radar->lidar).

    Person (target, lidar frame) from YOLO on D435 color lifted through the depth-image
    box into the lidar-in-frustum cloud; radar point (source, radar frame) from the
    moving-target BEV cluster centroid gated around the seed radar_to_lidar prediction.
    """
    import json

    # camera_projection helpers (person lifting + frame cache/seeking).
    _cam_dir = _SYNC_DIR / "camera_projection"
    if str(_cam_dir) not in sys.path:
        sys.path.insert(0, str(_cam_dir))
    from yolo_to_lidar import (
        _color_box_to_depth, _lidar_frustum_box, _lidar_points_at,
        _extract_frames, _frame_cache_dir,
    )
    from lib.camera_io import DEFAULT_COLOR_TOPIC
    from lib.lidar_io import open_pcap_scan_source, sensor_info_from_source
    from lib.bev_render import compute_range_azimuth, range_azimuth_to_bev
    from make_video import RadarLoader, load_radar_adc_frame

    with open(args.calibration_json, "r", encoding="utf-8") as f:
        calib = json.load(f)
    for k in ("intrinsics", "color_intrinsics"):
        if k not in calib:
            raise SystemExit(f"--auto_movers needs '{k}' in the calibration JSON.")
    if not has_extrinsic(calib, "camera_to_lidar"):
        raise SystemExit("--auto_movers needs 'camera_to_lidar' (or 'camera_to_lidar_auto').")
    if not has_extrinsic(calib, "radar_to_lidar"):
        raise SystemExit("--auto_movers needs a seed 'radar_to_lidar' (or 'radar_to_lidar_auto').")
    di, ci = calib["intrinsics"], calib["color_intrinsics"]
    dcal = (di["fx"], di["fy"], di["cx"], di["cy"])
    ccal = (ci["fx"], ci["fy"], ci["cx"], ci["cy"])
    # Lift via the BEST camera->lidar (prefer the ICP-refined camera_to_lidar_auto over the
    # hand-picked seed): the pedestrian's lidar-frame position is the radar-fit TARGET, so a
    # better camera calibration directly improves the radar solve.
    cam2lid, cam_src = get_extrinsic(calib, "camera_to_lidar")
    print(f"auto_movers: lifting pedestrians via '{cam_src}' (camera->lidar).")
    lid2cam = np.linalg.inv(cam2lid)
    # Gate around the current best estimate (auto refinement if one already exists).
    R2L, seed_src = get_extrinsic(calib, "radar_to_lidar")
    print(f"auto_movers: gating around '{seed_src}' seed.")
    seed_rot, seed_t = R2L[:2, :2], R2L[:2, 3]

    # Moving-target radar unless the user asked for something else.
    if getattr(args, "radar_doppler_mode", "all") in (None, "all"):
        args.radar_doppler_mode = "moving"
    if getattr(args, "radar_aoa", None) is None:
        args.radar_aoa = "dbf_nci"
    print(f"auto_movers: radar_processing={args.radar_processing}, "
          f"doppler_mode={args.radar_doppler_mode}, aoa={args.radar_aoa}")

    rows = _load_pair_rows(Path(args.camera_sync_csv))
    plan = []
    for pid in sorted(args.pairs):
        r = rows.get(pid)
        if not r:
            continue
        cidx = int(r.get("camera_color_idx", r.get("camera_idx", -1)))
        lidx = int(r.get("lidar_idx", -1))
        if min(cidx, lidx) >= 0:
            plan.append((pid, cidx, lidx))
    if not plan:
        raise SystemExit("auto_movers: no usable pairs (missing color/lidar idx).")

    cache = _frame_cache_dir(args)
    bag_ts = None
    if args.camera_index and Path(args.camera_index).is_file():
        idx = np.load(args.camera_index)
        bag_ts = idx["color_bag_ts_ns"] if "color_bag_ts_ns" in idx else None
    color_map = _extract_frames(args.camera_bag, DEFAULT_COLOR_TOPIC,
                                sorted({c for _, c, _ in plan}), cache, "color", bag_ts)

    lsrc = open_pcap_scan_source(args.lidar_pcap, args.lidar_metadata, index=True)
    lmeta = sensor_info_from_source(lsrc)
    radar = RadarLoader(args)
    if hasattr(radar, "display_bev_kw"):
        radar.display_bev_kw["res"] = float(getattr(args, "radar_bev_res", 0.1))
    lat_range, fwd_range = radar.lateral_range, radar.forward_range

    from ultralytics import YOLO
    model = YOLO(args.yolo_model)
    name_to_id = {v: k for k, v in model.names.items()}
    if args.person_class not in name_to_id:
        raise SystemExit(f"class '{args.person_class}' not in the YOLO model.")
    cls_id = name_to_id[args.person_class]

    n_keep = 0
    for pid, cidx, lidx in plan:
        color = color_map.get(cidx)
        if color is None:
            continue
        try:
            lpts = _lidar_points_at(lsrc, lmeta, lidx)
        except Exception as exc:  # noqa: BLE001
            print(f"  pair {pid}: lidar read failed ({exc})")
            continue
        res = model.predict(color[..., ::-1], conf=0.25, classes=[cls_id], verbose=False)[0]
        best = None
        for b in res.boxes:
            dbox = _color_box_to_depth([float(v) for v in b.xyxy[0].tolist()], ccal, dcal)
            box = _lidar_frustum_box(lpts, lid2cam, dcal, dbox, min_pts=5, gap_m=2.0)
            if box is None:
                continue
            fwd = 0.5 * (box["forward"][0] + box["forward"][1])
            lat = 0.5 * (box["lateral"][0] + box["lateral"][1])
            rng = float(np.hypot(fwd, lat))
            if rng < args.near_m and (best is None or rng < best[0]):
                best = (rng, np.array([fwd, lat]))
        if best is None:
            continue
        person = best[1]
        adc = load_radar_adc_frame(args.radar_h5, radar.adc_frame_for_pair(pid), radar.radar)
        raz = compute_range_azimuth(adc, radar.radar, angle="Azimuth",
                                    processing=args.radar_processing,
                                    doppler_mode=args.radar_doppler_mode,
                                    aoa_method=args.radar_aoa)
        bev = range_azimuth_to_bev(raz, radar.radar, lateral_range=lat_range,
                                   forward_range=fwd_range, res=float(getattr(args, "radar_bev_res", 0.1)))
        n_fwd, n_lat = bev.shape
        lat_axis = np.linspace(lat_range[0], lat_range[1], n_lat)
        fwd_axis = np.linspace(fwd_range[0], fwd_range[1], n_fwd)
        pred = seed_rot.T @ (person - seed_t)  # person in radar frame per the seed
        cen, snr = _radar_centroid_snr(bev, lat_axis, fwd_axis, pred, gate_m=args.gate_m)
        ok = cen is not None and snr >= args.snr_db
        print(f"  pair {pid}: person(fwd={person[0]:.1f},lat={person[1]:.1f},r={best[0]:.1f}) "
              f"snr={snr:5.1f}dB {'KEEP' if ok else 'drop'}"
              + (f" centroid(fwd={cen[0]:.1f},lat={cen[1]:.1f})" if cen is not None else ""))
        if ok:
            cs.add(cen.tolist(), person.tolist(), pid, note=f"snr={snr:.1f}dB")
            n_keep += 1
    print(f"auto_movers: {n_keep} SNR-gated centroid correspondence(s).")
    return cs


def _auto_movers_plot(args, cs: CorrespondenceSet, fit, seed_R2L: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src, tgt = cs.as_arrays()
    srot, st = seed_R2L[:2, :2], seed_R2L[:2, 3]  # seed captured BEFORE the solve wrote
    sy, uy = fit_yaw_deg_from_R(srot), fit_yaw_deg(fit)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(tgt[:, 1], tgt[:, 0], c="k", s=50, label="person (lidar)", zorder=3)
    sp = src @ srot.T + st
    ax.scatter(sp[:, 1], sp[:, 0], c="tab:orange", marker="x", s=50, label=f"radar @seed ({sy:+.1f}°)")
    ap = src @ fit.rotation.T + fit.translation
    ax.scatter(ap[:, 1], ap[:, 0], c="tab:green", marker="+", s=80, label=f"radar @fit ({uy:+.1f}°)")
    for i in range(len(src)):
        ax.plot([tgt[i, 1], ap[i, 1]], [tgt[i, 0], ap[i, 0]], "0.85", lw=0.6, zorder=1)
    ax.set_xlabel("Lateral Y (m)"); ax.set_ylabel("Forward X (m)")
    ax.set_aspect("equal"); ax.invert_xaxis(); ax.legend(fontsize=9)
    ax.set_title(f"auto_movers radar->lidar: fit {uy:+.1f}° vs seed {sy:+.1f}°, RMS {fit.rms:.2f} m")
    out = Path(args.auto_plot or (Path(args.calibration_json).parent
               / "calibration_labels" / "radar_lidar" / "auto_movers.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"auto_movers: wrote {out}")


def fit_yaw_deg_from_R(rot2: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(rot2[1, 0], rot2[0, 0])))


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())

    # --- auto-movers path: non-interactive pedestrian solve (equal-weight yaw) ---
    if getattr(args, "auto_movers", False):
        if not args.pairs and getattr(args, "pair_range", None):
            pr = args.pair_range
            lo, hi = int(pr[0]), int(pr[1])
            step = int(pr[2]) if len(pr) > 2 else 1
            args.pairs = list(range(lo, hi + 1, max(1, step)))
            print(f"--auto_movers sweeping pairs {lo}..{hi} step {step} ({len(args.pairs)} pairs).")
        if not args.pairs:
            print("--auto_movers needs --pairs or --pair_range (sync pair indices "
                  "spanning where people walk).")
            return 2
        if args.range_weighted or args.yaw_search:
            print("NOTE: --auto_movers ignores --range_weighted/--yaw_search "
                  "(azimuth is what constrains yaw; using equal-weight umeyama).")
            args.range_weighted = False
            args.yaw_search = None
        # Auto results are kept SEPARATE from the hand-picked ones: distinct key
        # (radar_to_lidar_auto) and correspondences file, so the manual solve survives.
        corr_path = Path(
            args.correspondences
            or (Path(args.calibration_json).parent / "correspondences" / "radar_lidar_auto.json")
        )
        cs = CorrespondenceSet(
            kind="radar_lidar", dim=2,
            source_frame="radar_sensor_topdown", target_frame="lidar_sensor_topdown",
            dataset=getattr(args, "dataset", "") or "",
        )
        import json as _json
        # capture the reference (current best) before the solve writes the auto key
        seed_R2L, _seed_src = get_extrinsic(_json.load(open(args.calibration_json)), "radar_to_lidar")
        _auto_mover_correspondences(args, cs)
        if args.accumulate and corr_path.is_file():
            prev = CorrespondenceSet.load(corr_path)
            have = {p.pair_idx for p in cs.points}
            added = sum(
                (cs.add(p.source, p.target, p.pair_idx, p.note) or 1)
                for p in prev.points if p.pair_idx not in have
            )
            print(f"auto_movers: accumulated {added} prior correspondence(s) from "
                  f"{corr_path.name} -> {len(cs)} total.")
        if len(cs) < 2:
            print("auto_movers: too few correspondences to solve; nothing written.")
            return 1
        rc = _solve_and_write(
            cs, args, corr_path,
            write_key="radar_to_lidar_auto",
            method="auto-movers: SNR-gated pedestrian centroids (moving-target BEV) + "
                   "equal-weight Umeyama",
        )
        try:
            src, tgt = cs.as_arrays()
            _auto_movers_plot(args, cs, umeyama_2d(src, tgt), seed_R2L)
        except Exception as exc:  # noqa: BLE001 - plot is best-effort
            print(f"auto_movers: plot skipped ({exc})")
        return rc

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
        or (Path(args.calibration_json).parent / "correspondences" / "radar_lidar_handpicked.json")
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

    from lib.bev_render import LidarScanReader, radar_bev_vlimits
    from make_video import RadarLoader

    radar = RadarLoader(args)
    dop_mode = str(getattr(args, "radar_doppler_mode", "all") or "all")
    if dop_mode != "all":
        print(
            f"Radar panel: doppler_mode={dop_mode} "
            f"(min_bins={args.radar_doppler_min_bins}), aoa={args.radar_aoa or 'capon'}"
        )
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

    # Labeled frames: every pick re-snapshots the annotated figure for the current
    # pair, so each PNG ends up holding all marks made on that pair; they are
    # stitched into a video once picking ends.
    label_dir = Path(
        args.label_dir
        or (Path(args.calibration_json).parent / "calibration_labels" / "radar_lidar")
    )
    recorder = LabeledFrameRecorder(label_dir, enabled=not args.no_label_frames)
    if recorder.enabled:
        print(f"Labeled frames -> {label_dir}", flush=True)

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

    dop_tag = "" if dop_mode == "all" else f" [doppler: {dop_mode}]"

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
        _vmin, _vmax = radar_bev_vlimits(f["radar_bev"])  # shared 60-99.5 pct contrast
        ax_r.imshow(
            f["radar_bev"], origin="lower", extent=extent, aspect="equal",
            cmap="viridis", interpolation="bilinear", vmin=_vmin, vmax=_vmax,
        )
        # Flip both panels together so left/right matches the camera calibrator
        # (physical). Both stay mutually consistent; data coords are unchanged.
        xlim = (lat_lim[1], lat_lim[0])
        # xlim = (lat_lim[0], lat_lim[1]) if args.no_flip_bev else (lat_lim[1], lat_lim[0])
        ax_r.set_xlim(*xlim)
        ax_r.set_ylim(*fwd_lim)
        ax_r.set_xlabel("Lateral Y (m)  [L/R matches camera]")
        ax_r.set_ylabel("Forward X (m)")
        ax_r.set_title(f"Radar BEV{dop_tag} (pair {f['pair_idx']}) — click reflector")
        if pts.size:
            # Unified clean look (matches viz_labels / calibrate_camera_lidar): uniform gray,
            # small points, range-cropped -- clearer for picking than the height-colored scatter.
            m = np.linalg.norm(pts[:, :2], axis=1) <= range_cap
            ax_l.scatter(pts[m, 1], pts[m, 0], s=2, c="0.6")
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

    def record_frame() -> None:
        recorder.capture(fig, current_pair_idx())

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
            on_record=record_frame,
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
            on_record=record_frame,
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
    if len(recorder):
        print(f"Saved {len(recorder)} labeled frame(s) in {label_dir}")
        recorder.write_video(args.label_video, fps=args.label_fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
