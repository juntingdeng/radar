"""Attach Intel RealSense camera frames to existing radar/lidar sync pairs.

Reads ``sync_pairs.csv`` (radar_t / lidar_t per row) and maps each row to the
nearest color-frame header timestamp in the camera ROS bag.  Optionally
estimates a constant clock offset (or full affine scale+offset) so the camera
clock is aligned to the lidar or radar before matching — the same approach as
``sync_radar_lidar.py``.

Writes ``camera_sync_pairs.csv`` with added columns:
  ``pair_idx``, ``camera_idx`` (color frame, for downstream compat),
  ``camera_color_idx``, ``camera_depth_idx``,
  ``camera_t`` (raw), ``camera_t_mapped`` (after offset/skew),
  ``camera_delta_ms``.

Run once per capture after ``sync_radar_lidar.py``::

    python sync/sync_camera_pairs.py -d 2026.05.10/18-05-08 --fit_offset
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from camera_compat import (
    DEFAULT_COLOR_TOPIC,
    DEFAULT_DEPTH_TOPIC,
    build_camera_index,
    load_camera_index,
)
from dataset_config import add_dataset_arguments, apply_dataset_config
from sync_utils import (
    fit_time_affine,
    fit_time_offset,
    format_unix_utc,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Map RealSense camera frames onto radar/lidar sync pairs."
    )
    p.add_argument("--sync_csv", default=None, help="Input sync_pairs.csv")
    p.add_argument(
        "--camera_output_csv",
        default=None,
        help="Output CSV path (default: camera_sync_pairs.csv next to sync_csv)",
    )
    p.add_argument(
        "--camera_bag",
        default=None,
        help="RealSense ROS bag (camera/*.bag)",
    )
    p.add_argument(
        "--camera_index",
        default=None,
        help="Cached timestamp index (.npz). Default: camera_index.npz next to sync_csv",
    )
    p.add_argument(
        "--color_topic",
        default=DEFAULT_COLOR_TOPIC,
        help="ROS color image topic",
    )
    p.add_argument(
        "--depth_topic",
        default=DEFAULT_DEPTH_TOPIC,
        help="ROS depth image topic",
    )
    p.add_argument(
        "--target_time",
        choices=["lidar_t", "radar_t"],
        default="lidar_t",
        help="Sync-pair timestamp to match camera frames against (default: lidar_t).",
    )
    p.add_argument(
        "--fit_offset",
        action="store_true",
        help="Estimate constant camera->target clock offset before matching (scale=1).",
    )
    p.add_argument(
        "--fit_skew",
        action="store_true",
        help="Estimate affine camera->target map: target ~= scale*camera + offset.",
    )
    p.add_argument(
        "--offset_search_s",
        type=float,
        default=5.0,
        help="Max absolute offset (seconds) searched when --fit_offset / --fit_skew is on.",
    )
    p.add_argument(
        "--skew_search_ppm",
        type=float,
        default=200.0,
        help="Max absolute clock-rate error (ppm) searched when --fit_skew is on.",
    )
    p.add_argument(
        "--max_delta_ms",
        type=float,
        default=50.0,
        help="Max |target_t - camera_t_mapped| for a valid match (30 fps ~ 33 ms).",
    )
    p.add_argument(
        "--rebuild_index",
        action="store_true",
        help="Rescan bag and rebuild camera timestamp index.",
    )
    p.add_argument(
        "--camera_output_json",
        default=None,
        help="Optional camera sync summary JSON path.",
    )
    add_dataset_arguments(p)
    return p


def _default_camera_index_path(sync_csv: Path, bag_path: Path) -> Path:
    candidate = sync_csv.parent / "camera_index.npz"
    if candidate.parent.is_dir():
        return candidate
    return bag_path.with_suffix(bag_path.suffix + ".camera_index.npz")


def _print_timeline(
    target_label: str,
    target_ts: np.ndarray,
    camera_ts_raw: np.ndarray,
    *,
    camera_time_scale: float = 1.0,
    camera_offset_s: float = 0.0,
) -> float:
    """Print timeline spans and overlap; return overlap in seconds."""
    camera_mapped = camera_ts_raw * float(camera_time_scale) + float(camera_offset_s)
    overlap_s = float(
        min(target_ts[-1], camera_mapped[-1]) - max(target_ts[0], camera_mapped[0])
    )
    print(
        f"{target_label}: {format_unix_utc(target_ts[0])} .. {format_unix_utc(target_ts[-1])} "
        f"({target_ts[-1] - target_ts[0]:.1f} s, {target_ts.size} frames)"
    )
    print(
        f"Camera raw: {format_unix_utc(camera_ts_raw[0])} .. {format_unix_utc(camera_ts_raw[-1])} "
        f"({camera_ts_raw[-1] - camera_ts_raw[0]:.1f} s, {camera_ts_raw.size} frames)"
    )
    if camera_time_scale != 1.0 or camera_offset_s != 0.0:
        print(
            f"Camera mapped: {format_unix_utc(camera_mapped[0])} .. "
            f"{format_unix_utc(camera_mapped[-1])} "
            f"(scale={camera_time_scale:.9f}, offset={camera_offset_s:+.3f} s)"
        )
    if overlap_s > 0:
        print(f"Timeline overlap: {overlap_s:.1f} s")
    else:
        gap = max(target_ts[0], camera_mapped[0]) - min(target_ts[-1], camera_mapped[-1])
        print(
            f"ERROR: {target_label} and camera timelines do not overlap.\n"
            f"  Gap: {gap:.1f} s\n"
            "  --fit_offset cannot fix this. Check file paths or bag topic."
        )
    return overlap_s


def _nearest_match(
    target_ts: np.ndarray,
    frame_ts_mapped: np.ndarray,
    frame_ts_raw: np.ndarray,
    max_delta_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match each target timestamp to the nearest (mapped) camera frame.

    Returns (indices, raw_ts, mapped_ts, delta_ms); index is -1 when outside threshold.
    """
    if frame_ts_mapped.size == 0:
        nan = np.full(target_ts.shape, np.nan)
        return np.full(target_ts.shape, -1, dtype=np.int64), nan, nan, nan

    order = np.argsort(frame_ts_mapped)
    sorted_t = frame_ts_mapped[order]
    i = np.searchsorted(sorted_t, target_ts, side="left")
    i = np.clip(i, 1, sorted_t.size - 1)
    left, right = i - 1, i
    choose_right = np.abs(target_ts - sorted_t[left]) > np.abs(sorted_t[right] - target_ts)
    nearest_sorted = np.where(choose_right, right, left)
    nearest = order[nearest_sorted]
    dt = (target_ts - frame_ts_mapped[nearest]) * 1e3
    valid = np.abs(dt) <= float(max_delta_ms)

    out_idx = np.where(valid, nearest.astype(np.int64), np.int64(-1))
    out_raw = np.where(valid, frame_ts_raw[nearest], np.nan)
    out_mapped = np.where(valid, frame_ts_mapped[nearest], np.nan)
    out_dt = np.where(valid, dt, np.nan)
    return out_idx, out_raw, out_mapped, out_dt


def attach_camera_columns(
    rows: List[dict],
    *,
    color_ts: np.ndarray,
    depth_ts: np.ndarray,
    target_time: str,
    camera_time_scale: float = 1.0,
    camera_to_target_offset_s: float = 0.0,
    max_delta_ms: float,
) -> List[dict]:
    color_ts = np.asarray(color_ts, dtype=np.float64)
    depth_ts = np.asarray(depth_ts, dtype=np.float64)
    scale = float(camera_time_scale)
    offset = float(camera_to_target_offset_s)
    color_mapped = color_ts * scale + offset
    depth_mapped = depth_ts * scale + offset
    target_t = np.asarray([float(r[target_time]) for r in rows], dtype=np.float64)

    c_idx, c_raw, c_mapped, c_dt = _nearest_match(target_t, color_mapped, color_ts, max_delta_ms)
    d_idx, _dr, _dm, _dd = _nearest_match(target_t, depth_mapped, depth_ts, max_delta_ms)

    out: List[dict] = []
    for i, row in enumerate(rows):
        updated = dict(row)
        updated["pair_idx"] = i
        ci = int(c_idx[i])
        di = int(d_idx[i])
        updated["camera_color_idx"] = ci
        updated["camera_depth_idx"] = di
        updated["camera_idx"] = ci  # alias for generate_annotations / project_detections
        if ci >= 0:
            updated["camera_t"] = f"{c_raw[i]:.9f}"
            updated["camera_t_mapped"] = f"{c_mapped[i]:.9f}"
            updated["camera_delta_ms"] = f"{c_dt[i]:.3f}"
        else:
            updated["camera_t"] = ""
            updated["camera_t_mapped"] = ""
            updated["camera_delta_ms"] = ""
        out.append(updated)
    return out


def write_camera_csv(rows: List[dict], path: Path) -> None:
    fieldnames = [
        "pair_idx",
        "radar_idx",
        "lidar_idx",
        "radar_t",
        "lidar_t",
        "delta_ms",
        "camera_idx",
        "camera_color_idx",
        "camera_depth_idx",
        "camera_t",
        "camera_t_mapped",
        "camera_delta_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    apply_dataset_config(args, required=["sync_csv", "camera_bag"])

    sync_csv = Path(args.sync_csv)
    bag_path = Path(args.camera_bag)
    if not bag_path.is_file():
        raise SystemExit(f"Camera bag not found: {bag_path}")

    index_path = (
        Path(args.camera_index)
        if args.camera_index
        else _default_camera_index_path(sync_csv, bag_path)
    )
    if args.rebuild_index:
        build_camera_index(
            bag_path,
            color_topic=args.color_topic,
            depth_topic=args.depth_topic,
            output_path=index_path,
        )
    index = load_camera_index(index_path, bag_path=bag_path)
    color_ts = np.asarray(index["color_ts"], dtype=np.float64)
    depth_ts = np.asarray(index["depth_ts"], dtype=np.float64)

    if color_ts.size == 0:
        raise SystemExit("No camera color frames found in bag/index.")

    print(
        f"Reading camera timestamps from {bag_path.name} ..."
    )
    print(
        f"Camera: {color_ts.size} color / {depth_ts.size} depth frames "
        f"({format_unix_utc(color_ts[0])} .. {format_unix_utc(color_ts[-1])})"
    )

    with open(sync_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {sync_csv}")

    target_label = args.target_time[:-2]  # "lidar_t" -> "lidar"
    target_ts = np.asarray([float(r[args.target_time]) for r in rows], dtype=np.float64)

    # --- clock alignment (mirrors sync_radar_lidar.py) ---
    scale = 1.0
    offset_s = 0.0
    fit_median_error_s = None
    offset_at_boundary = False
    skew_at_boundary = False

    overlap_s = _print_timeline(target_label, target_ts, color_ts)

    if args.fit_skew or args.fit_offset:
        if overlap_s <= 0:
            print(
                f"Skipping clock fit because timelines do not overlap "
                f"(gap ~{abs(overlap_s):.0f} s)."
            )
        elif args.fit_skew:
            scale, offset_s, fit_median_error_s = fit_time_affine(
                target_ts,
                color_ts,
                max_abs_shift_s=args.offset_search_s,
                max_skew_ppm=args.skew_search_ppm,
            )
            skew_ppm = (scale - 1.0) * 1e6
            if abs(offset_s) >= args.offset_search_s - 1e-9:
                offset_at_boundary = True
                print(
                    f"WARNING: estimated offset hit the search boundary "
                    f"(+/- {args.offset_search_s} s). Increase --offset_search_s."
                )
            if abs(skew_ppm) >= args.skew_search_ppm - 1e-6:
                skew_at_boundary = True
                print(
                    f"WARNING: estimated skew hit the search boundary "
                    f"(+/- {args.skew_search_ppm} ppm). Increase --skew_search_ppm."
                )
            print(
                f"Affine fit: scale={scale:.9f} ({skew_ppm:+.3f} ppm), "
                f"offset={offset_s:.6f} s, "
                f"median NN error={fit_median_error_s * 1e3:.2f} ms"
            )
            offset_only = fit_time_offset(
                target_ts, color_ts, max_abs_shift_s=args.offset_search_s
            )
            print(f"Offset-only comparison: offset={offset_only:.6f} s (scale fixed at 1.0)")
        else:  # --fit_offset only
            offset_s = fit_time_offset(
                target_ts,
                color_ts,
                max_abs_shift_s=args.offset_search_s,
            )
            if abs(offset_s) >= args.offset_search_s - 1e-9:
                offset_at_boundary = True
                print(
                    f"WARNING: estimated offset hit the search boundary "
                    f"(+/- {args.offset_search_s} s). Increase --offset_search_s."
                )
            print(f"Offset fit: camera_to_{target_label}_offset={offset_s:.6f} s")

        overlap_s = _print_timeline(
            target_label,
            target_ts,
            color_ts,
            camera_time_scale=scale,
            camera_offset_s=offset_s,
        )

    skew_ppm = (scale - 1.0) * 1e6
    print(
        f"Time map: {target_label} ~= {scale:.9f} * camera + {offset_s:.6f} s "
        f"({skew_ppm:+.3f} ppm skew)"
    )

    updated = attach_camera_columns(
        rows,
        color_ts=color_ts,
        depth_ts=depth_ts,
        target_time=args.target_time,
        camera_time_scale=scale,
        camera_to_target_offset_s=offset_s,
        max_delta_ms=args.max_delta_ms,
    )
    matched = sum(1 for r in updated if int(r["camera_color_idx"]) >= 0)

    out_csv = (
        Path(args.camera_output_csv)
        if args.camera_output_csv
        else sync_csv.parent / "camera_sync_pairs.csv"
    )
    write_camera_csv(updated, out_csv)
    print(f"Camera frames: {color_ts.size}")
    print(f"Sync pairs:    {len(updated)}")
    print(f"Matched pairs: {matched}")
    print(f"Wrote: {out_csv}")

    matched_deltas = [
        float(r["camera_delta_ms"])
        for r in updated
        if r.get("camera_delta_ms") not in (None, "")
    ]
    summary: dict = {
        "camera_bag": str(bag_path.resolve()),
        "camera_index": str(index_path.resolve()),
        "color_topic": args.color_topic,
        "depth_topic": args.depth_topic,
        "target_time": args.target_time,
        "color_frames": int(color_ts.size),
        "depth_frames": int(depth_ts.size),
        "sync_pairs": len(updated),
        "camera_matched_pairs": matched,
        "max_delta_ms_threshold": float(args.max_delta_ms),
        "estimated_camera_time_scale": float(scale),
        "estimated_camera_to_target_offset_s": float(offset_s),
        "estimated_skew_ppm": float(skew_ppm),
        "fit_median_error_s": fit_median_error_s,
        "offset_at_search_boundary": bool(offset_at_boundary),
        "skew_at_search_boundary": bool(skew_at_boundary),
        "color_t_start": float(color_ts[0]),
        "color_t_end": float(color_ts[-1]),
    }
    if matched_deltas:
        abs_d = np.abs(np.asarray(matched_deltas, dtype=np.float64))
        summary["median_abs_delta_ms"] = float(np.median(abs_d))
        summary["p95_abs_delta_ms"] = float(np.percentile(abs_d, 95))
        summary["max_abs_delta_ms"] = float(np.max(abs_d))
        print(
            f"Camera match error: "
            f"median={summary['median_abs_delta_ms']:.2f} ms, "
            f"p95={summary['p95_abs_delta_ms']:.2f} ms"
        )

    if args.camera_output_json:
        json_path = Path(args.camera_output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote: {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
