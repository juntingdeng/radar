"""Attach Intel RealSense camera frames to existing radar/lidar sync pairs.

Reads ``sync_pairs.csv`` (radar_t per row) and maps each row to the nearest
color-frame header timestamp in the camera ROS bag. Writes an augmented CSV with
``camera_color_idx``, ``camera_depth_idx``, ``camera_t``, ``camera_delta_ms``.

Run once per capture after ``sync_radar_lidar.py``::

    python sync/sync_camera_pairs.py -d 2026.05.10/18-05-08 --rebuild_index
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
    nearest_frame_indices,
)
from dataset_config import add_dataset_arguments, apply_dataset_config
from sync_utils import format_unix_utc


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Map RealSense camera frames onto radar/lidar sync pairs."
    )
    p.add_argument("--sync_csv", default=None, help="Input sync_pairs.csv")
    p.add_argument(
        "--output_csv",
        default=None,
        help="Output CSV (default: overwrite sync_csv with camera columns added)",
    )
    p.add_argument(
        "--camera_bag",
        default=None,
        help="RealSense ROS bag under {data_root}/camera/*.bag",
    )
    p.add_argument(
        "--camera_index",
        default=None,
        help="Cached timestamp index (.npz). Default: next to bag or res/.../camera_index.npz",
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
        "--max_delta_ms",
        type=float,
        default=50.0,
        help="Max |radar_t - camera_t| for a valid match (30 fps ~ 33 ms)",
    )
    p.add_argument(
        "--rebuild_index",
        action="store_true",
        help="Rescan bag and rebuild camera timestamp index",
    )
    p.add_argument(
        "--output_json",
        default=None,
        help="Optional camera sync summary JSON path",
    )
    add_dataset_arguments(p)
    return p


def _default_camera_index_path(sync_csv: Path, bag_path: Path) -> Path:
    if sync_csv.parent.name and (sync_csv.parent / "camera_index.npz").parent.is_dir():
        return sync_csv.parent / "camera_index.npz"
    return bag_path.with_suffix(bag_path.suffix + ".camera_index.npz")


def attach_camera_columns(
    rows: List[dict],
    *,
    color_ts: np.ndarray,
    depth_ts: np.ndarray,
    max_delta_ms: float,
) -> List[dict]:
    radar_t = np.asarray([float(r["radar_t"]) for r in rows], dtype=np.float64)
    color_idx, camera_t, color_delta = nearest_frame_indices(
        radar_t, color_ts, max_delta_ms=max_delta_ms
    )
    depth_idx, _depth_t, depth_delta = nearest_frame_indices(
        radar_t, depth_ts, max_delta_ms=max_delta_ms
    )

    out: List[dict] = []
    for i, row in enumerate(rows):
        updated = dict(row)
        updated["camera_color_idx"] = int(color_idx[i])
        updated["camera_depth_idx"] = int(depth_idx[i])
        if color_idx[i] >= 0:
            updated["camera_t"] = f"{camera_t[i]:.9f}"
            updated["camera_delta_ms"] = f"{color_delta[i]:.3f}"
        else:
            updated["camera_t"] = ""
            updated["camera_delta_ms"] = ""
        out.append(updated)
    return out


def write_camera_csv(rows: List[dict], path: Path) -> None:
    base_fields = ["radar_idx", "lidar_idx", "radar_t", "lidar_t", "delta_ms"]
    extra = ["camera_color_idx", "camera_depth_idx", "camera_t", "camera_delta_ms"]
    fieldnames = base_fields + extra
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

    index_path = Path(args.camera_index) if args.camera_index else _default_camera_index_path(
        sync_csv, bag_path
    )
    if args.rebuild_index:
        build_camera_index(
            bag_path,
            color_topic=args.color_topic,
            depth_topic=args.depth_topic,
            output_path=index_path,
        )
    index = load_camera_index(index_path, bag_path=bag_path)
    color_ts = index["color_ts"]
    depth_ts = index["depth_ts"]

    print(
        f"Camera: {len(color_ts)} color / {len(depth_ts)} depth frames "
        f"({format_unix_utc(color_ts[0])} .. {format_unix_utc(color_ts[-1])})"
    )

    with open(sync_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {sync_csv}")

    updated = attach_camera_columns(
        rows,
        color_ts=color_ts,
        depth_ts=depth_ts,
        max_delta_ms=args.max_delta_ms,
    )
    matched = sum(1 for r in updated if int(r["camera_color_idx"]) >= 0)
    deltas = [
        float(r["camera_delta_ms"])
        for r in updated
        if r.get("camera_delta_ms") not in (None, "")
    ]

    out_csv = Path(args.output_csv) if args.output_csv else sync_csv
    write_camera_csv(updated, out_csv)
    print(f"Wrote {len(updated)} rows ({matched} with camera match) -> {out_csv}")

    if deltas:
        abs_d = np.abs(np.asarray(deltas, dtype=np.float64))
        summary = {
            "camera_bag": str(bag_path.resolve()),
            "camera_index": str(index_path.resolve()),
            "color_topic": args.color_topic,
            "depth_topic": args.depth_topic,
            "color_frames": int(color_ts.size),
            "depth_frames": int(depth_ts.size),
            "sync_pairs": len(updated),
            "camera_matched_pairs": matched,
            "max_delta_ms_threshold": float(args.max_delta_ms),
            "median_abs_camera_delta_ms": float(np.median(abs_d)),
            "p95_abs_camera_delta_ms": float(np.percentile(abs_d, 95)),
            "color_t_start": float(color_ts[0]),
            "color_t_end": float(color_ts[-1]),
        }
        print(
            f"Camera match error: median={summary['median_abs_camera_delta_ms']:.2f} ms, "
            f"p95={summary['p95_abs_camera_delta_ms']:.2f} ms"
        )
        if args.output_json:
            json_path = Path(args.output_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Summary -> {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
