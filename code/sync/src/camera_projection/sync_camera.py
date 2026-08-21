"""CLI: synchronize camera frames to existing radar/lidar sync pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _THIS_DIR.parent
if str(_SYNC_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNC_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from lib.sync_utils import fit_time_affine, fit_time_offset  # noqa: E402
from timestamps import (  # noqa: E402
    infer_camera_timestamps,
    match_camera_to_sync_pairs,
    read_camera_timestamps,
    read_sync_pairs_csv,
    summarize_camera_pairs,
    write_camera_sync_csv,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Attach Intel D435 camera frames to existing radar/lidar sync pairs."
    )
    p.add_argument("--sync_csv", required=True, help="Existing radar/lidar sync_pairs.csv.")
    p.add_argument("--camera_timestamps", default=None, help="CSV/JSON/TXT/NPY/NPZ timestamps.")
    p.add_argument("--camera_start_time_s", type=float, default=None)
    p.add_argument("--camera_fps", type=float, default=None)
    p.add_argument("--camera_frame_count", type=int, default=None)
    p.add_argument(
        "--target_time",
        choices=["radar_t", "lidar_t"],
        default="radar_t",
        help="Existing sync-pair timestamp used for nearest camera matching.",
    )
    p.add_argument("--fit_offset", action="store_true")
    p.add_argument("--fit_skew", action="store_true")
    p.add_argument("--offset_search_s", type=float, default=5.0)
    p.add_argument("--skew_search_ppm", type=float, default=200.0)
    p.add_argument("--camera_time_scale", type=float, default=1.0)
    p.add_argument("--camera_to_radar_offset_s", type=float, default=0.0)
    p.add_argument("--max_delta_ms", type=float, default=100.0)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_json", default=None)
    return p


def _load_camera_frames(args):
    if args.camera_timestamps:
        return read_camera_timestamps(args.camera_timestamps)
    needed = (args.camera_start_time_s, args.camera_fps, args.camera_frame_count)
    if any(v is None for v in needed):
        raise ValueError(
            "Pass --camera_timestamps, or pass --camera_start_time_s, "
            "--camera_fps, and --camera_frame_count."
        )
    return infer_camera_timestamps(
        start_time_s=args.camera_start_time_s,
        fps=args.camera_fps,
        frame_count=args.camera_frame_count,
    )


def main() -> None:
    args = build_argparser().parse_args()
    sync_rows = read_sync_pairs_csv(args.sync_csv)
    camera_frames = _load_camera_frames(args)
    if not sync_rows:
        raise RuntimeError("sync_csv has no rows.")
    if not camera_frames:
        raise RuntimeError("No camera frames loaded.")

    scale = float(args.camera_time_scale)
    offset = float(args.camera_to_radar_offset_s)
    fit_error_s = None
    target_ts = np.asarray([r[args.target_time] for r in sync_rows], dtype=np.float64)
    camera_ts = np.asarray([f.timestamp for f in camera_frames], dtype=np.float64)

    if args.fit_skew:
        scale, offset, fit_error_s = fit_time_affine(
            target_ts,
            camera_ts,
            max_abs_shift_s=args.offset_search_s,
            max_skew_ppm=args.skew_search_ppm,
        )
    elif args.fit_offset:
        offset = fit_time_offset(
            target_ts,
            camera_ts,
            max_abs_shift_s=args.offset_search_s,
        )

    pairs = match_camera_to_sync_pairs(
        sync_rows,
        camera_frames,
        target_time=args.target_time,
        camera_time_scale=scale,
        camera_to_radar_offset_s=offset,
        max_delta_ms=args.max_delta_ms,
    )
    write_camera_sync_csv(pairs, args.output_csv)
    summary = summarize_camera_pairs(pairs)
    summary.update(
        {
            "sync_csv": str(Path(args.sync_csv)),
            "camera_frames": len(camera_frames),
            "target_time": args.target_time,
            "estimated_camera_time_scale": scale,
            "estimated_camera_to_radar_offset_s": offset,
            "fit_median_error_s": fit_error_s,
            "max_delta_ms": args.max_delta_ms,
            "output_csv": str(Path(args.output_csv)),
        }
    )
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(
        f"Matched {summary['matched_pairs']} camera/sync pairs "
        f"(median |delta|={summary.get('median_abs_delta_ms', float('nan')):.2f} ms)."
    )
    print(f"Time map: sync ~= {scale:.9f} * camera + {offset:.6f} s")
    print(f"Wrote: {args.output_csv}")
    if args.output_json:
        print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()

