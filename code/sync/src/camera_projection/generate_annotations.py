"""CLI: generate initial object proposal annotations directly from D435 depth.

Depth can come from two sources (mutually exclusive):
  --camera_bag  : read depth frames directly from a ROS bag file (preferred)
  --depth_dir   : read pre-extracted NPY/NPZ depth files from disk (legacy)

When --camera_bag is used the depth topic is scanned once in sequence; only
frames referenced in the camera_sync_csv are projected.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _THIS_DIR.parent
if str(_SYNC_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNC_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from annotations import empty_annotations, load_annotations, save_annotations  # noqa: E402
from cache import ProjectionCache  # noqa: E402
from geometry import depth_to_sensor_points, load_calibration, load_depth, robust_bounds  # noqa: E402

try:
    from lib.camera_io import (  # noqa: E402
        DEFAULT_DEPTH_TOPIC,
        iter_topic_messages,
        parse_ros1_image,
    )
    _CAMERA_COMPAT = True
except ImportError:
    DEFAULT_DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"
    _CAMERA_COMPAT = False


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate unlabeled top-down object proposals from synced D435 depth."
    )
    p.add_argument("--camera_sync_csv", required=True)
    p.add_argument("--calibration_json", required=True)
    p.add_argument("--output_annotations", required=True)
    p.add_argument("--existing_annotations", default=None)

    # --- depth source (one of the two is required) ---
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--camera_bag",
        default=None,
        help="ROS bag containing raw depth frames (preferred; reads depth_topic directly).",
    )
    src.add_argument(
        "--depth_dir",
        default=None,
        help="Directory of pre-extracted depth NPY/NPZ files (legacy).",
    )

    p.add_argument(
        "--depth_topic",
        default=DEFAULT_DEPTH_TOPIC,
        help="Depth image topic in the ROS bag (only used with --camera_bag).",
    )
    p.add_argument(
        "--depth_pattern",
        default="depth_{camera_idx:06d}.npy",
        help="Filename pattern relative to depth_dir. "
        "Fields: camera_idx, pair_idx, radar_idx, lidar_idx. "
        "(only used with --depth_dir)",
    )
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--refresh_cache", action="store_true")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--max_pairs", type=int, default=-1)
    p.add_argument("--stride", type=int, default=4, help="Pixel stride for full-frame projection.")
    p.add_argument("--min_depth_m", type=float, default=0.5)
    p.add_argument("--max_depth_m", type=float, default=30.0)
    p.add_argument("--scene_lateral", type=float, nargs=2, default=[-15.0, 15.0])
    p.add_argument("--scene_forward", type=float, nargs=2, default=[0.5, 30.0])
    p.add_argument("--grid_res_m", type=float, default=0.25)
    p.add_argument("--min_component_points", type=int, default=25)
    p.add_argument("--min_box_lateral_m", type=float, default=0.2)
    p.add_argument("--min_box_forward_m", type=float, default=0.2)
    p.add_argument("--max_box_lateral_m", type=float, default=6.0)
    p.add_argument("--max_box_forward_m", type=float, default=8.0)
    p.add_argument("--box_percentiles", type=float, nargs=2, default=[5.0, 95.0])
    p.add_argument("--label", default="object")
    p.add_argument("--color", default="yellow")
    p.add_argument("--replace_generated", action="store_true")
    return p


def read_camera_sync(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Prefer camera_depth_idx (from sync_camera_pairs.py); fall back to camera_idx.
            depth_idx_key = "camera_depth_idx" if "camera_depth_idx" in row else "camera_idx"
            depth_idx = int(row[depth_idx_key]) if row.get(depth_idx_key, "") != "" else -1
            if depth_idx < 0:
                continue
            rows.append(
                {
                    "pair_idx": int(row["pair_idx"]),
                    "camera_idx": int(row["camera_idx"]) if row.get("camera_idx", "") != "" else depth_idx,
                    "camera_depth_idx": depth_idx,
                    "radar_idx": int(row["radar_idx"]),
                    "lidar_idx": int(row["lidar_idx"]),
                }
            )
    return rows


def depth_path_for_row(args, row: dict) -> Optional[Path]:
    rel = args.depth_pattern.format(
        camera_idx=int(row["camera_idx"]),
        pair_idx=int(row["pair_idx"]),
        radar_idx=int(row["radar_idx"]),
        lidar_idx=int(row["lidar_idx"]),
    )
    p = Path(args.depth_dir) / rel
    return p if p.is_file() else None


def _cache_dir(args) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir)
    return Path(args.output_annotations).resolve().parent / "camera_projection_cache"


def filter_points(points: dict, args) -> dict:
    forward = np.asarray(points["forward"], dtype=np.float64)
    lateral = np.asarray(points["lateral"], dtype=np.float64)
    depth_m = np.asarray(points["depth_m"], dtype=np.float64)
    valid = (
        np.isfinite(forward)
        & np.isfinite(lateral)
        & np.isfinite(depth_m)
        & (depth_m >= args.min_depth_m)
        & (depth_m <= args.max_depth_m)
        & (forward >= args.scene_forward[0])
        & (forward <= args.scene_forward[1])
        & (lateral >= args.scene_lateral[0])
        & (lateral <= args.scene_lateral[1])
    )
    return {
        "forward": forward[valid],
        "lateral": lateral[valid],
        "depth_m": depth_m[valid],
    }


def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """Return component coordinates as arrays of [row, col]."""
    try:
        from scipy import ndimage

        labels, n_labels = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        return [np.argwhere(labels == i) for i in range(1, int(n_labels) + 1)]
    except Exception:
        visited = np.zeros(mask.shape, dtype=bool)
        comps: List[np.ndarray] = []
        rows, cols = np.nonzero(mask)
        for start_r, start_c in zip(rows, cols):
            if visited[start_r, start_c]:
                continue
            stack = [(int(start_r), int(start_c))]
            visited[start_r, start_c] = True
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if (
                            0 <= rr < mask.shape[0]
                            and 0 <= cc < mask.shape[1]
                            and mask[rr, cc]
                            and not visited[rr, cc]
                        ):
                            visited[rr, cc] = True
                            stack.append((rr, cc))
            comps.append(np.asarray(comp, dtype=np.int64))
        return comps


def proposals_from_points(points: dict, args) -> List[dict]:
    points = filter_points(points, args)
    if points["forward"].size < args.min_component_points:
        return []

    lat0, lat1 = [float(v) for v in args.scene_lateral]
    fwd0, fwd1 = [float(v) for v in args.scene_forward]
    res = float(args.grid_res_m)
    n_cols = max(1, int(np.ceil((lat1 - lat0) / res)))
    n_rows = max(1, int(np.ceil((fwd1 - fwd0) / res)))
    cols = np.floor((points["lateral"] - lat0) / res).astype(np.int64)
    rows = np.floor((points["forward"] - fwd0) / res).astype(np.int64)
    inside = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
    rows = rows[inside]
    cols = cols[inside]
    if rows.size < args.min_component_points:
        return []

    occ = np.zeros((n_rows, n_cols), dtype=bool)
    occ[rows, cols] = True
    comps = connected_components(occ)
    out: List[dict] = []
    for comp in comps:
        if comp.shape[0] < args.min_component_points:
            continue
        rmin, cmin = comp.min(axis=0)
        rmax, cmax = comp.max(axis=0)
        in_comp = (
            (rows >= rmin)
            & (rows <= rmax)
            & (cols >= cmin)
            & (cols <= cmax)
        )
        if int(np.count_nonzero(in_comp)) < args.min_component_points:
            continue
        lateral = points["lateral"][inside][in_comp]
        forward = points["forward"][inside][in_comp]
        depth_m = points["depth_m"][inside][in_comp]
        box_lat = robust_bounds(lateral, tuple(args.box_percentiles))
        box_fwd = robust_bounds(forward, tuple(args.box_percentiles))
        width = box_lat[1] - box_lat[0]
        length = box_fwd[1] - box_fwd[0]
        if width < args.min_box_lateral_m or length < args.min_box_forward_m:
            continue
        if width > args.max_box_lateral_m or length > args.max_box_forward_m:
            continue
        out.append(
            {
                "lateral": [float(box_lat[0]), float(box_lat[1])],
                "forward": [float(box_fwd[0]), float(box_fwd[1])],
                "n_depth_points": int(depth_m.size),
                "median_depth_m": float(np.median(depth_m)),
            }
        )
    return out


def _add_proposals(
    objects: dict,
    pair_idx: int,
    camera_idx: int,
    points: dict,
    args,
    replace_generated: bool,
) -> int:
    """Project points into proposals and append to objects[pair_idx]. Returns count added."""
    boxes = list(objects.get(str(pair_idx), []))
    if replace_generated and str(pair_idx) in objects:
        boxes = [b for b in boxes if str(b.get("source")) != "camera_depth_cluster"]
    proposals = proposals_from_points(points, args)
    for prop_i, prop in enumerate(proposals):
        boxes.append(
            {
                "id": f"depth_cluster_{camera_idx:06d}_{prop_i:03d}",
                "label": args.label,
                "lateral": prop["lateral"],
                "forward": prop["forward"],
                "color": args.color,
                "source": "camera_depth_cluster",
                "camera_idx": camera_idx,
                "n_depth_points": prop["n_depth_points"],
                "median_depth_m": prop["median_depth_m"],
            }
        )
    if boxes:
        objects[str(pair_idx)] = boxes
    return len(proposals)


def _process_from_dir(args, rows, calib, annotations, projection_cache) -> tuple[int, int]:
    """Legacy path: read pre-extracted depth files from disk."""
    objects = annotations.setdefault("objects", {})
    generated = 0
    skipped = 0
    depth_cache: Dict[Path, object] = {}
    for row in rows:
        pair_idx = int(row["pair_idx"])
        camera_idx = int(row["camera_idx"])
        depth_path = depth_path_for_row(args, row)
        if depth_path is None:
            skipped += 1
            continue

        points = None
        cache_key = None
        if projection_cache is not None:
            cache_key = projection_cache.key_for_depth_frame(
                depth_path=depth_path,
                camera_idx=camera_idx,
                stride=args.stride,
            )
            points = projection_cache.load_points(cache_key)
        if points is None:
            if projection_cache is not None:
                projection_cache.record_miss()
            if depth_path not in depth_cache:
                depth_cache[depth_path] = load_depth(depth_path)
            points = depth_to_sensor_points(depth_cache[depth_path], calib, stride=args.stride)
            if projection_cache is not None and cache_key is not None:
                projection_cache.save_points(cache_key, **points)

        generated += _add_proposals(
            objects, pair_idx, camera_idx, points, args,
            replace_generated=args.replace_generated,
        )
    return generated, skipped


def _process_from_bag(args, rows, calib, annotations, projection_cache) -> tuple[int, int]:
    """Primary path: read depth frames directly from the ROS bag."""
    if not _CAMERA_COMPAT:
        raise ImportError(
            "camera_compat not found. Install rosbags: pip install rosbags"
        )
    objects = annotations.setdefault("objects", {})
    bag_path = Path(args.camera_bag)

    # Build lookup: depth_frame_idx -> list of (pair_idx, camera_idx)
    needed: Dict[int, List[tuple[int, int]]] = {}
    for row in rows:
        depth_idx = int(row["camera_depth_idx"])
        if depth_idx < 0:
            continue
        needed.setdefault(depth_idx, []).append(
            (int(row["pair_idx"]), int(row["camera_idx"]))
        )

    if not needed:
        return 0, len(rows)

    # Check cache for all needed frames first.
    uncached: set[int] = set()
    cached_points: Dict[int, dict] = {}
    if projection_cache is not None:
        for depth_idx in needed:
            key = projection_cache.key_for_bag_depth_frame(
                bag_path=bag_path,
                depth_topic=args.depth_topic,
                frame_idx=depth_idx,
                stride=args.stride,
            )
            pts = projection_cache.load_points(key)
            if pts is not None:
                cached_points[depth_idx] = pts
            else:
                uncached.add(depth_idx)
    else:
        uncached = set(needed.keys())

    # Scan bag once for all uncached depth frames.
    if uncached:
        max_needed = max(uncached)
        print(
            f"Scanning bag depth topic ({args.depth_topic}) for "
            f"{len(uncached)} uncached frames (up to frame {max_needed})..."
        )
        bag_points: Dict[int, dict] = {}
        for i, (_bag_ts, raw) in enumerate(iter_topic_messages(bag_path, args.depth_topic)):
            if i > max_needed:
                break
            if i not in uncached:
                continue
            img = parse_ros1_image(raw)
            pts = depth_to_sensor_points(img.data, calib, stride=args.stride)
            bag_points[i] = pts
            if projection_cache is not None:
                key = projection_cache.key_for_bag_depth_frame(
                    bag_path=bag_path,
                    depth_topic=args.depth_topic,
                    frame_idx=i,
                    stride=args.stride,
                )
                projection_cache.save_points(key, **pts)
                projection_cache.record_miss()
        cached_points.update(bag_points)

    # Generate proposals from projected points.
    generated = 0
    skipped = 0
    for depth_idx, pairs in needed.items():
        pts = cached_points.get(depth_idx)
        if pts is None:
            skipped += len(pairs)
            continue
        for pair_idx, camera_idx in pairs:
            generated += _add_proposals(
                objects, pair_idx, camera_idx, pts, args,
                replace_generated=args.replace_generated,
            )
    return generated, skipped


def main() -> None:
    args = build_argparser().parse_args()

    if args.camera_bag is None and args.depth_dir is None:
        raise SystemExit(
            "Provide either --camera_bag (read from ROS bag) "
            "or --depth_dir (pre-extracted depth files)."
        )

    calib = load_calibration(args.calibration_json)
    rows = read_camera_sync(args.camera_sync_csv)
    rows = rows if args.max_pairs < 0 else rows[: args.max_pairs]

    annotations = (
        load_annotations(args.existing_annotations)
        if args.existing_annotations
        else empty_annotations()
    )

    projection_cache = None
    if not args.no_cache:
        projection_cache = ProjectionCache(
            _cache_dir(args),
            calibration_json=args.calibration_json,
            refresh=args.refresh_cache,
        )

    if args.camera_bag:
        generated, skipped = _process_from_bag(args, rows, calib, annotations, projection_cache)
        source_desc = f"bag {Path(args.camera_bag).name} / {args.depth_topic}"
    else:
        generated, skipped = _process_from_dir(args, rows, calib, annotations, projection_cache)
        source_desc = f"depth_dir {args.depth_dir}"

    save_annotations(annotations, args.output_annotations)

    if projection_cache is not None:
        projection_cache.write_manifest(
            {
                "mode": "depth_cluster_generation",
                "camera_sync_csv": str(Path(args.camera_sync_csv)),
                "source": source_desc,
                "stride": int(args.stride),
                "output_annotations": str(Path(args.output_annotations)),
                "generated": int(generated),
                "skipped": int(skipped),
            }
        )
    print(
        f"Generated {generated} depth-cluster proposals -> {args.output_annotations} "
        f"(skipped {skipped} frames with no depth)."
    )
    if projection_cache is not None:
        print(
            "Projection cache: "
            f"{projection_cache.hits} hits, {projection_cache.misses} misses, "
            f"{projection_cache.writes} writes -> {projection_cache.cache_dir}"
        )


if __name__ == "__main__":
    main()
