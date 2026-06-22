"""CLI: project camera detections/depth into sync_annotations top-down JSON."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _THIS_DIR.parent
if str(_SYNC_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNC_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from cache import ProjectionCache  # noqa: E402
from geometry import (  # noqa: E402
    bbox_depth_to_sensor_points,
    load_calibration,
    load_depth,
    sensor_points_to_topdown_box,
    topdown_box_from_center_depth,
)
from annotations import empty_annotations, load_annotations, save_annotations  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Project D435 detections into radar/lidar top-down annotation boxes."
    )
    p.add_argument("--camera_sync_csv", required=True)
    p.add_argument("--detections_json", required=True)
    p.add_argument("--calibration_json", required=True)
    p.add_argument("--output_annotations", required=True)
    p.add_argument("--existing_annotations", default=None)
    p.add_argument("--depth_dir", default=None)
    p.add_argument(
        "--depth_pattern",
        default="depth_{camera_idx:06d}.npy",
        help="Pattern relative to depth_dir. Fields: camera_idx, pair_idx.",
    )
    p.add_argument("--bbox_format", choices=["xywh", "xyxy"], default="xywh")
    p.add_argument("--min_depth_m", type=float, default=0.2)
    p.add_argument("--max_depth_m", type=float, default=80.0)
    p.add_argument("--min_points", type=int, default=20)
    p.add_argument(
        "--box_percentiles",
        type=float,
        nargs=2,
        default=[5.0, 95.0],
        metavar=("LOW", "HIGH"),
        help="Robust percentiles for lateral/forward box bounds.",
    )
    p.add_argument("--default_width_m", type=float, default=1.0)
    p.add_argument("--default_length_m", type=float, default=1.0)
    p.add_argument("--score_threshold", type=float, default=0.0)
    p.add_argument("--max_pairs", type=int, default=-1)
    p.add_argument(
        "--cache_dir",
        default=None,
        help="Projection cache directory. Default: <output_annotations parent>/camera_projection_cache.",
    )
    p.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable disk cache and only use the per-run memory cache.",
    )
    p.add_argument(
        "--refresh_cache",
        action="store_true",
        help="Ignore existing cached projected points and rewrite them.",
    )
    return p


def read_camera_sync(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "pair_idx": int(row["pair_idx"]),
                    "camera_idx": int(row["camera_idx"]),
                    "radar_idx": int(row["radar_idx"]),
                    "lidar_idx": int(row["lidar_idx"]),
                }
            )
    return rows


def _det_frame_key(det: dict) -> Optional[int]:
    for key in ("camera_idx", "frame_idx", "image_id", "frame", "index"):
        if key in det:
            return int(det[key])
    return None


def _bbox(det: dict):
    if "bbox" in det:
        return det["bbox"]
    if all(k in det for k in ("x", "y", "w", "h")):
        return [det["x"], det["y"], det["w"], det["h"]]
    if all(k in det for k in ("x0", "y0", "x1", "y1")):
        return [det["x0"], det["y0"], det["x1"], det["y1"]]
    raise ValueError(f"Detection has no bbox: {det!r}")


def _label(det: dict) -> str:
    return str(det.get("label") or det.get("category_name") or det.get("class") or "object")


def _score(det: dict) -> Optional[float]:
    for key in ("score", "confidence", "conf"):
        if key in det and det[key] is not None:
            return float(det[key])
    return None


def load_detections_by_frame(path: str | Path) -> Dict[int, List[dict]]:
    """Load simple or COCO-like detections keyed by camera frame index."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    categories = {}
    if isinstance(payload, dict):
        for cat in payload.get("categories", []):
            if "id" in cat:
                categories[int(cat["id"])] = str(cat.get("name", cat["id"]))

    if isinstance(payload, list):
        rows = payload
    elif "annotations" in payload:
        rows = payload["annotations"]
    elif "detections" in payload:
        rows = payload["detections"]
    elif "frames" in payload:
        rows = []
        for frame in payload["frames"]:
            frame_idx = _det_frame_key(frame)
            for det in frame.get("detections", frame.get("annotations", [])):
                item = dict(det)
                if frame_idx is not None:
                    item.setdefault("camera_idx", frame_idx)
                rows.append(item)
    else:
        raise ValueError(f"Unsupported detections JSON shape: {path}")

    by_frame: Dict[int, List[dict]] = {}
    for det in rows:
        frame_idx = _det_frame_key(det)
        if frame_idx is None:
            continue
        det = dict(det)
        if "category_id" in det and not any(
            k in det for k in ("label", "category_name", "class")
        ):
            det["label"] = categories.get(int(det["category_id"]), str(det["category_id"]))
        by_frame.setdefault(int(frame_idx), []).append(det)
    return by_frame


def _depth_path(args, row: dict) -> Optional[Path]:
    if not args.depth_dir:
        return None
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


def main() -> None:
    args = build_argparser().parse_args()
    calib = load_calibration(args.calibration_json)
    sync_rows = read_camera_sync(args.camera_sync_csv)
    detections = load_detections_by_frame(args.detections_json)
    annotations = (
        load_annotations(args.existing_annotations)
        if args.existing_annotations
        else empty_annotations()
    )
    objects = annotations.setdefault("objects", {})

    projected = 0
    skipped = 0
    depth_cache: Dict[Path, object] = {}
    projection_cache = None
    if not args.no_cache:
        projection_cache = ProjectionCache(
            _cache_dir(args),
            calibration_json=args.calibration_json,
            refresh=args.refresh_cache,
        )
    rows = sync_rows if args.max_pairs < 0 else sync_rows[: args.max_pairs]
    for row in rows:
        pair_idx = int(row["pair_idx"])
        camera_idx = int(row["camera_idx"])
        frame_dets = detections.get(camera_idx, [])
        if not frame_dets:
            continue

        depth_path = _depth_path(args, row)
        depth = None

        boxes = list(objects.get(str(pair_idx), []))
        for det_i, det in enumerate(frame_dets):
            score = _score(det)
            if score is not None and score < args.score_threshold:
                continue
            bbox = _bbox(det)
            box = None
            if depth_path is not None:
                det_id = str(det.get("id") or f"cam{camera_idx:06d}_{det_i:03d}")
                points = None
                if projection_cache is not None and depth_path is not None:
                    cache_key = projection_cache.key_for_detection(
                        depth_path=depth_path,
                        camera_idx=camera_idx,
                        pair_idx=pair_idx,
                        detection_id=det_id,
                        bbox=list(bbox),
                        bbox_format=args.bbox_format,
                    )
                    points = projection_cache.load_points(cache_key)
                else:
                    cache_key = None
                if points is None:
                    if projection_cache is not None:
                        projection_cache.record_miss()
                    if depth is None:
                        if depth_path not in depth_cache:
                            depth_cache[depth_path] = load_depth(depth_path)
                        depth = depth_cache[depth_path]
                    points = bbox_depth_to_sensor_points(
                        depth,
                        bbox,
                        calib,
                        bbox_format=args.bbox_format,
                    )
                    if projection_cache is not None and cache_key is not None:
                        projection_cache.save_points(cache_key, **points)
                box = sensor_points_to_topdown_box(
                    points,
                    min_depth_m=args.min_depth_m,
                    max_depth_m=args.max_depth_m,
                    min_points=args.min_points,
                    percentiles=tuple(args.box_percentiles),
                )
            elif "depth_m" in det:
                box = topdown_box_from_center_depth(
                    bbox,
                    float(det["depth_m"]),
                    calib,
                    bbox_format=args.bbox_format,
                    width_m=args.default_width_m,
                    length_m=args.default_length_m,
                )
            if box is None:
                skipped += 1
                continue
            label = _label(det)
            ann = {
                "id": str(det.get("id") or f"cam{camera_idx:06d}_{det_i:03d}"),
                "label": label,
                "lateral": box["lateral"],
                "forward": box["forward"],
                "color": str(det.get("color", "lime")),
            }
            if score is not None:
                ann["score"] = score
            ann["source"] = "camera_projection"
            ann["camera_idx"] = camera_idx
            ann["n_depth_points"] = box["n_depth_points"]
            ann["median_depth_m"] = box["median_depth_m"]
            boxes.append(ann)
            projected += 1
        if boxes:
            objects[str(pair_idx)] = boxes

    save_annotations(annotations, args.output_annotations)
    if projection_cache is not None:
        projection_cache.write_manifest(
            {
                "camera_sync_csv": str(Path(args.camera_sync_csv)),
                "detections_json": str(Path(args.detections_json)),
                "depth_dir": str(Path(args.depth_dir)) if args.depth_dir else None,
                "depth_pattern": args.depth_pattern,
                "bbox_format": args.bbox_format,
                "output_annotations": str(Path(args.output_annotations)),
                "projected": int(projected),
                "skipped": int(skipped),
            }
        )
    print(
        f"Projected {projected} detections into {args.output_annotations} "
        f"(skipped {skipped})."
    )
    if projection_cache is not None:
        print(
            "Projection cache: "
            f"{projection_cache.hits} hits, {projection_cache.misses} misses, "
            f"{projection_cache.writes} writes -> {projection_cache.cache_dir}"
        )


if __name__ == "__main__":
    main()
