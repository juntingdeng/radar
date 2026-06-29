"""Synchronized object annotations for radar/lidar top-down views.

Annotations are keyed by sync **pair** index (row in sync_pairs.csv). Boxes use the
shared sensor top-down frame (lateral Y, forward X) in meters — same as aligned video.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PathLike = Union[str, Path]

ANNOTATION_SCHEMA = {
    "coordinate_frame": "sensor_topdown",
    "units": "m",
    "x_axis": "lateral_y",
    "y_axis": "forward_x",
}


def _normalize_box(raw: dict) -> dict:
    lateral = raw.get("lateral")
    forward = raw.get("forward")
    if lateral is None or forward is None:
        raise ValueError(f"Annotation box must have lateral and forward: {raw!r}")
    lat = (float(lateral[0]), float(lateral[1]))
    fwd = (float(forward[0]), float(forward[1]))
    if lat[1] <= lat[0] or fwd[1] <= fwd[0]:
        raise ValueError(f"Box bounds must be increasing: {raw!r}")
    out = {
        "id": str(raw.get("id", "")),
        "label": str(raw.get("label", "")),
        "lateral": list(lat),
        "forward": list(fwd),
    }
    if "color" in raw:
        out["color"] = str(raw["color"])
    return out


def empty_annotations() -> dict:
    return {**ANNOTATION_SCHEMA, "objects": {}}


def load_annotations(path: Optional[PathLike]) -> dict:
    if path is None:
        return empty_annotations()
    p = Path(path)
    if not p.is_file():
        return empty_annotations()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    objects = data.get("objects", {})
    norm: Dict[str, List[dict]] = {}
    for pair_key, boxes in objects.items():
        norm[str(pair_key)] = [_normalize_box(b) for b in boxes]
    return {**ANNOTATION_SCHEMA, "objects": norm}


def save_annotations(data: dict, path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = empty_annotations()
    payload["objects"] = {
        str(k): [_normalize_box(b) for b in v] for k, v in data.get("objects", {}).items()
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_sync_pair_maps(
    sync_csv: PathLike,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Return lidar_idx->pair, radar_idx->pair, pair->lidar_idx (first match per key)."""
    import csv

    lidar_to_pair: Dict[int, int] = {}
    radar_to_pair: Dict[int, int] = {}
    pair_to_lidar: Dict[int, int] = {}
    with open(sync_csv, newline="", encoding="utf-8") as f:
        for pair_idx, row in enumerate(csv.DictReader(f)):
            lidx = int(row["lidar_idx"])
            ridx = int(row["radar_idx"])
            lidar_to_pair.setdefault(lidx, pair_idx)
            radar_to_pair.setdefault(ridx, pair_idx)
            pair_to_lidar[pair_idx] = lidx
    return lidar_to_pair, radar_to_pair, pair_to_lidar


def boxes_for_pair(data: dict, pair_idx) -> List[dict]:
    if pair_idx is None:
        return []
    return list(data.get("objects", {}).get(str(int(pair_idx)), []))


def pair_idx_from_lidar(lidar_to_pair: Dict[int, int], lidar_idx: int) -> Optional[int]:
    return lidar_to_pair.get(int(lidar_idx))


def pair_idx_from_radar(
    radar_to_pair: Dict[int, int],
    *,
    pair_idx: Optional[int] = None,
    radar_idx: Optional[int] = None,
) -> Optional[int]:
    if pair_idx is not None:
        return int(pair_idx)
    if radar_idx is not None:
        return radar_to_pair.get(int(radar_idx))
    return None


def draw_topdown_boxes(
    ax: matplotlib.axes.Axes,
    boxes: List[dict],
    *,
    default_color: str = "lime",
    linewidth: float = 2.0,
) -> None:
    """Draw synced boxes on a top-down axis (lateral X, forward Y)."""
    for box in boxes:
        lat0, lat1 = box["lateral"]
        fwd0, fwd1 = box["forward"]
        color = str(box.get("color", default_color))
        rect = Rectangle(
            (lat0, fwd0),
            lat1 - lat0,
            fwd1 - fwd0,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
        )
        ax.add_patch(rect)
        label = str(box.get("label") or box.get("id") or "").strip()
        if label:
            ax.text(
                lat0,
                fwd1,
                label,
                color=color,
                fontsize=8,
                va="bottom",
                ha="left",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
            )


def upsert_box(
    data: dict,
    pair_idx: int,
    box: dict,
    *,
    replace_id: Optional[str] = None,
) -> dict:
    """Add or replace a box on one pair; returns updated data dict."""
    key = str(int(pair_idx))
    norm = _normalize_box(box)
    boxes = list(data.setdefault("objects", {}).get(key, []))
    if replace_id:
        boxes = [b for b in boxes if str(b.get("id")) != str(replace_id)]
    boxes.append(norm)
    data.setdefault("objects", {})[key] = boxes
    return data


def box_from_selector(extents: Tuple[float, float, float, float]) -> dict:
    """Convert matplotlib RectangleSelector extents to annotation box."""
    x0, x1, y0, y1 = extents
    return {
        "lateral": [float(min(x0, x1)), float(max(x0, x1))],
        "forward": [float(min(y0, y1)), float(max(y0, y1))],
    }


def pick_box_two_clicks(
    ax: matplotlib.axes.Axes,
    *,
    label: str = "object",
    box_id: str = "",
    color: str = "yellow",
) -> Optional[dict]:
    """Click two corners on a top-down axis (lateral, forward) in meters."""
    print("Click bottom-left corner, then top-right corner. Middle-click or Ctrl+C to cancel.")
    pts = plt.ginput(2, timeout=0)
    if len(pts) < 2:
        print("Cancelled.")
        return None
    (x0, y0), (x1, y1) = pts
    box = box_from_selector((x0, x1, y0, y1))
    box["label"] = label
    box["id"] = box_id or label
    box["color"] = color
    draw_topdown_boxes(ax, [box])
    ax.figure.canvas.draw_idle()
    return box
