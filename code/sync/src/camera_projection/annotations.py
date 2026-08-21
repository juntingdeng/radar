"""Annotation JSON helpers compatible with sync_annotations.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


ANNOTATION_SCHEMA = {
    "coordinate_frame": "sensor_topdown",
    "units": "m",
    "x_axis": "lateral_y",
    "y_axis": "forward_x",
}


def empty_annotations() -> dict:
    return {**ANNOTATION_SCHEMA, "objects": {}}


def normalize_box(raw: dict) -> dict:
    lateral = raw.get("lateral")
    forward = raw.get("forward")
    if lateral is None or forward is None:
        raise ValueError(f"Annotation box must have lateral and forward: {raw!r}")
    lat = [float(lateral[0]), float(lateral[1])]
    fwd = [float(forward[0]), float(forward[1])]
    if lat[1] <= lat[0] or fwd[1] <= fwd[0]:
        raise ValueError(f"Box bounds must be increasing: {raw!r}")
    out = dict(raw)
    out["id"] = str(out.get("id", ""))
    out["label"] = str(out.get("label", ""))
    out["lateral"] = lat
    out["forward"] = fwd
    if "color" in out:
        out["color"] = str(out["color"])
    return out


def load_annotations(path: Optional[str | Path]) -> dict:
    if path is None:
        return empty_annotations()
    p = Path(path)
    if not p.is_file():
        return empty_annotations()
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out = empty_annotations()
    for pair_key, boxes in payload.get("objects", {}).items():
        out["objects"][str(pair_key)] = [normalize_box(b) for b in boxes]
    return out


def save_annotations(data: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = empty_annotations()
    payload["objects"] = {
        str(k): [normalize_box(b) for b in v]
        for k, v in data.get("objects", {}).items()
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

