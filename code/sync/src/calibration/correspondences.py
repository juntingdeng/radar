"""Load/save hand-picked point correspondences for extrinsic calibration.

A correspondence file decouples *picking* (interactive, needs a GUI) from
*solving* (headless, testable). Each entry pairs one point in the source
sensor frame with the same physical point in the target sensor frame.

Schema::

    {
      "kind": "camera_lidar" | "radar_lidar",
      "source_frame": "camera_depth_optical",
      "target_frame": "lidar_sensor",
      "dim": 3,                       # 3 for camera_lidar, 2 for radar_lidar
      "units": "m",
      "dataset": "2026.05.10/18-05-08",
      "points": [
        {"pair_idx": 812, "source": [x, y, z], "target": [x, y, z],
         "note": "corner of retaining wall"},
        ...
      ]
    }

For radar_lidar, ``source``/``target`` are 2D ``[forward_x, lateral_y]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Correspondence:
    source: Tuple[float, ...]
    target: Tuple[float, ...]
    pair_idx: Optional[int] = None
    note: str = ""

    def to_dict(self) -> dict:
        d: dict = {"source": list(self.source), "target": list(self.target)}
        if self.pair_idx is not None:
            d["pair_idx"] = int(self.pair_idx)
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class CorrespondenceSet:
    kind: str  # "camera_lidar" | "radar_lidar"
    dim: int  # 3 or 2
    source_frame: str
    target_frame: str
    points: List[Correspondence] = field(default_factory=list)
    dataset: str = ""
    units: str = "m"

    def add(self, source, target, pair_idx=None, note="") -> None:
        source = tuple(float(v) for v in source)
        target = tuple(float(v) for v in target)
        if len(source) != self.dim or len(target) != self.dim:
            raise ValueError(
                f"Expected {self.dim}D points, got source={len(source)}, target={len(target)}"
            )
        self.points.append(Correspondence(source, target, pair_idx, note))

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.points:
            return (np.empty((0, self.dim)), np.empty((0, self.dim)))
        src = np.asarray([p.source for p in self.points], dtype=np.float64)
        tgt = np.asarray([p.target for p in self.points], dtype=np.float64)
        return src, tgt

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "dim": self.dim,
            "units": self.units,
            "dataset": self.dataset,
            "points": [p.to_dict() for p in self.points],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CorrespondenceSet":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cs = cls(
            kind=data.get("kind", ""),
            dim=int(data["dim"]),
            source_frame=data.get("source_frame", ""),
            target_frame=data.get("target_frame", ""),
            dataset=data.get("dataset", ""),
            units=data.get("units", "m"),
        )
        for p in data.get("points", []):
            cs.add(p["source"], p["target"], p.get("pair_idx"), p.get("note", ""))
        return cs

    def __len__(self) -> int:
        return len(self.points)
