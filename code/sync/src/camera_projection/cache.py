"""Disk cache for camera projection intermediates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


def file_fingerprint(path: str | Path) -> dict:
    p = Path(path)
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def json_file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ProjectionCache:
    """Cache projected points per detection bbox.

    Cached arrays are intentionally lower-level than final boxes:
    ``forward``, ``lateral``, and ``depth_m`` for all finite depth pixels in the
    detection crop. Final filtering/percentiles happen later, so post-processing can
    be rerun without touching raw depth files.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        calibration_json: str | Path,
        refresh: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.points_dir = self.cache_dir / "projected_points"
        self.points_dir.mkdir(parents=True, exist_ok=True)
        self.refresh = bool(refresh)
        self.calibration_sha256 = json_file_sha256(calibration_json)
        self.manifest_path = self.cache_dir / "manifest.json"
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def key_for_detection(
        self,
        *,
        depth_path: str | Path,
        camera_idx: int,
        pair_idx: int,
        detection_id: str,
        bbox: list,
        bbox_format: str,
    ) -> str:
        payload = {
            "version": 1,
            "kind": "camera_projection_points",
            "calibration_sha256": self.calibration_sha256,
            "depth": file_fingerprint(depth_path),
            "camera_idx": int(camera_idx),
            "pair_idx": int(pair_idx),
            "detection_id": str(detection_id),
            "bbox": [float(v) for v in bbox],
            "bbox_format": str(bbox_format),
        }
        return stable_hash(payload)

    def key_for_depth_frame(
        self,
        *,
        depth_path: str | Path,
        camera_idx: int,
        stride: int,
    ) -> str:
        payload = {
            "version": 1,
            "kind": "camera_projection_frame_points",
            "calibration_sha256": self.calibration_sha256,
            "depth": file_fingerprint(depth_path),
            "camera_idx": int(camera_idx),
            "stride": int(stride),
        }
        return stable_hash(payload)

    def key_for_bag_depth_frame(
        self,
        *,
        bag_path: str | Path,
        depth_topic: str,
        frame_idx: int,
        stride: int,
    ) -> str:
        payload = {
            "version": 1,
            "kind": "camera_projection_bag_frame_points",
            "calibration_sha256": self.calibration_sha256,
            "bag": file_fingerprint(bag_path),
            "depth_topic": str(depth_topic),
            "frame_idx": int(frame_idx),
            "stride": int(stride),
        }
        return stable_hash(payload)

    def path_for_key(self, key: str) -> Path:
        return self.points_dir / key[:2] / f"{key}.npz"

    def load_points(self, key: str) -> Optional[dict]:
        if self.refresh:
            return None
        path = self.path_for_key(key)
        if not path.is_file():
            return None
        data = np.load(path, allow_pickle=False)
        self.hits += 1
        return {
            "forward": np.asarray(data["forward"], dtype=np.float64),
            "lateral": np.asarray(data["lateral"], dtype=np.float64),
            "depth_m": np.asarray(data["depth_m"], dtype=np.float64),
        }

    def save_points(
        self,
        key: str,
        *,
        forward: np.ndarray,
        lateral: np.ndarray,
        depth_m: np.ndarray,
    ) -> None:
        path = self.path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            forward=np.asarray(forward, dtype=np.float32),
            lateral=np.asarray(lateral, dtype=np.float32),
            depth_m=np.asarray(depth_m, dtype=np.float32),
        )
        self.writes += 1

    def record_miss(self) -> None:
        self.misses += 1

    def write_manifest(self, extra: Optional[dict] = None) -> None:
        payload = {
            "version": 1,
            "kind": "camera_projection_cache",
            "calibration_sha256": self.calibration_sha256,
            "points_dir": str(self.points_dir),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "writes": int(self.writes),
        }
        if extra:
            payload.update(extra)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
