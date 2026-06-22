"""Camera timestamp loading and nearest-neighbor sync helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    index: int
    timestamp: float


@dataclass(frozen=True)
class CameraPair:
    pair_idx: int
    radar_idx: int
    lidar_idx: int
    radar_t: float
    lidar_t: float
    camera_idx: int
    camera_t: float
    camera_t_mapped: float
    delta_ms: float


def _timestamp_key(row: dict) -> Optional[str]:
    for key in ("camera_t", "timestamp", "time", "t", "unix_time", "unix_t"):
        if key in row and row[key] not in ("", None):
            return key
    return None


def _index_key(row: dict) -> Optional[str]:
    for key in ("camera_idx", "frame_idx", "frame", "index", "idx", "id"):
        if key in row and row[key] not in ("", None):
            return key
    return None


def camera_frames_from_timestamps(values: Sequence[float]) -> List[CameraFrame]:
    return [CameraFrame(i, float(t)) for i, t in enumerate(values)]


def read_camera_timestamps(path: str | Path) -> List[CameraFrame]:
    """Read camera timestamps from CSV, JSON, TXT, NPY, or NPZ.

    CSV rows may contain ``timestamp``/``camera_t`` and optional ``frame_idx``.
    JSON may be a list of timestamps, a list of frame dicts, or ``{"frames": [...]}``.
    NPZ uses the first of ``camera_t``, ``timestamps``, ``time``, or ``t``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return camera_frames_from_timestamps(np.load(p).astype(np.float64).reshape(-1))
    if suffix == ".npz":
        data = np.load(p)
        for key in ("camera_t", "timestamps", "timestamp", "time", "t"):
            if key in data:
                return camera_frames_from_timestamps(
                    np.asarray(data[key], dtype=np.float64).reshape(-1)
                )
        raise ValueError(f"No timestamp array found in {p}")
    if suffix in (".txt", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else None
        vals = np.loadtxt(p, delimiter=delimiter, dtype=np.float64)
        if vals.ndim == 2:
            vals = vals[:, -1]
        return camera_frames_from_timestamps(vals.reshape(-1))
    if suffix == ".csv":
        frames: List[CameraFrame] = []
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_i, row in enumerate(reader):
                tk = _timestamp_key(row)
                if tk is None:
                    raise ValueError(f"No timestamp column found in {p}")
                ik = _index_key(row)
                idx = int(float(row[ik])) if ik else row_i
                frames.append(CameraFrame(idx, float(row[tk])))
        return frames
    if suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("frames", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"JSON timestamps must be a list or have frames[]: {p}")
        if rows and isinstance(rows[0], dict):
            frames = []
            for row_i, row in enumerate(rows):
                tk = _timestamp_key(row)
                if tk is None:
                    raise ValueError(f"No timestamp field in frame row {row_i}: {p}")
                ik = _index_key(row)
                idx = int(float(row[ik])) if ik else row_i
                frames.append(CameraFrame(idx, float(row[tk])))
            return frames
        return camera_frames_from_timestamps([float(v) for v in rows])
    raise ValueError(f"Unsupported camera timestamp file: {p}")


def infer_camera_timestamps(
    *,
    start_time_s: float,
    fps: float,
    frame_count: int,
) -> List[CameraFrame]:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0")
    ts = float(start_time_s) + np.arange(int(frame_count), dtype=np.float64) / float(fps)
    return camera_frames_from_timestamps(ts)


def read_sync_pairs_csv(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for pair_idx, row in enumerate(csv.DictReader(f)):
            rows.append(
                {
                    "pair_idx": pair_idx,
                    "radar_idx": int(row["radar_idx"]),
                    "lidar_idx": int(row["lidar_idx"]),
                    "radar_t": float(row["radar_t"]),
                    "lidar_t": float(row["lidar_t"]),
                }
            )
    return rows


def map_camera_timestamps(
    frames: Sequence[CameraFrame],
    *,
    camera_time_scale: float = 1.0,
    camera_to_radar_offset_s: float = 0.0,
) -> np.ndarray:
    raw = np.asarray([f.timestamp for f in frames], dtype=np.float64)
    return raw * float(camera_time_scale) + float(camera_to_radar_offset_s)


def match_camera_to_sync_pairs(
    sync_rows: Sequence[dict],
    camera_frames: Sequence[CameraFrame],
    *,
    target_time: str = "radar_t",
    camera_time_scale: float = 1.0,
    camera_to_radar_offset_s: float = 0.0,
    max_delta_ms: float = 100.0,
) -> List[CameraPair]:
    """Attach each existing radar/lidar sync pair to the nearest camera frame."""
    if target_time not in ("radar_t", "lidar_t"):
        raise ValueError("target_time must be 'radar_t' or 'lidar_t'")
    if not sync_rows or not camera_frames:
        return []

    cam_mapped = map_camera_timestamps(
        camera_frames,
        camera_time_scale=camera_time_scale,
        camera_to_radar_offset_s=camera_to_radar_offset_s,
    )
    order = np.argsort(cam_mapped)
    sorted_t = cam_mapped[order]
    targets = np.asarray([r[target_time] for r in sync_rows], dtype=np.float64)

    idx = np.searchsorted(sorted_t, targets, side="left")
    idx = np.clip(idx, 1, sorted_t.size - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(targets - sorted_t[left]) > np.abs(sorted_t[right] - targets)
    nearest_sorted = np.where(choose_right, right, left)
    nearest = order[nearest_sorted]
    delta_ms = (targets - cam_mapped[nearest]) * 1e3

    out: List[CameraPair] = []
    for row, cam_i, cam_t_adj, dt in zip(sync_rows, nearest, cam_mapped[nearest], delta_ms):
        if abs(float(dt)) > max_delta_ms:
            continue
        frame = camera_frames[int(cam_i)]
        out.append(
            CameraPair(
                pair_idx=int(row["pair_idx"]),
                radar_idx=int(row["radar_idx"]),
                lidar_idx=int(row["lidar_idx"]),
                radar_t=float(row["radar_t"]),
                lidar_t=float(row["lidar_t"]),
                camera_idx=int(frame.index),
                camera_t=float(frame.timestamp),
                camera_t_mapped=float(cam_t_adj),
                delta_ms=float(dt),
            )
        )
    return out


def write_camera_sync_csv(pairs: Iterable[CameraPair], path: str | Path) -> None:
    fieldnames = [
        "pair_idx",
        "radar_idx",
        "lidar_idx",
        "radar_t",
        "lidar_t",
        "camera_idx",
        "camera_t",
        "camera_t_mapped",
        "delta_ms",
    ]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "pair_idx": pair.pair_idx,
                    "radar_idx": pair.radar_idx,
                    "lidar_idx": pair.lidar_idx,
                    "radar_t": f"{pair.radar_t:.9f}",
                    "lidar_t": f"{pair.lidar_t:.9f}",
                    "camera_idx": pair.camera_idx,
                    "camera_t": f"{pair.camera_t:.9f}",
                    "camera_t_mapped": f"{pair.camera_t_mapped:.9f}",
                    "delta_ms": f"{pair.delta_ms:.3f}",
                }
            )


def summarize_camera_pairs(pairs: Sequence[CameraPair]) -> dict:
    if not pairs:
        return {"matched_pairs": 0}
    d = np.asarray([p.delta_ms for p in pairs], dtype=np.float64)
    return {
        "matched_pairs": len(pairs),
        "median_abs_delta_ms": float(np.median(np.abs(d))),
        "p95_abs_delta_ms": float(np.percentile(np.abs(d), 95)),
        "max_abs_delta_ms": float(np.max(np.abs(d))),
        "camera_idx_start": int(pairs[0].camera_idx),
        "camera_idx_end": int(pairs[-1].camera_idx),
        "pair_idx_start": int(pairs[0].pair_idx),
        "pair_idx_end": int(pairs[-1].pair_idx),
    }

