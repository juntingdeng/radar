"""D435 depth to sensor-topdown projection utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float
    camera_to_sensor: np.ndarray


def _matrix_from_payload(payload: dict) -> np.ndarray:
    for key in (
        "camera_to_sensor",
        "camera_to_lidar",
        "T_camera_to_sensor",
        "T_camera_to_lidar",
        "extrinsic",
        "transform",
    ):
        if key in payload:
            mat = np.asarray(payload[key], dtype=np.float64)
            if mat.shape != (4, 4):
                raise ValueError(f"{key} must be a 4x4 matrix")
            return mat
    return np.eye(4, dtype=np.float64)


def load_calibration(path: str | Path) -> CameraCalibration:
    """Load camera intrinsics and camera->sensor transform from JSON.

    Supported JSON shapes:
      {"intrinsics": {"fx":..., "fy":..., "cx":..., "cy":...}, "camera_to_lidar": [[...]]}
      {"K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "depth_scale": 0.001}
    The output sensor frame is assumed to be Ouster-style: X forward, Y lateral, Z up.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    intr = payload.get("intrinsics", payload)
    if "K" in payload:
        k = np.asarray(payload["K"], dtype=np.float64)
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    else:
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])
    depth_scale = float(payload.get("depth_scale", intr.get("depth_scale", 0.001)))
    return CameraCalibration(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        depth_scale=depth_scale,
        camera_to_sensor=_matrix_from_payload(payload),
    )


def bbox_to_xyxy(bbox: Sequence[float], fmt: str = "xywh") -> Tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values: {bbox!r}")
    x0, y0, a, b = [float(v) for v in bbox]
    if fmt == "xywh":
        x1, y1 = x0 + a, y0 + b
    elif fmt == "xyxy":
        x1, y1 = a, b
    else:
        raise ValueError("bbox format must be xywh or xyxy")
    ix0, ix1 = sorted((int(np.floor(x0)), int(np.ceil(x1))))
    iy0, iy1 = sorted((int(np.floor(y0)), int(np.ceil(y1))))
    return ix0, iy0, ix1, iy1


def crop_valid_depth(
    depth: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    *,
    depth_scale: float,
    min_depth_m: float = 0.2,
    max_depth_m: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    x0 = int(np.clip(x0, 0, w))
    x1 = int(np.clip(x1, 0, w))
    y0 = int(np.clip(y0, 0, h))
    y1 = int(np.clip(y1, 0, h))
    if x1 <= x0 or y1 <= y0:
        return np.empty(0), np.empty(0), np.empty(0)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64) * float(depth_scale)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)
    return xx[valid].astype(np.float64), yy[valid].astype(np.float64), patch[valid]


def crop_finite_depth(
    depth: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    *,
    depth_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all finite, positive depth samples in a bbox."""
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    x0 = int(np.clip(x0, 0, w))
    x1 = int(np.clip(x1, 0, w))
    y0 = int(np.clip(y0, 0, h))
    y1 = int(np.clip(y1, 0, h))
    if x1 <= x0 or y1 <= y0:
        return np.empty(0), np.empty(0), np.empty(0)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64) * float(depth_scale)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0)
    return xx[valid].astype(np.float64), yy[valid].astype(np.float64), patch[valid]


def pixels_depth_to_camera(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    calib: CameraCalibration,
) -> np.ndarray:
    x = (u - calib.cx) * z / calib.fx
    y = (v - calib.cy) * z / calib.fy
    return np.column_stack((x, y, z))


def transform_points(points_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz.reshape(0, 3)
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    hom = np.column_stack((points_xyz, ones))
    return (np.asarray(transform_4x4, dtype=np.float64) @ hom.T).T[:, :3]


def robust_bounds(values: np.ndarray, percentiles: Tuple[float, float]) -> Tuple[float, float]:
    lo, hi = np.percentile(values, percentiles)
    if float(hi) <= float(lo):
        eps = max(0.05, abs(float(lo)) * 0.01)
        return float(lo - eps), float(hi + eps)
    return float(lo), float(hi)


def bbox_depth_to_sensor_points(
    depth: np.ndarray,
    bbox: Sequence[float],
    calib: CameraCalibration,
    *,
    bbox_format: str = "xywh",
) -> dict:
    """Project all finite bbox depth pixels into the shared sensor frame."""
    xyxy = bbox_to_xyxy(bbox, fmt=bbox_format)
    u, v, z = crop_finite_depth(depth, xyxy, depth_scale=calib.depth_scale)
    if z.size == 0:
        return {
            "forward": np.empty(0, dtype=np.float64),
            "lateral": np.empty(0, dtype=np.float64),
            "depth_m": np.empty(0, dtype=np.float64),
        }
    cam_pts = pixels_depth_to_camera(u, v, z, calib)
    sensor_pts = transform_points(cam_pts, calib.camera_to_sensor)
    return {
        "forward": sensor_pts[:, 0].astype(np.float64),
        "lateral": sensor_pts[:, 1].astype(np.float64),
        "depth_m": z.astype(np.float64),
    }


def depth_to_sensor_points(
    depth: np.ndarray,
    calib: CameraCalibration,
    *,
    stride: int = 4,
) -> dict:
    """Project a whole depth frame into the shared sensor frame with pixel subsampling."""
    if stride <= 0:
        raise ValueError("stride must be > 0")
    depth_arr = np.asarray(depth)
    h, w = depth_arr.shape[:2]
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    sampled = np.asarray(depth_arr[0:h:stride, 0:w:stride], dtype=np.float64)
    z = sampled.reshape(-1) * float(calib.depth_scale)
    u = xx.reshape(-1).astype(np.float64)
    v = yy.reshape(-1).astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return {
            "forward": np.empty(0, dtype=np.float64),
            "lateral": np.empty(0, dtype=np.float64),
            "depth_m": np.empty(0, dtype=np.float64),
        }
    cam_pts = pixels_depth_to_camera(u[valid], v[valid], z[valid], calib)
    sensor_pts = transform_points(cam_pts, calib.camera_to_sensor)
    return {
        "forward": sensor_pts[:, 0].astype(np.float64),
        "lateral": sensor_pts[:, 1].astype(np.float64),
        "depth_m": z[valid].astype(np.float64),
    }


def sensor_points_to_topdown_box(
    points: dict,
    *,
    min_depth_m: float = 0.2,
    max_depth_m: float = 80.0,
    min_points: int = 20,
    percentiles: Tuple[float, float] = (5.0, 95.0),
) -> Optional[dict]:
    """Build a robust top-down box from cached/projected point arrays."""
    forward = np.asarray(points["forward"], dtype=np.float64)
    lateral = np.asarray(points["lateral"], dtype=np.float64)
    depth_m = np.asarray(points["depth_m"], dtype=np.float64)
    valid = (
        np.isfinite(forward)
        & np.isfinite(lateral)
        & np.isfinite(depth_m)
        & (depth_m >= float(min_depth_m))
        & (depth_m <= float(max_depth_m))
    )
    if int(np.count_nonzero(valid)) < int(min_points):
        return None
    forward = forward[valid]
    lateral = lateral[valid]
    depth_m = depth_m[valid]
    lat0, lat1 = robust_bounds(lateral, percentiles)
    fwd0, fwd1 = robust_bounds(forward, percentiles)
    if fwd1 <= 0:
        return None
    return {
        "lateral": [lat0, lat1],
        "forward": [max(0.0, fwd0), fwd1],
        "n_depth_points": int(depth_m.size),
        "median_depth_m": float(np.median(depth_m)),
    }


def bbox_depth_to_topdown_box(
    depth: np.ndarray,
    bbox: Sequence[float],
    calib: CameraCalibration,
    *,
    bbox_format: str = "xywh",
    min_depth_m: float = 0.2,
    max_depth_m: float = 80.0,
    min_points: int = 20,
    percentiles: Tuple[float, float] = (5.0, 95.0),
) -> Optional[dict]:
    points = bbox_depth_to_sensor_points(depth, bbox, calib, bbox_format=bbox_format)
    return sensor_points_to_topdown_box(
        points,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        min_points=min_points,
        percentiles=percentiles,
    )


def load_depth(path: str | Path) -> np.ndarray:
    """Load a depth map from NPY/NPZ or an image file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(p))
    if suffix == ".npz":
        data = np.load(p)
        for key in ("depth", "depth_m", "arr_0"):
            if key in data:
                return np.asarray(data[key])
        raise ValueError(f"No depth array found in {p}")
    try:
        import imageio.v3 as iio

        return np.asarray(iio.imread(p))
    except Exception:
        from PIL import Image

        return np.asarray(Image.open(p))


def topdown_box_from_center_depth(
    bbox: Sequence[float],
    depth_m: float,
    calib: CameraCalibration,
    *,
    bbox_format: str = "xywh",
    width_m: float = 1.0,
    length_m: float = 1.0,
) -> dict:
    """Fallback projection when a detector provides one depth value but no depth map."""
    x0, y0, x1, y1 = bbox_to_xyxy(bbox, fmt=bbox_format)
    u = np.asarray([(x0 + x1) * 0.5], dtype=np.float64)
    v = np.asarray([(y0 + y1) * 0.5], dtype=np.float64)
    z = np.asarray([float(depth_m)], dtype=np.float64)
    sensor = transform_points(pixels_depth_to_camera(u, v, z, calib), calib.camera_to_sensor)
    forward = float(sensor[0, 0])
    lateral = float(sensor[0, 1])
    return {
        "lateral": [lateral - width_m * 0.5, lateral + width_m * 0.5],
        "forward": [max(0.0, forward - length_m * 0.5), forward + length_m * 0.5],
        "n_depth_points": 1,
        "median_depth_m": float(depth_m),
    }
