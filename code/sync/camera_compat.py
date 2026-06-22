"""Intel RealSense (R435) camera data from ROS bag recordings.

Camera captures live under ``{data_root}/camera/*.bag`` (ROS bag v2, recorded by
Intel RealSense SDK). Color and depth are ``sensor_msgs/Image`` on:

- ``/device_0/sensor_1/Color_0/image/data``  (rgb8)
- ``/device_0/sensor_0/Depth_0/image/data``  (16UC1, millimeters)

Sync uses each image **header.stamp** (Unix seconds), not the bag record timestamp.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

# Default RealSense ROS topics (R435 / librealsense ROS wrapper).
DEFAULT_COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
DEFAULT_DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"

INDEX_VERSION = 1


@dataclass(frozen=True)
class ParsedImage:
    header_t: float
    frame_id: str
    height: int
    width: int
    encoding: str
    step: int
    data: np.ndarray


def _read_ros1_string(raw: bytes, offset: int) -> Tuple[str, int]:
    (slen,) = struct.unpack_from("<I", raw, offset)
    offset += 4
    value = raw[offset : offset + slen].decode("utf-8", errors="replace")
    return value, offset + slen


def parse_ros1_image_header(raw: bytes) -> Tuple[float, int, int, str, int]:
    """Parse ``sensor_msgs/Image`` header fields without copying pixel data.

    Returns ``(header_t, height, width, encoding, data_offset)``.
    """
    offset = 0
    offset += 4  # seq
    sec, nsec = struct.unpack_from("<II", raw, offset)
    offset += 8
    _frame_id, offset = _read_ros1_string(raw, offset)
    height, width = struct.unpack_from("<II", raw, offset)
    offset += 8
    encoding, offset = _read_ros1_string(raw, offset)
    offset += 1  # is_bigendian
    offset += 4  # step
    offset += 4  # data length prefix for uint8[]
    return float(sec) + float(nsec) * 1e-9, int(height), int(width), encoding, offset


def parse_ros1_image(raw: bytes) -> ParsedImage:
    """Deserialize a ROS1 ``sensor_msgs/Image`` message into a numpy array."""
    offset = 0
    offset += 4  # seq
    sec, nsec = struct.unpack_from("<II", raw, offset)
    offset += 8
    frame_id, offset = _read_ros1_string(raw, offset)
    height, width = struct.unpack_from("<II", raw, offset)
    offset += 8
    encoding, offset = _read_ros1_string(raw, offset)
    offset += 1  # is_bigendian
    (step,) = struct.unpack_from("<I", raw, offset)
    offset += 4
    (dlen,) = struct.unpack_from("<I", raw, offset)
    offset += 4
    payload = raw[offset : offset + dlen]
    header_t = float(sec) + float(nsec) * 1e-9

    if encoding == "rgb8":
        arr = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
    elif encoding in ("16UC1", "mono16"):
        arr = np.frombuffer(payload, dtype=np.uint16).reshape(height, width)
    elif encoding == "bgr8":
        arr = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
    else:
        arr = np.frombuffer(payload, dtype=np.uint8)

    return ParsedImage(
        header_t=header_t,
        frame_id=frame_id,
        height=int(height),
        width=int(width),
        encoding=encoding,
        step=int(step),
        data=arr,
    )


def _require_rosbags():
    try:
        from rosbags.rosbag1 import Reader
    except ImportError as exc:
        raise ImportError(
            "Camera bag reading requires the 'rosbags' package. "
            "Install with: pip install rosbags"
        ) from exc
    return Reader


def list_bag_topics(bag_path: PathLike) -> List[Tuple[str, str]]:
    Reader = _require_rosbags()
    with Reader(Path(bag_path)) as reader:
        return [(c.topic, c.msgtype) for c in reader.connections]


def iter_topic_messages(
    bag_path: PathLike,
    topic: str,
) -> Iterator[Tuple[int, bytes]]:
    """Yield ``(bag_record_ts_ns, raw_message)`` for one topic."""
    Reader = _require_rosbags()
    bag_path = Path(bag_path)
    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            raise KeyError(f"Topic not found in bag: {topic}")
        for _conn, ts, raw in reader.messages(connections=connections):
            yield int(ts), bytes(raw)


def index_image_topic(
    bag_path: PathLike,
    topic: str,
    *,
    progress_every: int = 2000,
) -> np.ndarray:
    """Return header timestamps (seconds) for every message on ``topic``."""
    timestamps: List[float] = []
    for i, (_bag_ts, raw) in enumerate(iter_topic_messages(bag_path, topic)):
        hdr_t, _h, _w, _enc, _off = parse_ros1_image_header(raw)
        timestamps.append(hdr_t)
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"  {topic}: indexed {i + 1} frames...", flush=True)
    return np.asarray(timestamps, dtype=np.float64)


def read_image_at_index(
    bag_path: PathLike,
    topic: str,
    frame_idx: int,
) -> ParsedImage:
    """Sequential read until ``frame_idx`` (0-based) on ``topic``."""
    if frame_idx < 0:
        raise IndexError(f"frame_idx must be >= 0, got {frame_idx}")
    for i, (_bag_ts, raw) in enumerate(iter_topic_messages(bag_path, topic)):
        if i == frame_idx:
            return parse_ros1_image(raw)
    raise IndexError(f"frame_idx {frame_idx} out of range for topic {topic}")


def nearest_frame_index(
    query_t: float,
    frame_ts: Sequence[float],
) -> Tuple[int, float]:
    """Return ``(index, delta_ms)`` for the nearest camera frame to ``query_t``."""
    ts = np.asarray(frame_ts, dtype=np.float64)
    if ts.size == 0:
        return -1, float("nan")
    if ts.size == 1:
        return 0, float((query_t - ts[0]) * 1e3)

    idx = int(np.searchsorted(ts, query_t, side="left"))
    idx = min(max(idx, 1), ts.size - 1)
    left, right = idx - 1, idx
    if abs(query_t - ts[left]) <= abs(ts[right] - query_t):
        best = left
    else:
        best = right
    return best, float((query_t - ts[best]) * 1e3)


def nearest_frame_indices(
    query_ts: Sequence[float],
    frame_ts: Sequence[float],
    *,
    max_delta_ms: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each query time to nearest frame index (or -1 if outside threshold)."""
    query_ts = np.asarray(query_ts, dtype=np.float64)
    indices = np.full(query_ts.shape, -1, dtype=np.int64)
    delta_ms = np.full(query_ts.shape, np.nan, dtype=np.float64)
    camera_t = np.full(query_ts.shape, np.nan, dtype=np.float64)
    ts = np.asarray(frame_ts, dtype=np.float64)
    if ts.size == 0:
        return indices, camera_t, delta_ms

    for i, qt in enumerate(query_ts):
        idx, dt = nearest_frame_index(float(qt), ts)
        if abs(dt) <= max_delta_ms:
            indices[i] = idx
            delta_ms[i] = dt
            camera_t[i] = float(ts[idx])
    return indices, camera_t, delta_ms


def bag_file_signature(bag_path: PathLike) -> dict:
    p = Path(bag_path)
    st = p.stat()
    return {"path": str(p.resolve()), "size": int(st.st_size), "mtime": float(st.st_mtime)}


def default_index_path(bag_path: PathLike) -> Path:
    p = Path(bag_path)
    return p.with_suffix(p.suffix + ".camera_index.npz")


def index_image_topics_single_pass(
    bag_path: PathLike,
    topics: Sequence[str],
    *,
    progress_every: int = 2000,
) -> Dict[str, np.ndarray]:
    """Index header timestamps for multiple topics in one bag scan."""
    Reader = _require_rosbags()
    bag_path = Path(bag_path)
    topic_set = set(topics)
    buckets: Dict[str, List[float]] = {t: [] for t in topics}
    counts = {t: 0 for t in topics}

    with Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in topic_set]
        if not connections:
            raise KeyError(f"No requested topics in bag: {topics}")
        for conn, _ts, raw in reader.messages(connections=connections):
            topic = conn.topic
            hdr_t, _h, _w, _enc, _off = parse_ros1_image_header(raw)
            buckets[topic].append(hdr_t)
            counts[topic] += 1
            total = sum(counts.values())
            if progress_every > 0 and total % progress_every == 0:
                print(f"  indexed {total} image messages...", flush=True)

    return {t: np.asarray(buckets[t], dtype=np.float64) for t in topics}


def build_camera_index(
    bag_path: PathLike,
    *,
    color_topic: str = DEFAULT_COLOR_TOPIC,
    depth_topic: str = DEFAULT_DEPTH_TOPIC,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Scan bag once and cache color/depth header timestamps."""
    bag_path = Path(bag_path)
    output_path = Path(output_path) if output_path else default_index_path(bag_path)
    print(f"Indexing camera bag (header timestamps only): {bag_path}")
    print("  (first open on USB can take several minutes)", flush=True)

    indexed = index_image_topics_single_pass(bag_path, [color_topic, depth_topic])
    color_ts = indexed[color_topic]
    depth_ts = indexed[depth_topic]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        color_ts=color_ts,
        depth_ts=depth_ts,
        color_topic=np.asarray(color_topic),
        depth_topic=np.asarray(depth_topic),
        bag_signature=json.dumps(bag_file_signature(bag_path)),
        index_version=np.asarray(INDEX_VERSION),
    )
    print(
        f"Camera index: {len(color_ts)} color, {len(depth_ts)} depth frames -> {output_path}"
    )
    return output_path


def load_camera_index(
    index_path: PathLike,
    *,
    bag_path: Optional[PathLike] = None,
    rebuild: bool = False,
) -> dict:
    """Load cached timestamps; rebuild when missing or stale."""
    index_path = Path(index_path)
    if rebuild or not index_path.is_file():
        if bag_path is None:
            raise FileNotFoundError(
                f"Camera index not found: {index_path}. Pass bag_path to build."
            )
        build_camera_index(bag_path, output_path=index_path)
    data = np.load(index_path, allow_pickle=False)
    meta = {
        "color_ts": np.asarray(data["color_ts"], dtype=np.float64),
        "depth_ts": np.asarray(data["depth_ts"], dtype=np.float64),
        "color_topic": str(np.asarray(data["color_topic"]).item()),
        "depth_topic": str(np.asarray(data["depth_topic"]).item()),
        "index_path": str(index_path.resolve()),
    }
    if bag_path is not None and "bag_signature" in data:
        expected = bag_file_signature(bag_path)
        cached = json.loads(str(np.asarray(data["bag_signature"]).item()))
        if cached.get("path") != expected["path"] or cached.get("size") != expected["size"]:
            print("Camera index cache stale (bag path/size changed); rebuilding...")
            build_camera_index(bag_path, output_path=index_path)
            return load_camera_index(index_path, bag_path=bag_path, rebuild=False)
    return meta


def pick_bag_by_prefix(camera_dir: PathLike, prefix: str) -> Optional[Path]:
    """Find ``{prefix}*.bag`` under ``camera_dir`` (e.g. prefix ``20260510_2128``)."""
    camera_dir = Path(camera_dir)
    matches = sorted(camera_dir.glob(f"{prefix}*.bag"))
    return matches[0] if matches else None


def rgb_to_display(arr: np.ndarray) -> np.ndarray:
    """Return float RGB in [0, 1] for matplotlib."""
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.float32) / 255.0
    raise ValueError(f"Expected HxWx3 rgb array, got {arr.shape}")


def depth_to_display(
    depth_mm: np.ndarray,
    *,
    max_m: float = 10.0,
    invalid_value: int = 0,
) -> np.ndarray:
    """Normalize depth (uint16 mm) to [0, 1] for display."""
    d = depth_mm.astype(np.float32)
    if invalid_value == 0:
        valid = d > 0
    else:
        valid = d != float(invalid_value)
    out = np.zeros_like(d, dtype=np.float32)
    if not np.any(valid):
        return out
    out[valid] = np.clip(d[valid] / (max_m * 1000.0), 0.0, 1.0)
    return out
