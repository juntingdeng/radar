"""Load radar/lidar frames and render range-azimuth + BEV images for sync video."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_SYNC_DIR = Path(__file__).resolve().parent
if str(_SYNC_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNC_DIR))
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
from scipy.interpolate import griddata

# Allow imports from code/utils
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from utils.capon import aoa_capon  # noqa: E402
from utils.cfar import cfar  # noqa: E402


def load_sync_pairs(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load indices and delta_ms only (legacy)."""
    r, l, d, _, _ = load_sync_pairs_full(csv_path)
    return r, l, d


def load_sync_pairs_full(
    csv_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load sync CSV including wall-clock timestamps.

    When the CSV includes camera columns (from ``sync_camera_pairs.py``), also
    returns camera indices and timing. Use ``load_sync_pairs_with_camera`` for the
    extended tuple.
    """
    r, l, d, rt, lt, *_ = load_sync_pairs_with_camera(csv_path)
    return r, l, d, rt, lt


def load_sync_pairs_with_camera(
    csv_path: str | Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load sync CSV; camera fields are -1 / nan when columns are absent."""
    radar_idx: List[int] = []
    lidar_idx: List[int] = []
    delta_ms: List[float] = []
    radar_t: List[float] = []
    lidar_t: List[float] = []
    camera_color_idx: List[int] = []
    camera_depth_idx: List[int] = []
    camera_delta_ms: List[float] = []
    has_camera = False

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_camera = bool(reader.fieldnames and "camera_color_idx" in reader.fieldnames)
        for row in reader:
            radar_idx.append(int(row["radar_idx"]))
            lidar_idx.append(int(row["lidar_idx"]))
            delta_ms.append(float(row["delta_ms"]))
            radar_t.append(float(row["radar_t"]))
            lidar_t.append(float(row["lidar_t"]))
            if has_camera:
                camera_color_idx.append(int(row.get("camera_color_idx", -1)))
                camera_depth_idx.append(int(row.get("camera_depth_idx", -1)))
                raw_cd = row.get("camera_delta_ms", "")
                camera_delta_ms.append(float(raw_cd) if raw_cd not in ("", None) else float("nan"))
            else:
                camera_color_idx.append(-1)
                camera_depth_idx.append(-1)
                camera_delta_ms.append(float("nan"))

    n = len(radar_idx)
    return (
        np.asarray(radar_idx, dtype=np.int64),
        np.asarray(lidar_idx, dtype=np.int64),
        np.asarray(delta_ms, dtype=np.float64),
        np.asarray(radar_t, dtype=np.float64),
        np.asarray(lidar_t, dtype=np.float64),
        np.asarray(camera_color_idx, dtype=np.int64),
        np.asarray(camera_depth_idx, dtype=np.int64),
        np.asarray(camera_delta_ms, dtype=np.float64),
    )


def infer_sync_packets_per_frame(h5_path: str | Path, sync_radar_frame_count: int) -> int:
    """Infer packets-per-frame used when sync CSV was built."""
    with h5py.File(h5_path, "r") as h5f:
        n_packets = int(h5f["scan"]["packet"].shape[0])
    if sync_radar_frame_count <= 0:
        raise ValueError("sync_radar_frame_count must be > 0")
    return max(1, n_packets // sync_radar_frame_count)


SAMPLES_PER_H5_PACKET = 728  # int16 samples per row in collect.py H5 table


def radar_samples_per_frame(radar) -> int:
    return int(radar.num_adc_samples * radar.num_chirps * radar.num_rx)


def effective_adc_packets_per_frame(radar) -> int:
    """Packets needed per ADC frame: ceil(samples_per_frame / 728)."""
    spf = radar_samples_per_frame(radar)
    return (spf + SAMPLES_PER_H5_PACKET - 1) // SAMPLES_PER_H5_PACKET


def sync_radar_idx_to_packet_row(sync_radar_idx: int, sync_packets_per_frame: int) -> int:
    """Map sync CSV radar_idx -> first packet row in H5 for that sync-time block."""
    return int(sync_radar_idx) * int(sync_packets_per_frame)


def radar_total_samples(n_packets: int) -> int:
    return int(n_packets) * SAMPLES_PER_H5_PACKET


def radar_adc_bounds(h5_path: str | Path, radar) -> Tuple[int, int, int, int]:
    """Return (n_packets, samples_per_frame, max_adc_frame, total_samples)."""
    with h5py.File(h5_path, "r") as h5f:
        n_packets = int(h5f["scan"]["packet"].shape[0])
    spf = radar_samples_per_frame(radar)
    total = radar_total_samples(n_packets)
    max_adc_frame = total // spf - 1
    if max_adc_frame < 0:
        raise ValueError("No complete ADC frame found in H5.")
    return n_packets, spf, int(max_adc_frame), int(total)


def packet_row_to_adc_frame(packet_row: int, radar, max_adc_frame: int) -> int:
    """Map H5 packet row to sample-aligned ADC frame index."""
    sample_start = int(packet_row) * SAMPLES_PER_H5_PACKET
    frame_idx = sample_start // radar_samples_per_frame(radar)
    return int(np.clip(frame_idx, 0, max_adc_frame))


def load_radar_adc_frame_at_packet_row(
    h5_path: str | Path,
    packet_row: int,
    radar,
    *,
    max_adc_frame: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Load one ADC frame near an H5 packet row. Returns (frame, adc_frame_idx)."""
    if max_adc_frame is None:
        _, _, max_adc_frame, _ = radar_adc_bounds(h5_path, radar)
    frame_idx = packet_row_to_adc_frame(packet_row, radar, max_adc_frame)
    return load_radar_adc_frame(h5_path, frame_idx, radar), frame_idx


def load_radar_adc_frame_by_time(
    h5_path: str | Path,
    radar_t: float,
    radar,
    packet_ts: np.ndarray,
    max_adc_frame: int,
) -> Tuple[np.ndarray, int]:
    """Load ADC frame for wall-clock time via nearest packet timestamp."""
    frame_idx = radar_t_to_adc_frame(radar_t, packet_ts, radar, max_adc_frame)
    return load_radar_adc_frame(h5_path, frame_idx, radar), frame_idx


def load_radar_adc_frame(h5_path: str | Path, frame_idx: int, radar) -> np.ndarray:
    """Load one ADC frame by sample offset (same logic as process.py)."""
    spf = radar_samples_per_frame(radar)
    sample_start = int(frame_idx) * spf
    sample_end = sample_start + spf

    p0 = sample_start // SAMPLES_PER_H5_PACKET
    p1 = (sample_end + SAMPLES_PER_H5_PACKET - 1) // SAMPLES_PER_H5_PACKET

    with h5py.File(h5_path, "r") as h5f:
        table = h5f["scan"]["packet"]
        if p0 >= table.shape[0]:
            raise IndexError(
                f"Radar frame {frame_idx} starts at sample {sample_start} "
                f"(packet {p0}) but table has {table.shape[0]} rows."
            )
        chunk = table[p0:p1]
        stream = np.array(chunk["packet_data"], dtype=np.uint16).view(np.int16).reshape(-1)

    offset = sample_start - p0 * SAMPLES_PER_H5_PACKET
    adc_raw = stream[offset : offset + spf]
    if adc_raw.size < spf:
        raise ValueError(
            f"Frame {frame_idx}: got {adc_raw.size} samples, expected {spf}."
        )

    frame = adc_raw.reshape(radar.num_chirps, radar.num_rx, radar.num_adc_samples)
    adc_tx = [frame[i:: radar.num_tx, :, :] for i in range(radar.num_tx)]
    return np.concatenate(adc_tx, axis=-2)


def radar_adc_fingerprint(adc_frame: np.ndarray) -> dict:
    """Cheap identity for raw ADC frame (testbench / cache checks)."""
    x = np.asarray(adc_frame, dtype=np.int32)
    return {
        "adc_checksum": int(np.sum(x.astype(np.int64))),
        "adc_abs_max": float(np.max(np.abs(x))),
        "adc_mean": float(np.mean(x)),
    }


def resolve_radar_processing(
    processing: str = "legacy",
    *,
    clutter_removal: Optional[bool] = None,
    doppler_mean_removal: Optional[bool] = None,
    range_window: Optional[str] = None,
    doppler_window: Optional[str] = None,
    zero_doppler_notch: Optional[bool] = None,
    zero_doppler_half_width: Optional[int] = None,
    aoa_method: Optional[str] = None,
) -> dict:
    """Resolve radar FFT/angle options. ``processing='ti'`` approximates TI visualizer demo."""
    mode = str(processing).lower()
    if mode not in ("legacy", "ti"):
        raise ValueError(f"Unknown radar processing {processing!r}; use 'legacy' or 'ti'.")

    opts = {
        "clutter_removal": False,
        "doppler_mean_removal": False,
        "range_window": None,
        "doppler_window": None,
        "zero_doppler_notch": False,
        "zero_doppler_half_width": 2,
        "aoa_method": "capon",
    }
    if mode == "ti":
        opts.update(
            {
                # Chirp mean + Hamming windows + zero-Doppler notch (MTI front-end).
                # Keep Capon for angle estimation; doppler_mean_removal and dbf_nci
                # caused horizontal range stripes / flat noise in practice.
                "clutter_removal": True,
                "doppler_mean_removal": False,
                "range_window": "hamming",
                "doppler_window": "hamming",
                "zero_doppler_notch": True,
                "zero_doppler_half_width": 4,
                "aoa_method": "capon",
            }
        )
    if clutter_removal is not None:
        opts["clutter_removal"] = bool(clutter_removal)
    if doppler_mean_removal is not None:
        opts["doppler_mean_removal"] = bool(doppler_mean_removal)
    if range_window is not None:
        opts["range_window"] = str(range_window) if range_window else None
    if doppler_window is not None:
        opts["doppler_window"] = str(doppler_window) if doppler_window else None
    if zero_doppler_notch is not None:
        opts["zero_doppler_notch"] = bool(zero_doppler_notch)
    if zero_doppler_half_width is not None:
        opts["zero_doppler_half_width"] = int(zero_doppler_half_width)
    if aoa_method is not None:
        opts["aoa_method"] = str(aoa_method).lower()
    if opts["aoa_method"] not in ("capon", "dbf_nci", "dbf_coh"):
        raise ValueError(
            f"Unknown aoa_method {opts['aoa_method']!r}; use capon, dbf_nci, or dbf_coh."
        )
    return opts


def _apply_1d_window(data: np.ndarray, window: str, axis: int) -> np.ndarray:
    n = int(data.shape[axis])
    name = str(window).lower()
    if name == "hamming":
        w = np.hamming(n)
    elif name in ("hanning", "hann"):
        w = np.hanning(n)
    else:
        raise ValueError(f"Unknown FFT window {window!r}; use hamming or hanning.")
    shape = [1] * data.ndim
    shape[axis] = n
    return data * w.reshape(shape)


def _adc_to_range_doppler(adc_data: np.ndarray, proc: dict) -> np.ndarray:
    if proc["clutter_removal"]:
        adc_data = adc_data - adc_data.mean(axis=0, keepdims=True)
    if proc["range_window"]:
        adc_data = _apply_1d_window(adc_data, proc["range_window"], axis=2)
    range_cube = np.fft.fft(adc_data, axis=2).transpose(2, 1, 0)
    if proc["doppler_window"]:
        range_cube = _apply_1d_window(range_cube, proc["doppler_window"], axis=2)
    range_doppler = np.fft.fftshift(np.fft.fft(range_cube, axis=2), axes=2)
    if proc["zero_doppler_notch"]:
        n_dop = range_doppler.shape[2]
        center = n_dop // 2
        hw = int(proc["zero_doppler_half_width"])
        lo = max(0, center - hw)
        hi = min(n_dop, center + hw + 1)
        range_doppler[:, :, lo:hi] = 0
    if proc.get("doppler_mean_removal"):
        range_doppler = range_doppler - range_doppler.mean(axis=2, keepdims=True)
    return range_doppler


def _range_azimuth_power(
    range_doppler_mean: np.ndarray,
    steering_vector: np.ndarray,
    aoa_method: str,
) -> np.ndarray:
    """Angle spectrum per range bin. Returns linear power (not dB)."""
    sv = np.asarray(steering_vector)
    rd = range_doppler_mean
    n_range = int(rd.shape[0])
    n_angles = int(sv.shape[0])
    out = np.zeros((n_range, n_angles), dtype=np.float64)

    if aoa_method == "capon":
        for ri in range(n_range):
            spec = aoa_capon(rd[ri, ...], sv)
            out[ri, :] = np.abs(spec) ** 2
        return out

    if aoa_method == "dbf_nci":
        # Non-coherent sum over Doppler: sum_d |a^H x_d|^2.
        for ri in range(n_range):
            x = rd[ri, ...]
            beams = np.abs(sv @ x) ** 2
            out[ri, :] = np.sum(beams, axis=1)
        return out

    if aoa_method == "dbf_coh":
        # Coherent Doppler sum per antenna, then |a^H s|^2 (lightweight TI-style).
        for ri in range(n_range):
            snap = np.sum(rd[ri, ...], axis=1)
            out[ri, :] = np.abs(sv @ snap) ** 2
        return out

    raise ValueError(f"Unknown aoa_method {aoa_method!r}")


def compute_range_azimuth(
    adc_frame: np.ndarray,
    radar,
    angle: str = "Azimuth",
    *,
    processing: str = "legacy",
    **processing_kw,
) -> np.ndarray:
    """Range-azimuth power map (dB).

    processing:
      - ``legacy`` (default): raw FFT + Capon (matches process.py / visualizer.py).
      - ``ti``: chirp MTI + Hamming windows + zero-Doppler notch, then Capon (legacy AOA).
    """
    proc = resolve_radar_processing(processing, **processing_kw)
    adc_data = adc_frame.astype(np.float64)
    range_doppler = _adc_to_range_doppler(adc_data, proc)

    if angle == "Elevation":
        range_doppler_mean = range_doppler.reshape(
            range_doppler.shape[0], radar.num_rx, radar.num_tx, range_doppler.shape[-1]
        )
        range_doppler_mean = np.mean(range_doppler_mean, axis=1)
        sv_idx = 0
    else:
        range_doppler_mean = range_doppler
        sv_idx = 1

    steering_vector = radar.steering_vectors[sv_idx]
    power = _range_azimuth_power(
        range_doppler_mean, steering_vector, proc["aoa_method"]
    )
    range_azimuth = np.flipud(power)
    range_azimuth = 10 * np.log10(range_azimuth + 1e-12)
    range_azimuth = range_azimuth[range_azimuth.shape[0] // 2 :, :][::2]
    return np.fliplr(range_azimuth)


DEFAULT_SCENE_LATERAL_RANGE = (-40.0, 40.0)
DEFAULT_SCENE_FORWARD_RANGE = (0.0, 20.0)
DEFAULT_SCENE_GRID_RES = 0.25
# Fixed bounds used when writing disk cache (display can crop without rebuild).
CANONICAL_SCENE_LATERAL_RANGE = (-40.0, 40.0)
CANONICAL_SCENE_FORWARD_RANGE = (0.0, 80.0)


def resolve_scene_bounds(
    lateral_range: Optional[Tuple[float, float]] = None,
    forward_range: Optional[Tuple[float, float]] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return (lateral_Y, forward_X) meter bounds for top-down panels."""
    lat = tuple(lateral_range) if lateral_range is not None else DEFAULT_SCENE_LATERAL_RANGE
    fwd = tuple(forward_range) if forward_range is not None else DEFAULT_SCENE_FORWARD_RANGE
    return (float(lat[0]), float(lat[1])), (float(fwd[0]), float(fwd[1]))


def scene_bounds_from_args(args) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    lat = tuple(getattr(args, "scene_lateral", DEFAULT_SCENE_LATERAL_RANGE))
    fwd = tuple(getattr(args, "scene_forward", DEFAULT_SCENE_FORWARD_RANGE))
    return resolve_scene_bounds(lat, fwd)


def canonical_scene_bounds() -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return CANONICAL_SCENE_LATERAL_RANGE, CANONICAL_SCENE_FORWARD_RANGE


def clamp_display_bounds(
    display_lateral: Tuple[float, float],
    display_forward: Tuple[float, float],
    *,
    cache_lateral: Tuple[float, float] = CANONICAL_SCENE_LATERAL_RANGE,
    cache_forward: Tuple[float, float] = CANONICAL_SCENE_FORWARD_RANGE,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Clip display bounds to the cached grid (warn if clipped)."""
    lat = (
        max(display_lateral[0], cache_lateral[0]),
        min(display_lateral[1], cache_lateral[1]),
    )
    fwd = (
        max(display_forward[0], cache_forward[0]),
        min(display_forward[1], cache_forward[1]),
    )
    if lat != tuple(display_lateral) or fwd != tuple(display_forward):
        print(
            "WARNING: display scene bounds clipped to cache grid: "
            f"lateral {display_lateral} -> {lat}, forward {display_forward} -> {fwd}"
        )
    return lat, fwd


def _topdown_bin_slice(
    axis_range: Tuple[float, float],
    res: float,
    sub_range: Tuple[float, float],
) -> slice:
    start, end = axis_range
    lo = max(sub_range[0], start)
    hi = min(sub_range[1], end)
    n_bins = int(round((end - start) / res))
    i0 = int(np.clip(np.round((lo - start) / res), 0, n_bins))
    i1 = int(np.clip(np.round((hi - start) / res), 0, n_bins))
    if i1 <= i0:
        i1 = min(i0 + 1, n_bins)
    return slice(i0, i1)


def crop_topdown_panel(
    panel: np.ndarray,
    *,
    panel_lateral: Tuple[float, float],
    panel_forward: Tuple[float, float],
    display_lateral: Tuple[float, float],
    display_forward: Tuple[float, float],
    res: float = DEFAULT_SCENE_GRID_RES,
) -> np.ndarray:
    """Crop a cached top-down grid to the requested display bounds."""
    if panel.size == 0:
        return panel
    row_sl = _topdown_bin_slice(panel_forward, res, display_forward)
    col_sl = _topdown_bin_slice(panel_lateral, res, display_lateral)
    return panel[row_sl, col_sl]


def scene_topdown_extent(
    lateral_range: Tuple[float, float] = DEFAULT_SCENE_LATERAL_RANGE,
    forward_range: Tuple[float, float] = DEFAULT_SCENE_FORWARD_RANGE,
) -> Tuple[float, float, float, float]:
    """Matplotlib imshow extent: [lateral_left, lateral_right, forward_bottom, forward_top]."""
    return (
        float(lateral_range[0]),
        float(lateral_range[1]),
        float(forward_range[0]),
        float(forward_range[1]),
    )


def radar_range_azimuth_extent(
    radar, shape: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """Physical axes for range-azimuth heatmaps: azimuth (deg) x range (m)."""
    _, n_az = int(shape[0]), int(shape[1])
    range_max = float(radar.max_range) / 2.0
    az_deg = np.linspace(-90.0, 90.0, n_az)
    return (float(az_deg[0]), float(az_deg[-1]), 0.0, range_max)


def lidar_view_is_topdown(view: str) -> bool:
    return str(view).lower() in ("bev", "pointcloud")


def use_radar_bev_panel(args) -> bool:
    if bool(getattr(args, "show_radar_bev", False)):
        return True
    return bool(getattr(args, "align_scene", True)) and lidar_view_is_topdown(
        getattr(args, "lidar_view", "bev")
    )


def range_azimuth_to_bev(
    range_azimuth: np.ndarray,
    radar,
    *,
    lateral_range: Tuple[float, float] = DEFAULT_SCENE_LATERAL_RANGE,
    forward_range: Tuple[float, float] = DEFAULT_SCENE_FORWARD_RANGE,
    res: float = DEFAULT_SCENE_GRID_RES,
) -> np.ndarray:
    """Radar BEV via polar->Cartesian interpolation on the shared scene grid."""
    n_angles = range_azimuth.shape[1]
    # Use full azimuth steering grid (-90..+90 deg) so the BEV fans out naturally.
    angle_bins = np.linspace(-np.pi / 2, np.pi / 2, n_angles)
    range_az = np.flipud(range_azimuth)

    n_range_bins = range_az.shape[0]
    range_bins = np.linspace(0, radar.max_range / 2, n_range_bins)
    r_grid, a_grid = np.meshgrid(range_bins, angle_bins, indexing="ij")
    x = r_grid * np.sin(a_grid)
    y = r_grid * np.cos(a_grid)

    xlin = np.arange(lateral_range[0], lateral_range[1] + res, res)
    ylin = np.arange(forward_range[0], forward_range[1] + res, res)
    x_grid, y_grid = np.meshgrid(xlin, ylin)
    points = np.column_stack((x.flatten(), y.flatten()))
    values = range_az.flatten()
    bev = griddata(points, values, (x_grid, y_grid), method="linear", fill_value=np.nan)
    return bev


def _prepare_scan_points(
    metadata: Any,
    scan: Any,
    *,
    x_range: Tuple[float, float] = (-40.0, 40.0),
    y_range: Tuple[float, float] = DEFAULT_SCENE_FORWARD_RANGE,
    z_range: Tuple[float, float] = (-5.0, 15.0),
    min_range_mm: float = 500.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Filter one scan to valid XYZ points; reflectivity matches staggered XYZ indices."""
    from ouster_compat import scan_range_flat, scan_reflectivity_staggered, scan_to_xyz

    xyz = scan_to_xyz(metadata, scan).reshape(-1, 3)
    ranges = scan_range_flat(scan).reshape(-1)
    valid = np.isfinite(xyz).all(axis=1) & (ranges >= float(min_range_mm))
    valid &= (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])
    # Ouster sensor frame: X=forward, Y=lateral, Z=up.
    valid &= (xyz[:, 0] >= y_range[0]) & (xyz[:, 0] <= y_range[1])
    valid &= (xyz[:, 1] >= x_range[0]) & (xyz[:, 1] <= x_range[1])
    pts = xyz[valid]
    if pts.size == 0:
        return pts, None
    refl = scan_reflectivity_staggered(scan).reshape(-1)[valid].astype(np.float32)
    return pts, refl


def _histogram_topdown(
    pts: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    res: float,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Top-down grid with rows=forward (X), cols=lateral (Y)."""
    x_edges = np.arange(x_range[0], x_range[1] + res, res)
    y_edges = np.arange(y_range[0], y_range[1] + res, res)
    if pts.size == 0:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=np.float32)
    kwargs: dict = {}
    if weights is not None:
        kwargs["weights"] = weights
    grid, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[y_edges, x_edges], **kwargs)
    return grid.astype(np.float32)


def iter_lidar_bev_frames(
    pcap_path: str | Path,
    metadata_json_path: Optional[str | Path],
    *,
    x_range: Tuple[float, float] = (-40.0, 40.0),
    y_range: Tuple[float, float] = DEFAULT_SCENE_FORWARD_RANGE,
    res: float = DEFAULT_SCENE_GRID_RES,
    z_range: Tuple[float, float] = (-5.0, 15.0),
    min_range_mm: float = 500.0,
) -> Iterator[np.ndarray]:
    """Yield lidar BEV intensity images (one per scan) using Ouster SDK."""
    try:
        from ouster_compat import (
            close_source,
            get_ouster_api,
            iter_scans,
            open_pcap_scan_source,
            sensor_info_from_source,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Ouster Python SDK required for lidar BEV. Install with: pip install ouster-sdk"
        ) from exc

    if get_ouster_api() == "legacy" and metadata_json_path is None:
        raise ValueError("metadata_json_path is required for ouster-sdk < 0.16.")

    source = open_pcap_scan_source(pcap_path, metadata_json_path)
    metadata = sensor_info_from_source(source)

    try:
        for scan in iter_scans(source):
            pts, refl = _prepare_scan_points(
                metadata,
                scan,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                min_range_mm=min_range_mm,
            )
            yield _histogram_topdown(pts, x_range, y_range, res, weights=refl)
    finally:
        close_source(source)


def scan_to_bev(
    metadata: Any,
    scan: Any,
    *,
    x_range: Tuple[float, float] = (-40.0, 40.0),
    y_range: Tuple[float, float] = DEFAULT_SCENE_FORWARD_RANGE,
    res: float = DEFAULT_SCENE_GRID_RES,
    z_range: Tuple[float, float] = (-5.0, 15.0),
    min_range_mm: float = 500.0,
) -> np.ndarray:
    """Build lidar BEV from one Ouster scan."""
    pts, refl = _prepare_scan_points(
        metadata,
        scan,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        min_range_mm=min_range_mm,
    )
    return _histogram_topdown(pts, x_range, y_range, res, weights=refl)


def scan_to_pointcloud_panel(
    metadata: Any,
    scan: Any,
    *,
    x_range: Tuple[float, float] = (-40.0, 40.0),
    y_range: Tuple[float, float] = DEFAULT_SCENE_FORWARD_RANGE,
    res: float = DEFAULT_SCENE_GRID_RES,
    z_range: Tuple[float, float] = (-5.0, 15.0),
    min_range_mm: float = 500.0,
) -> np.ndarray:
    """Top-down point density from XYZ (no reflectivity weighting) for PCAP read checks."""
    pts, _ = _prepare_scan_points(
        metadata,
        scan,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        min_range_mm=min_range_mm,
    )
    return _histogram_topdown(pts, x_range, y_range, res)


def scan_to_range_panel(metadata: Any, scan: Any) -> np.ndarray:
    """Native H×W reflectivity image (destaggered) straight from the PCAP scan.

    Row 0 is the top beam (Ouster convention); display with imshow(..., origin='upper').
    """
    from ouster_compat import scan_reflectivity

    return np.asarray(scan_reflectivity(scan, metadata), dtype=np.float32)


def scan_to_lidar_panel(
    metadata: Any,
    scan: Any,
    view: str = "bev",
    **kwargs,
) -> np.ndarray:
    """Render one lidar scan for video: bev | pointcloud | range."""
    view = str(view).lower()
    if view == "bev":
        return scan_to_bev(metadata, scan, **kwargs)
    if view == "pointcloud":
        return scan_to_pointcloud_panel(metadata, scan, **kwargs)
    if view == "range":
        return scan_to_range_panel(metadata, scan)
    raise ValueError(f"Unknown lidar view {view!r}; use bev, pointcloud, or range.")


def lidar_panel_for_imshow(panel: np.ndarray, view: str) -> np.ndarray:
    """Format a cached lidar panel for imshow (matches lidar_frame_testbench)."""
    view = str(view).lower()
    arr = np.asarray(panel, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0)


def lidar_panel_color_limits(
    panel: np.ndarray, view: str
) -> Tuple[float, float]:
    """Per-frame color limits matching lidar_frame_testbench (vmin=0, vmax=p99)."""
    view = str(view).lower()
    arr = np.nan_to_num(np.asarray(panel, dtype=np.float32), nan=0.0)
    nz = arr[arr > 0]
    if nz.size == 0:
        return (0.0, 1.0)
    vmax = float(np.percentile(nz, 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.max(nz))
    return (0.0, max(vmax, 1.0))


def lidar_imshow_origin(view: str) -> str:
    """imshow origin for each lidar view."""
    return "upper" if str(view).lower() == "range" else "lower"


def lidar_panel_axis(
    view: str,
    *,
    lateral_range: Tuple[float, float] = DEFAULT_SCENE_LATERAL_RANGE,
    forward_range: Tuple[float, float] = DEFAULT_SCENE_FORWARD_RANGE,
) -> dict:
    """Axis metadata for lidar imshow panels."""
    view = str(view).lower()
    if view in ("bev", "pointcloud"):
        return {
            "extent": scene_topdown_extent(lateral_range, forward_range),
            "aspect": "equal",
            "xlabel": "Lateral Y (m)",
            "ylabel": "Forward X (m)",
            "origin": "lower",
        }
    return {
        "extent": None,
        "aspect": "auto",
        "xlabel": "azimuth column",
        "ylabel": "beam (0 = top)",
        "origin": "upper",
    }


def radar_panel_axis(
    panel: np.ndarray,
    radar,
    *,
    use_bev: bool,
    lateral_range: Tuple[float, float] = DEFAULT_SCENE_LATERAL_RANGE,
    forward_range: Tuple[float, float] = DEFAULT_SCENE_FORWARD_RANGE,
) -> dict:
    """Axis metadata for radar imshow panels."""
    if use_bev:
        return {
            "extent": scene_topdown_extent(lateral_range, forward_range),
            "aspect": "equal",
            "xlabel": "Lateral Y (m)",
            "ylabel": "Forward X (m)",
            "origin": "lower",
        }
    return {
        "extent": radar_range_azimuth_extent(radar, panel.shape),
        "aspect": "auto",
        "xlabel": "Azimuth (deg)",
        "ylabel": "Range (m)",
        "origin": "lower",
    }


BEV_CACHE_VERSION = 8


def _file_stamp(path: Optional[str | Path]) -> Optional[dict]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    st = p.stat()
    return {"path": str(p.resolve()), "size": int(st.st_size), "mtime": float(st.st_mtime)}


class VizFrameCache:
    """Disk + in-memory cache for radar panels and lidar BEV arrays."""

    def __init__(
        self,
        cache_dir: Optional[str | Path],
        manifest: dict,
        *,
        refresh: bool = False,
    ):
        self.enabled = cache_dir is not None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.refresh = bool(refresh)
        self.manifest = dict(manifest)
        self.lidar_view = str(manifest.get("lidar_view", "bev"))
        self._radar_mem: Dict[int, np.ndarray] = {}
        self._lidar_mem: Dict[int, np.ndarray] = {}

    @classmethod
    def from_render_args(cls, args, *, cfg_path: str, sync_ppf: int) -> "VizFrameCache":
        if getattr(args, "no_cache", False) or not getattr(args, "cache_dir", None):
            return cls(None, {}, refresh=False)
        manifest = {
            "bev_version": BEV_CACHE_VERSION,
            "radar_h5": _file_stamp(args.radar_h5),
            "lidar_pcap": _file_stamp(args.lidar_pcap),
            "lidar_metadata": _file_stamp(args.lidar_metadata),
            "cfg_file": _file_stamp(cfg_path),
            "sync_ppf": int(sync_ppf),
            "radar_load_mode": str(args.radar_load_mode),
            "show_radar_bev": bool(use_radar_bev_panel(args)),
            "lidar_view": str(getattr(args, "lidar_view", "bev")),
            "radar_processing": str(getattr(args, "radar_processing", "legacy")),
            "radar_aoa_method": "capon",
            "align_scene": bool(getattr(args, "align_scene", True)),
            "radar_cache_kind": "range_azimuth",
            "cache_lateral": list(CANONICAL_SCENE_LATERAL_RANGE),
            "cache_forward": list(CANONICAL_SCENE_FORWARD_RANGE),
            "cache_res": float(DEFAULT_SCENE_GRID_RES),
        }
        cache = cls(args.cache_dir, manifest, refresh=getattr(args, "refresh_cache", False))
        if cache.enabled and not cache.refresh and not cache._manifest_matches_disk():
            print("Cache manifest mismatch with inputs; missing frames will be recomputed.")
        return cache

    def _manifest_path(self) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / "manifest.json"

    def _manifest_matches_disk(self) -> bool:
        if not self.enabled or self.refresh:
            return False
        path = self._manifest_path()
        if not path.is_file():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
        for key, value in self.manifest.items():
            if on_disk.get(key) != value:
                return False
        return True

    def load_disk_manifest_extra(self) -> dict:
        if not self.enabled:
            return {}
        path = self._manifest_path()
        if not path.is_file():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: v for k, v in data.items() if k not in self.manifest}
        except (json.JSONDecodeError, OSError):
            return {}

    def save_manifest(self, extra: Optional[dict] = None) -> None:
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(self.manifest)
        if extra:
            payload.update(extra)
        with open(self._manifest_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _radar_ra_path(self, adc_frame: int) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / "radar" / f"ra_{int(adc_frame):06d}.npy"

    def _radar_path(self, adc_frame: int) -> Path:
        """Legacy rendered-panel cache (pre-v8)."""
        assert self.cache_dir is not None
        return self.cache_dir / "radar" / f"adc_{int(adc_frame):06d}.npy"

    def _lidar_path(self, lidx: int) -> Path:
        assert self.cache_dir is not None
        prefix = {"bev": "scan", "pointcloud": "pc", "range": "range"}.get(
            self.lidar_view, "scan"
        )
        return self.cache_dir / "lidar" / f"{prefix}_{int(lidx):06d}.npy"

    def has_lidar(self, lidx: int) -> bool:
        if not self.enabled or self.refresh:
            return False
        lidx = int(lidx)
        if lidx in self._lidar_mem:
            return True
        return self._lidar_path(lidx).is_file()

    def get_lidar(self, lidx: int) -> Optional[np.ndarray]:
        lidx = int(lidx)
        if lidx in self._lidar_mem:
            return self._lidar_mem[lidx]
        if not self.enabled or self.refresh:
            return None
        path = self._lidar_path(lidx)
        if path.is_file():
            arr = np.load(path)
            self._lidar_mem[lidx] = arr
            return arr
        return None

    def put_lidar(self, lidx: int, bev: np.ndarray) -> None:
        lidx = int(lidx)
        self._lidar_mem[lidx] = bev
        if self.enabled:
            path = self._lidar_path(lidx)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, bev)

    def has_radar(self, adc_frame: int) -> bool:
        if not self.enabled or self.refresh:
            return False
        adc_frame = int(adc_frame)
        if adc_frame in self._radar_mem:
            return True
        return self._radar_ra_path(adc_frame).is_file()

    def get_radar(self, adc_frame: int) -> Optional[np.ndarray]:
        """Load cached range-azimuth (dB); BEV is derived at display time."""
        adc_frame = int(adc_frame)
        if adc_frame in self._radar_mem:
            return self._radar_mem[adc_frame]
        if not self.enabled or self.refresh:
            return None
        path = self._radar_ra_path(adc_frame)
        if path.is_file():
            arr = np.load(path)
            self._radar_mem[adc_frame] = arr
            return arr
        return None

    def put_radar(self, adc_frame: int, range_azimuth: np.ndarray) -> None:
        adc_frame = int(adc_frame)
        self._radar_mem[adc_frame] = range_azimuth
        if self.enabled:
            path = self._radar_ra_path(adc_frame)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, range_azimuth)

    def all_lidar_cached(self, lidar_indices: List[int]) -> bool:
        if not self.enabled or self.refresh:
            return False
        return all(self.has_lidar(i) for i in lidar_indices)


def build_radar_frame_time_table(
    packet_ts: np.ndarray, radar, max_adc_frame: int
) -> np.ndarray:
    """Per-ADC-frame timestamp at the sample-aligned start of each frame."""
    spf = radar_samples_per_frame(radar)
    packet_idx = (np.arange(max_adc_frame + 1, dtype=np.int64) * spf) // SAMPLES_PER_H5_PACKET
    packet_idx = np.clip(packet_idx, 0, packet_ts.size - 1)
    times = packet_ts[packet_idx].astype(np.float64)
    # Enforce non-decreasing times for searchsorted.
    return np.maximum.accumulate(times)


def radar_t_to_adc_frame(
    radar_t: float,
    packet_ts: np.ndarray,
    radar,
    max_adc_frame: int,
) -> int:
    """Map wall-clock time to ADC frame via nearest packet timestamp."""
    rt = float(radar_t)
    if packet_ts.size == 0:
        return 0
    # Prefer searchsorted when timestamps are monotonic (fast).
    if np.all(packet_ts[1:] >= packet_ts[:-1]):
        pi = int(np.searchsorted(packet_ts, rt, side="left"))
        if pi >= packet_ts.size:
            pi = packet_ts.size - 1
        elif pi > 0 and abs(packet_ts[pi] - rt) > abs(packet_ts[pi - 1] - rt):
            pi -= 1
    else:
        pi = int(np.argmin(np.abs(packet_ts - rt)))
    sample_start = pi * SAMPLES_PER_H5_PACKET
    frame_idx = sample_start // radar_samples_per_frame(radar)
    return int(np.clip(frame_idx, 0, max_adc_frame))


def sync_radar_idx_to_adc_frame(
    radar_sync_idx: int, sync_packets_per_frame: int, radar, max_adc_frame: int
) -> int:
    """Map sync CSV radar_idx to ADC frame (same packet blocks as sync_radar_lidar.py)."""
    packet_row = int(radar_sync_idx) * int(sync_packets_per_frame)
    sample_start = packet_row * SAMPLES_PER_H5_PACKET
    frame_idx = sample_start // radar_samples_per_frame(radar)
    return int(np.clip(frame_idx, 0, max_adc_frame))


class LidarScanReader:
    """Lidar BEV via PCAP access in sync-order (sequential iter_scans indexing)."""

    def __init__(
        self,
        pcap_path: str | Path,
        metadata_json_path: Optional[str | Path],
        *,
        frame_cache: Optional[VizFrameCache] = None,
    ):
        self._pcap_path = Path(pcap_path)
        self._metadata_json_path = (
            Path(metadata_json_path) if metadata_json_path is not None else None
        )
        self._frame_cache = frame_cache
        self._bev_cache: Dict[int, np.ndarray] = {}
        self._source = None
        self._scan_iter: Optional[Iterator[Any]] = None
        self._stream_pos = 0
        self._open_source()
        self._validate_scan_variation()

    def _open_source(self) -> None:
        from ouster_compat import (
            close_source,
            iter_scans,
            open_pcap_scan_source,
            scan_source_length,
            sensor_info_from_source,
        )

        if self._source is not None:
            close_source(self._source)
        self._source = open_pcap_scan_source(
            self._pcap_path, self._metadata_json_path, index=True
        )
        self.metadata = sensor_info_from_source(self._source)
        self._len = scan_source_length(self._source)
        self._scan_iter = iter_scans(self._source)
        self._stream_pos = 0

    def _validate_scan_variation(self) -> None:
        if self._len is None or self._len < 2:
            return
        from ouster_compat import scan_range_checksum

        probe_b = min(10, self._len - 1)
        scan_a = self._get_scan_sync_order(0)
        checksum_a = scan_range_checksum(scan_a)
        scan_b = self._get_scan_sync_order(probe_b)
        checksum_b = scan_range_checksum(scan_b)
        self._open_source()
        if checksum_a == checksum_b:
            print(
                f"WARNING: lidar scans 0 and {probe_b} have identical RANGE checksum; "
                "PCAP indexing may be wrong."
            )
        else:
            print(
                f"  lidar scan checksums: idx0={checksum_a} idx{probe_b}={checksum_b} (vary OK)"
            )

    def _get_scan_sync_order(self, scan_idx: int) -> Any:
        """Advance sequential iterator to scan_idx (matches sync CSV lidar_idx)."""
        if self._scan_iter is None:
            raise RuntimeError("Lidar scan iterator not initialized.")
        scan_idx = int(scan_idx)
        if scan_idx < self._stream_pos:
            self._open_source()
        while self._stream_pos <= scan_idx:
            try:
                scan = next(self._scan_iter)
            except StopIteration as exc:
                raise IndexError(f"Scan index {scan_idx} not in PCAP.") from exc
            if self._stream_pos == scan_idx:
                self._stream_pos += 1
                return scan
            self._stream_pos += 1
        raise IndexError(f"Scan index {scan_idx} not in PCAP.")

    def __len__(self) -> int:
        if self._len is None:
            raise RuntimeError("PCAP scan count unknown; cannot validate lidar_idx.")
        return self._len

    def raw_fingerprint(self, scan_idx: int) -> dict:
        """Read raw LidarScan at sync-order index (bypasses BEV/disk cache)."""
        from ouster_compat import scan_fingerprint

        scan = self._get_scan_sync_order(int(scan_idx))
        return scan_fingerprint(scan)

    def bev_from_scan_idx(self, scan_idx: int, **bev_kw) -> np.ndarray:
        """Compute BEV from PCAP without using disk cache (for cache validation)."""
        scan = self._get_scan_sync_order(int(scan_idx))
        return scan_to_bev(self.metadata, scan, **bev_kw)

    def panel_from_scan_idx(self, scan_idx: int, view: str = "bev", **kwargs) -> np.ndarray:
        """Compute lidar panel from PCAP without disk cache."""
        scan = self._get_scan_sync_order(int(scan_idx))
        return scan_to_lidar_panel(self.metadata, scan, view=view, **kwargs)

    def get_bev(self, scan_idx: int, **bev_kw) -> np.ndarray:
        return self.get_panel(scan_idx, view="bev", **bev_kw)

    def get_panel(self, scan_idx: int, *, view: str = "bev", **kwargs) -> np.ndarray:
        scan_idx = int(scan_idx)
        cache_key = (view, scan_idx)
        if cache_key in self._bev_cache:
            return self._bev_cache[cache_key]
        if self._frame_cache is not None and self._frame_cache.lidar_view == view:
            cached = self._frame_cache.get_lidar(scan_idx)
            if cached is not None:
                self._bev_cache[cache_key] = cached
                return cached
        scan = self._get_scan_sync_order(scan_idx)
        panel = scan_to_lidar_panel(self.metadata, scan, view=view, **kwargs)
        self._bev_cache[cache_key] = panel
        if self._frame_cache is not None and self._frame_cache.lidar_view == view:
            self._frame_cache.put_lidar(scan_idx, panel)
        return panel

    def close(self) -> None:
        from ouster_compat import close_source

        if self._source is not None:
            close_source(self._source)
            self._source = None


def _pick_lidar_probe_indices(unique_lidar: List[int], *, full: bool) -> List[int]:
    if not unique_lidar:
        return []
    if full:
        if len(unique_lidar) <= 12:
            return list(unique_lidar)
        step = max(1, len(unique_lidar) // 10)
        picked = list(unique_lidar[::step])
        if unique_lidar[-1] not in picked:
            picked.append(unique_lidar[-1])
        return sorted(set(picked))
    mid = unique_lidar[len(unique_lidar) // 2]
    return sorted({unique_lidar[0], mid, unique_lidar[-1]})


def print_lidar_diagnosis(
    lidar_indices: List[int],
    *,
    load_bev_display: Callable[[int], np.ndarray],
    raw_fingerprint: Optional[Callable[[int], dict]] = None,
    fresh_bev: Optional[Callable[[int], np.ndarray]] = None,
    full: bool = False,
) -> dict:
    """
    Layered lidar diagnostic:
      1) raw PCAP scan fingerprints (RANGE checksum / timestamp) — reading
      2) BEV array stats — post-processing (histogram / axes)
      3) optional cached-vs-fresh BEV diff — stale disk cache
    """
    probe = _pick_lidar_probe_indices(sorted({int(i) for i in lidar_indices}), full=full)
    if len(probe) < 2:
        print("Lidar diagnostic: need at least 2 unique lidar_idx values.")
        return {"ok": False}

    print("\n=== Lidar diagnostic (reading vs BEV) ===")
    if raw_fingerprint is None:
        print("  [raw PCAP] skipped — PCAP not open (BEV/disk-cache only).")
        print("  Use --diagnose_lidar to force raw scan verification.")
    else:
        print(
            "  Columns: lidar_idx | range_checksum | timestamp | "
            "bev_max | bev_nz | mean|Δ| vs first"
        )

    first_bev: Optional[np.ndarray] = None
    checksums: List[int] = []
    bev_diffs: List[float] = []
    cache_diffs: List[float] = []

    for lidx in probe:
        bev = load_bev_display(lidx)
        if first_bev is None:
            first_bev = bev
        bev_diff = float(np.mean(np.abs(bev - first_bev))) if first_bev is not None else 0.0
        bev_diffs.append(bev_diff)

        cksum = ts = None
        if raw_fingerprint is not None:
            fp = raw_fingerprint(lidx)
            cksum = int(fp["range_checksum"])
            checksums.append(cksum)
            ts = fp.get("timestamp_s")
            ts_s = f"{ts:.3f}" if ts is not None else "n/a"
            cache_note = ""
            if fresh_bev is not None:
                fresh = np.nan_to_num(fresh_bev(lidx).T, nan=0.0)
                cache_diff = float(np.mean(np.abs(bev - fresh)))
                cache_diffs.append(cache_diff)
                if cache_diff > 1e-3:
                    cache_note = f"  cache≠fresh {cache_diff:.4f}"
            print(
                f"  {lidx:5d} | {cksum:14d} | {ts_s:>11s} | "
                f"{float(np.max(bev)):9.1f} | {int(np.count_nonzero(bev)):6d} | "
                f"{bev_diff:8.4f}{cache_note}"
            )
        else:
            print(
                f"  {lidx:5d} | {'n/a':>14s} | {'n/a':>11s} | "
                f"{float(np.max(bev)):9.1f} | {int(np.count_nonzero(bev)):6d} | "
                f"{bev_diff:8.4f}"
            )

    unique_ck = len(set(checksums))
    bev_span = float(max(bev_diffs) if bev_diffs else 0.0)
    print("---")
    if raw_fingerprint is not None:
        if unique_ck < len(checksums):
            print(
                f"  READING: FAIL — only {unique_ck}/{len(checksums)} unique RANGE "
                "checksums (likely stuck on one PCAP scan / bad index)."
            )
        else:
            print(
                f"  READING: OK — {unique_ck}/{len(checksums)} probes have distinct "
                "raw RANGE checksums."
            )
    if bev_span < 1e-3:
        print(
            "  BEV: FAIL — histogram output identical across probes "
            "(post-processing or empty scene)."
        )
    elif bev_span < 0.5:
        print(
            f"  BEV: subtle — mean|Δ| up to {bev_span:.4f} "
            "(real motion may be hard to see with --fixed_color_scale)."
        )
    else:
        print(f"  BEV: OK — mean|Δ| up to {bev_span:.4f} across probes.")

    if cache_diffs and max(cache_diffs) > 1e-3:
        print(
            "  CACHE: WARN — disk cache BEV differs from fresh PCAP BEV; "
            "try --refresh_cache."
        )
    elif cache_diffs:
        print("  CACHE: OK — disk cache matches fresh PCAP BEV on probed indices.")

    print("=== end lidar diagnostic ===\n")
    return {
        "ok": (raw_fingerprint is None or unique_ck == len(checksums)) and bev_span >= 1e-3,
        "unique_range_checksums": unique_ck,
        "bev_mean_diff_max": bev_span,
    }


def build_lidar_bev_index(
    pcap_path: str | Path,
    metadata_json_path: Optional[str | Path],
    max_index: int,
    **kwargs,
) -> Dict[int, np.ndarray]:
    """Read pcap once and cache BEV frames up to max_index (inclusive). Prefer LidarScanReader."""
    cache: Dict[int, np.ndarray] = {}
    for i, bev in enumerate(iter_lidar_bev_frames(pcap_path, metadata_json_path, **kwargs)):
        cache[i] = bev
        if i >= max_index:
            break
    return cache
