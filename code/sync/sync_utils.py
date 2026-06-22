"""Utilities for radar/lidar timestamp extraction and synchronization."""

from __future__ import annotations

import csv
import json
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sys
from pathlib import Path

import h5py
import numpy as np

_SYNC_DIR = Path(__file__).resolve().parent
if str(_SYNC_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNC_DIR))


@dataclass
class SyncPair:
    radar_idx: int
    lidar_idx: int
    radar_t: float
    lidar_t: float
    delta_ms: float


def read_radar_packet_timestamps(h5_path: str | Path) -> np.ndarray:
    """Load radar packet timestamps from TI collector H5."""
    with h5py.File(h5_path, "r") as h5f:
        packet_table = h5f["scan"]["packet"]
        ts = np.asarray(packet_table["t"], dtype=np.float64)
    return ts


def radar_packets_to_frames(
    packet_ts: Sequence[float], packets_per_frame: int
) -> np.ndarray:
    """Convert radar packet timestamps to per-frame timestamps by block averaging."""
    if packets_per_frame <= 0:
        raise ValueError("packets_per_frame must be > 0")
    packet_ts = np.asarray(packet_ts, dtype=np.float64)
    n_full = packet_ts.size // packets_per_frame
    if n_full == 0:
        return np.empty((0,), dtype=np.float64)
    trimmed = packet_ts[: n_full * packets_per_frame]
    return trimmed.reshape(n_full, packets_per_frame).mean(axis=1)


def _read_pcap_global_header(fp) -> Tuple[str, bool]:
    """Return (endianness, nano_ts). Supports classic PCAP only."""
    magic = fp.read(4)
    if len(magic) != 4:
        raise ValueError("Invalid pcap: file too small.")
    if magic == b"\xd4\xc3\xb2\xa1":
        endian, nano_ts = "<", False
    elif magic == b"\xa1\xb2\xc3\xd4":
        endian, nano_ts = ">", False
    elif magic == b"\x4d\x3c\xb2\xa1":
        endian, nano_ts = "<", True
    elif magic == b"\xa1\xb2\x3c\x4d":
        endian, nano_ts = ">", True
    else:
        raise ValueError("Unsupported pcap magic (pcapng not supported in this script).")
    rest = fp.read(20)
    if len(rest) != 20:
        raise ValueError("Invalid pcap global header.")
    return endian, nano_ts


def _extract_udp_ports_from_ethernet(pkt: bytes) -> Optional[Tuple[int, int]]:
    """Extract (src_port, dst_port) from Ethernet+IPv4+UDP packet, if available."""
    if len(pkt) < 14:
        return None
    ether_type = struct.unpack("!H", pkt[12:14])[0]
    if ether_type != 0x0800:  # IPv4
        return None
    if len(pkt) < 34:
        return None
    ip_start = 14
    version_ihl = pkt[ip_start]
    ihl = (version_ihl & 0x0F) * 4
    if ihl < 20:
        return None
    protocol = pkt[ip_start + 9]
    if protocol != 17:  # UDP
        return None
    udp_start = ip_start + ihl
    if len(pkt) < udp_start + 8:
        return None
    src_port, dst_port = struct.unpack("!HH", pkt[udp_start : udp_start + 4])
    return src_port, dst_port


def read_lidar_packet_timestamps_from_pcap(
    pcap_path: str | Path,
    udp_port: Optional[int] = 7502,
) -> np.ndarray:
    """Read lidar UDP packet timestamps from pcap record headers."""
    timestamps: List[float] = []
    with open(pcap_path, "rb") as fp:
        endian, nano_ts = _read_pcap_global_header(fp)
        rec_struct = struct.Struct(endian + "IIII")
        ts_scale = 1e-9 if nano_ts else 1e-6

        while True:
            rec = fp.read(16)
            if not rec:
                break
            if len(rec) != 16:
                break
            ts_sec, ts_frac, incl_len, _orig_len = rec_struct.unpack(rec)
            pkt = fp.read(incl_len)
            if len(pkt) != incl_len:
                break

            if udp_port is not None:
                ports = _extract_udp_ports_from_ethernet(pkt)
                if ports is None:
                    continue
                if udp_port not in ports:
                    continue

            timestamps.append(float(ts_sec) + float(ts_frac) * ts_scale)
    return np.asarray(timestamps, dtype=np.float64)


def lidar_packets_to_frames(
    lidar_packet_ts: Sequence[float], packets_per_frame: int
) -> np.ndarray:
    """Convert lidar packet timestamps to per-frame timestamps by block averaging."""
    if packets_per_frame <= 0:
        raise ValueError("packets_per_frame must be > 0")
    lidar_packet_ts = np.asarray(lidar_packet_ts, dtype=np.float64)
    n_full = lidar_packet_ts.size // packets_per_frame
    if n_full == 0:
        return np.empty((0,), dtype=np.float64)
    trimmed = lidar_packet_ts[: n_full * packets_per_frame]
    return trimmed.reshape(n_full, packets_per_frame).mean(axis=1)


def try_read_ouster_scan_timestamps(
    pcap_path: str | Path,
    metadata_json_path: Optional[str | Path],
) -> Optional[np.ndarray]:
    """Read scan timestamps using Ouster SDK when available; otherwise return None."""
    try:
        from ouster_compat import (
            close_source,
            get_ouster_api,
            iter_scans,
            open_pcap_scan_source,
            scan_timestamp_min_s,
        )
    except Exception:
        return None

    if get_ouster_api() == "legacy" and metadata_json_path is None:
        return None

    source = open_pcap_scan_source(pcap_path, metadata_json_path)
    timestamps: List[float] = []
    try:
        from ouster_compat import scan_source_length

        n_hint = scan_source_length(source)
        if n_hint is not None:
            print(
                f"Reading lidar scan timestamps from PCAP ({n_hint} scans; "
                "no progress for a few minutes on external drives)..."
            )
        else:
            print("Reading lidar scan timestamps from PCAP (scan count unknown)...")
        for i, scan in enumerate(iter_scans(source)):
            ts_s = scan_timestamp_min_s(scan)
            if ts_s is not None:
                timestamps.append(ts_s)
            if n_hint is not None and (i + 1) % 2000 == 0:
                print(f"  lidar timestamps: {i + 1}/{n_hint} scans")
        if timestamps:
            print(f"  lidar timestamps: done ({len(timestamps)} scans)")
    finally:
        close_source(source)
    if not timestamps:
        return None
    return np.asarray(timestamps, dtype=np.float64)


def format_unix_utc(ts: float) -> str:
    """Human-readable UTC time for a Unix timestamp."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def estimate_timeline_overlap_s(
    radar_ts: Sequence[float],
    lidar_ts: Sequence[float],
    *,
    lidar_time_scale: float = 1.0,
    lidar_to_radar_offset_s: float = 0.0,
) -> float:
    """Overlap duration in seconds after mapping lidar -> radar time (can be negative)."""
    radar_ts = np.asarray(radar_ts, dtype=np.float64)
    lidar_adj = adjust_lidar_timestamps(
        lidar_ts,
        lidar_time_scale=lidar_time_scale,
        lidar_to_radar_offset_s=lidar_to_radar_offset_s,
    )
    if radar_ts.size == 0 or lidar_adj.size == 0:
        return 0.0
    return float(min(radar_ts[-1], lidar_adj[-1]) - max(radar_ts[0], lidar_adj[0]))


def print_timeline_overlap_report(
    radar_ts: Sequence[float],
    lidar_ts: Sequence[float],
    *,
    lidar_time_scale: float = 1.0,
    lidar_to_radar_offset_s: float = 0.0,
) -> float:
    """Print radar/lidar spans and overlap; return overlap seconds."""
    radar_ts = np.asarray(radar_ts, dtype=np.float64)
    lidar_raw = np.asarray(lidar_ts, dtype=np.float64)
    lidar_adj = adjust_lidar_timestamps(
        lidar_raw,
        lidar_time_scale=lidar_time_scale,
        lidar_to_radar_offset_s=lidar_to_radar_offset_s,
    )
    overlap_s = estimate_timeline_overlap_s(
        radar_ts,
        lidar_raw,
        lidar_time_scale=lidar_time_scale,
        lidar_to_radar_offset_s=lidar_to_radar_offset_s,
    )

    print(
        f"Radar: {format_unix_utc(radar_ts[0])} .. {format_unix_utc(radar_ts[-1])} "
        f"({radar_ts[-1] - radar_ts[0]:.1f} s, {radar_ts.size} frames)"
    )
    print(
        f"Lidar raw: {format_unix_utc(lidar_raw[0])} .. {format_unix_utc(lidar_raw[-1])} "
        f"({lidar_raw[-1] - lidar_raw[0]:.1f} s, {lidar_raw.size} frames)"
    )
    if lidar_time_scale != 1.0 or lidar_to_radar_offset_s != 0.0:
        print(
            f"Lidar mapped: {format_unix_utc(lidar_adj[0])} .. {format_unix_utc(lidar_adj[-1])} "
            f"(scale={lidar_time_scale:.9f}, offset={lidar_to_radar_offset_s:+.3f} s)"
        )

    if overlap_s > 0:
        print(f"Timeline overlap: {overlap_s:.1f} s")
    else:
        gap_s = max(radar_ts[0], lidar_adj[0]) - min(radar_ts[-1], lidar_adj[-1])
        print(
            "ERROR: radar and lidar timelines do not overlap.\n"
            f"  Gap between recordings: {gap_s:.1f} s ({gap_s / 3600.0:.2f} h)\n"
            "  --fit_offset cannot fix this (it only shifts by seconds, not hours).\n"
            "  Use radar/lidar files recorded at the same time, or check file paths."
        )
    return overlap_s


def adjust_lidar_timestamps(
    lidar_ts: Sequence[float],
    *,
    lidar_time_scale: float = 1.0,
    lidar_to_radar_offset_s: float = 0.0,
) -> np.ndarray:
    """Map raw lidar times to radar time: adjusted = scale * raw + offset."""
    lidar_ts = np.asarray(lidar_ts, dtype=np.float64)
    return lidar_ts * float(lidar_time_scale) + float(lidar_to_radar_offset_s)


def _median_nearest_neighbor_error_s(
    radar_ts: np.ndarray, adjusted_lidar_ts: np.ndarray
) -> float:
    """Median |radar_nearest - adjusted_lidar| in seconds."""
    if adjusted_lidar_ts.size == 0:
        return float("inf")
    if radar_ts.size == 1:
        return float(np.median(np.abs(radar_ts[0] - adjusted_lidar_ts)))

    idx = np.searchsorted(radar_ts, adjusted_lidar_ts, side="left")
    idx = np.clip(idx, 1, radar_ts.size - 1)
    left = radar_ts[idx - 1]
    right = radar_ts[idx]
    nearest = np.where(
        np.abs(adjusted_lidar_ts - left) <= np.abs(right - adjusted_lidar_ts),
        left,
        right,
    )
    return float(np.median(np.abs(nearest - adjusted_lidar_ts)))


def fit_time_offset(
    radar_ts: Sequence[float],
    lidar_ts: Sequence[float],
    max_abs_shift_s: float = 5.0,
    n_candidates: int = 2001,
) -> float:
    """Estimate constant lidar->radar offset (scale=1; radar ~= lidar + offset)."""
    scale, offset, _ = fit_time_affine(
        radar_ts,
        lidar_ts,
        max_abs_shift_s=max_abs_shift_s,
        max_skew_ppm=0.0,
        n_offset_candidates=n_candidates,
        n_scale_candidates=1,
    )
    if scale != 1.0:
        raise RuntimeError("fit_time_offset internal error: expected scale=1.")
    return offset


def _fit_offset_for_scale(
    radar_ts: np.ndarray,
    lidar_ts: np.ndarray,
    scale: float,
    max_abs_shift_s: float,
    n_candidates: int,
) -> Tuple[float, float]:
    """Best offset and median NN error for a fixed lidar time scale."""
    scaled = lidar_ts * scale
    shift_grid = np.linspace(-max_abs_shift_s, max_abs_shift_s, n_candidates)
    best_shift = 0.0
    best_score = float("inf")
    for shift in shift_grid:
        score = _median_nearest_neighbor_error_s(radar_ts, scaled + shift)
        if score < best_score:
            best_score = score
            best_shift = float(shift)
    return best_shift, best_score


def fit_time_affine(
    radar_ts: Sequence[float],
    lidar_ts: Sequence[float],
    max_abs_shift_s: float = 5.0,
    max_skew_ppm: float = 200.0,
    n_offset_candidates: int = 401,
    n_scale_candidates: int = 41,
) -> Tuple[float, float, float]:
    """Estimate lidar->radar affine map: radar ~= scale * lidar + offset.

    Coarse grid over scale, 1D offset search per scale, then local refine.
    Returns (scale, offset_s, median_error_s).
    """
    radar_ts = np.asarray(radar_ts, dtype=np.float64)
    lidar_ts = np.asarray(lidar_ts, dtype=np.float64)
    if radar_ts.size == 0 or lidar_ts.size == 0:
        raise ValueError("Empty timestamps are not syncable.")

    if max_skew_ppm <= 0.0 or n_scale_candidates <= 1:
        offset, score = _fit_offset_for_scale(
            radar_ts,
            lidar_ts,
            1.0,
            max_abs_shift_s,
            max(n_offset_candidates, 3),
        )
        return 1.0, offset, score

    ppm_grid = np.linspace(-max_skew_ppm, max_skew_ppm, n_scale_candidates)
    best_scale = 1.0
    best_offset = 0.0
    best_score = float("inf")
    best_ppm = 0.0

    for ppm in ppm_grid:
        scale = 1.0 + float(ppm) * 1e-6
        offset, score = _fit_offset_for_scale(
            radar_ts,
            lidar_ts,
            scale,
            max_abs_shift_s,
            n_offset_candidates,
        )
        if score < best_score:
            best_score = score
            best_scale = scale
            best_offset = offset
            best_ppm = float(ppm)

    if n_scale_candidates >= 3:
        step = float(ppm_grid[1] - ppm_grid[0])
        fine_ppm = np.linspace(
            best_ppm - step,
            best_ppm + step,
            max(9, n_scale_candidates // 4),
        )
        for ppm in fine_ppm:
            scale = 1.0 + float(ppm) * 1e-6
            offset, score = _fit_offset_for_scale(
                radar_ts,
                lidar_ts,
                scale,
                max_abs_shift_s,
                max(n_offset_candidates, 801),
            )
            if score < best_score:
                best_score = score
                best_scale = scale
                best_offset = offset

    return best_scale, best_offset, best_score


def compute_pair_timing_drift(
    pairs: Sequence[SyncPair],
    raw_lidar_frame_ts: Sequence[float],
) -> Dict[str, Optional[float]]:
    """Linear drift of matched-pair delta_ms vs raw lidar time (diagnostic)."""
    if not pairs:
        return {
            "delta_ms_drift_slope_ms_per_s": None,
            "delta_ms_drift_intercept_ms": None,
            "delta_ms_drift_total_ms": None,
            "implied_skew_ppm_from_pairs": None,
        }

    raw_lidar_frame_ts = np.asarray(raw_lidar_frame_ts, dtype=np.float64)
    raw_t = np.asarray([raw_lidar_frame_ts[p.lidar_idx] for p in pairs], dtype=np.float64)
    delta = np.asarray([p.delta_ms for p in pairs], dtype=np.float64)
    t_rel = raw_t - raw_t[0]
    if t_rel.size < 2 or float(t_rel[-1]) <= 0.0:
        return {
            "delta_ms_drift_slope_ms_per_s": 0.0,
            "delta_ms_drift_intercept_ms": float(np.median(delta)),
            "delta_ms_drift_total_ms": 0.0,
            "implied_skew_ppm_from_pairs": 0.0,
        }

    slope, intercept = np.linalg.lstsq(
        np.vstack([t_rel, np.ones_like(t_rel)]).T,
        delta,
        rcond=None,
    )[0]
    duration = float(t_rel[-1])
    # If radar ~= scale*lidar+offset, scale error shows as roughly (1-scale)*1000 ms/s.
    implied_ppm = float(-slope * 1000.0)
    return {
        "delta_ms_drift_slope_ms_per_s": float(slope),
        "delta_ms_drift_intercept_ms": float(intercept),
        "delta_ms_drift_total_ms": float(slope * duration),
        "implied_skew_ppm_from_pairs": implied_ppm,
    }


def nearest_neighbor_pairs(
    radar_ts: Sequence[float],
    lidar_ts: Sequence[float],
    lidar_to_radar_offset_s: float = 0.0,
    lidar_time_scale: float = 1.0,
    max_delta_ms: float = 100.0,
) -> List[SyncPair]:
    """Match each lidar timestamp to nearest radar timestamp."""
    radar_ts = np.asarray(radar_ts, dtype=np.float64)
    lidar_ts = adjust_lidar_timestamps(
        lidar_ts,
        lidar_time_scale=lidar_time_scale,
        lidar_to_radar_offset_s=lidar_to_radar_offset_s,
    )
    if radar_ts.size == 0 or lidar_ts.size == 0:
        return []
    if radar_ts.size == 1:
        delta_ms = (radar_ts[0] - lidar_ts) * 1e3
        return [
            SyncPair(0, i, float(radar_ts[0]), float(lidar_ts[i]), float(delta_ms[i]))
            for i in range(lidar_ts.size)
            if abs(delta_ms[i]) <= max_delta_ms
        ]

    idx = np.searchsorted(radar_ts, lidar_ts, side="left")
    idx = np.clip(idx, 1, radar_ts.size - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(lidar_ts - radar_ts[left]) > np.abs(radar_ts[right] - lidar_ts)
    nearest_idx = np.where(choose_right, right, left)
    nearest_ts = radar_ts[nearest_idx]
    delta_ms = (nearest_ts - lidar_ts) * 1e3

    out: List[SyncPair] = []
    for lidar_idx, (ridx, rt, lt, dt) in enumerate(
        zip(nearest_idx, nearest_ts, lidar_ts, delta_ms)
    ):
        if abs(float(dt)) <= max_delta_ms:
            out.append(
                SyncPair(
                    radar_idx=int(ridx),
                    lidar_idx=int(lidar_idx),
                    radar_t=float(rt),
                    lidar_t=float(lt),
                    delta_ms=float(dt),
                )
            )
    return out


def write_pairs_csv(pairs: Iterable[SyncPair], csv_path: str | Path) -> None:
    """Write synchronization pairs as CSV."""
    fieldnames = ["radar_idx", "lidar_idx", "radar_t", "lidar_t", "delta_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in pairs:
            writer.writerow(
                {
                    "radar_idx": p.radar_idx,
                    "lidar_idx": p.lidar_idx,
                    "radar_t": f"{p.radar_t:.9f}",
                    "lidar_t": f"{p.lidar_t:.9f}",
                    "delta_ms": f"{p.delta_ms:.3f}",
                }
            )


def nearest_neighbor_delta_ms(
    radar_ts: Sequence[float],
    lidar_ts: Sequence[float],
    lidar_to_radar_offset_s: float = 0.0,
    lidar_time_scale: float = 1.0,
) -> np.ndarray:
    """Return per-lidar-frame nearest-neighbor timing error in ms (no threshold)."""
    radar_ts = np.asarray(radar_ts, dtype=np.float64)
    lidar_ts = adjust_lidar_timestamps(
        lidar_ts,
        lidar_time_scale=lidar_time_scale,
        lidar_to_radar_offset_s=lidar_to_radar_offset_s,
    )
    if radar_ts.size == 0 or lidar_ts.size == 0:
        return np.empty((0,), dtype=np.float64)
    if radar_ts.size == 1:
        return (radar_ts[0] - lidar_ts) * 1e3

    idx = np.searchsorted(radar_ts, lidar_ts, side="left")
    idx = np.clip(idx, 1, radar_ts.size - 1)
    left = idx - 1
    right = idx
    choose_right = np.abs(lidar_ts - radar_ts[left]) > np.abs(radar_ts[right] - lidar_ts)
    nearest_idx = np.where(choose_right, right, left)
    return (radar_ts[nearest_idx] - lidar_ts) * 1e3


def write_summary_json(
    json_path: str | Path,
    *,
    radar_count: int,
    lidar_count: int,
    matched_count: int,
    estimated_offset_s: float,
    median_abs_delta_ms: Optional[float],
    p95_abs_delta_ms: Optional[float],
    radar_t_start: Optional[float] = None,
    radar_t_end: Optional[float] = None,
    lidar_t_start: Optional[float] = None,
    lidar_t_end: Optional[float] = None,
    offset_at_search_boundary: bool = False,
    best_unmatched_median_abs_delta_ms: Optional[float] = None,
    radar_packets_per_frame: Optional[int] = None,
    radar_adc_packets_per_frame: Optional[int] = None,
    timeline_overlap_s: Optional[float] = None,
    estimated_lidar_time_scale: float = 1.0,
    estimated_skew_ppm: float = 0.0,
    fit_median_error_s: Optional[float] = None,
    skew_at_search_boundary: bool = False,
    timing_drift: Optional[Dict[str, Optional[float]]] = None,
) -> None:
    """Write compact sync summary."""
    payload = {
        "radar_frames": int(radar_count),
        "lidar_frames": int(lidar_count),
        "matched_pairs": int(matched_count),
        "radar_packets_per_frame": radar_packets_per_frame,
        "radar_adc_packets_per_frame": radar_adc_packets_per_frame,
        "timeline_overlap_s": timeline_overlap_s,
        "estimated_lidar_to_radar_offset_s": float(estimated_offset_s),
        "estimated_lidar_time_scale": float(estimated_lidar_time_scale),
        "estimated_skew_ppm": float(estimated_skew_ppm),
        "fit_median_error_s": fit_median_error_s,
        "median_abs_delta_ms": median_abs_delta_ms,
        "p95_abs_delta_ms": p95_abs_delta_ms,
        "radar_t_start": radar_t_start,
        "radar_t_end": radar_t_end,
        "lidar_t_start": lidar_t_start,
        "lidar_t_end": lidar_t_end,
        "offset_at_search_boundary": bool(offset_at_search_boundary),
        "skew_at_search_boundary": bool(skew_at_search_boundary),
        "best_unmatched_median_abs_delta_ms": best_unmatched_median_abs_delta_ms,
    }
    if timing_drift:
        payload["timing_drift"] = timing_drift
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
