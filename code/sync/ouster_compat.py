"""Ouster SDK compatibility helpers (0.16.x and legacy pre-0.16 imports)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np


def _import_legacy() -> Tuple[Any, Any]:
    from ouster import client, pcap  # type: ignore

    return client, pcap


def _import_modern() -> Tuple[Any, Any]:
    from ouster.sdk import core, pcap  # type: ignore

    return core, pcap


def get_ouster_api() -> str:
    """Return 'modern' (>=0.16) or 'legacy' (<0.16)."""
    try:
        from ouster.sdk import core  # noqa: F401

        return "modern"
    except ImportError:
        pass
    try:
        from ouster import client  # noqa: F401

        return "legacy"
    except ImportError as exc:
        raise ImportError(
            "Ouster SDK not found. Install with: pip install ouster-sdk"
        ) from exc


def open_pcap_scan_source(
    pcap_path: str | Path,
    metadata_json_path: Optional[str | Path] = None,
    *,
    index: bool = True,
) -> Any:
    """Open a PCAP scan source (API-compatible wrapper).

    index=True is required for random access via source[scan_idx] (Ouster SDK 0.16+).
    """
    pcap_path = str(pcap_path)
    api = get_ouster_api()

    if api == "modern":
        from ouster.sdk import pcap  # type: ignore
        from ouster.sdk.util.metadata import resolve_metadata  # type: ignore

        meta_kw: dict = {"index": bool(index)}
        if metadata_json_path is not None:
            meta_path = str(metadata_json_path)
        else:
            resolved = resolve_metadata(pcap_path)
            meta_path = resolved
        if meta_path:
            meta_kw["meta"] = [meta_path]
        return pcap.PcapScanSource(pcap_path, **meta_kw)

    client, pcap = _import_legacy()
    if metadata_json_path is None:
        raise ValueError("metadata_json_path is required for legacy ouster-sdk (<0.16).")
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = client.SensorInfo(f.read())
    return pcap.Pcap(pcap_path, metadata)


def sensor_info_from_source(source: Any) -> Any:
    api = get_ouster_api()
    if api == "modern":
        return source.sensor_info[0]
    return source.metadata


def iter_scans(source: Any) -> Iterator[Any]:
    """Yield LidarScan objects from a PCAP source."""
    api = get_ouster_api()
    if api == "modern":
        for scan_set in source:
            if scan_set is None:
                continue
            # LidarScanSet: one scan per sensor in the set
            if hasattr(scan_set, "__len__") and len(scan_set) > 0:
                scan = scan_set[0]
            else:
                scan = scan_set
            if scan is not None:
                yield scan
        return

    from ouster import client  # type: ignore

    for scan in client.Scans(source):
        yield scan


def scan_timestamp_min_s(scan: Any) -> Optional[float]:
    """Earliest valid scan timestamp in seconds."""
    api = get_ouster_api()
    if api == "modern":
        from ouster.sdk import core  # type: ignore

        if hasattr(scan, "get_first_valid_packet_timestamp"):
            ts = int(scan.get_first_valid_packet_timestamp())
            if ts > 0:
                return ts * 1e-9
        ts_arr = scan.timestamp
        import numpy as np

        ts_arr = np.asarray(ts_arr, dtype=np.int64)
        ts_arr = ts_arr[ts_arr > 0]
        if ts_arr.size:
            return float(ts_arr.min()) * 1e-9
        return None

    import numpy as np

    ts = np.asarray(scan.timestamp, dtype=np.int64)
    ts = ts[ts > 0]
    if ts.size:
        return float(ts.min()) * 1e-9
    return None


def scan_to_xyz(metadata: Any, scan: Any):
    """Return HxWx3 XYZ point cloud for a scan (meters, sensor frame)."""
    api = get_ouster_api()
    if api == "modern":
        from ouster.sdk import core  # type: ignore

        lut = core.XYZLut(metadata)
        # SDK 0.16+: project staggered RANGE field (not the scan object).
        ranges = scan.field(core.ChanField.RANGE)
        return np.asarray(lut(ranges))

    from ouster import client  # type: ignore

    return client.XYZLut(metadata)(scan)


def _chan_field_range() -> Any:
    api = get_ouster_api()
    if api == "modern":
        from ouster.sdk import core  # type: ignore

        return core.ChanField.RANGE
    from ouster import client  # type: ignore

    return client.ChanField.RANGE


def _chan_field_reflectivity() -> Any:
    api = get_ouster_api()
    if api == "modern":
        from ouster.sdk import core  # type: ignore

        return core.ChanField.REFLECTIVITY
    from ouster import client  # type: ignore

    return client.ChanField.REFLECTIVITY


def scan_range_flat(scan: Any) -> np.ndarray:
    """Return HxW range (mm) in native staggered layout."""
    return np.asarray(scan.field(_chan_field_range()), dtype=np.float64)


def scan_reflectivity_staggered(scan: Any) -> np.ndarray:
    """Return HxW reflectivity in native staggered layout (matches scan_to_xyz)."""
    return np.asarray(scan.field(_chan_field_reflectivity()), dtype=np.float32)


def scan_reflectivity(scan: Any, metadata: Any = None):
    """Return HxW reflectivity (destaggered when SDK supports it)."""
    api = get_ouster_api()
    if api == "modern":
        from ouster.sdk import core  # type: ignore

        refl = np.asarray(scan.field(core.ChanField.REFLECTIVITY))
        if metadata is not None and hasattr(core, "destagger"):
            refl = np.asarray(core.destagger(metadata, refl))
        return refl
    from ouster import client  # type: ignore

    refl = np.asarray(scan.field(client.ChanField.REFLECTIVITY))
    if metadata is not None:
        refl = np.asarray(client.destagger(metadata, refl))
    return refl


def close_source(source: Any) -> None:
    if hasattr(source, "close"):
        source.close()


def _scan_from_scan_set(scan_set: Any) -> Any:
    if scan_set is None:
        raise IndexError("Empty scan set.")
    if hasattr(scan_set, "__len__") and len(scan_set) > 0:
        return scan_set[0]
    return scan_set


def scan_range_checksum(scan: Any) -> int:
    """Cheap fingerprint to verify scan-to-scan variation."""
    api = get_ouster_api()
    if api == "modern":
        from ouster.sdk import core  # type: ignore

        ranges = scan.field(core.ChanField.RANGE)
    else:
        from ouster import client  # type: ignore

        ranges = scan.field(client.ChanField.RANGE)
    return int(np.sum(np.asarray(ranges, dtype=np.uint64)))


def scan_fingerprint(scan: Any) -> dict:
    """Raw-scan identity for diagnosing PCAP read vs BEV post-processing."""
    api = get_ouster_api()
    if api == "modern":
        from ouster.sdk import core  # type: ignore

        ranges = np.asarray(scan.field(core.ChanField.RANGE))
    else:
        from ouster import client  # type: ignore

        ranges = np.asarray(scan.field(client.ChanField.RANGE))
    return {
        "range_checksum": int(np.sum(ranges.astype(np.uint64))),
        "range_nonzero": int(np.count_nonzero(ranges)),
        "timestamp_s": scan_timestamp_min_s(scan),
    }


def get_scan_at_index(source: Any, scan_idx: int) -> Any:
    """Load scan by sync-order index (same ordering as iter_scans / sync step)."""
    scan_idx = int(scan_idx)
    api = get_ouster_api()
    if api == "modern":
        # Use sequential nth(), not source[scan_idx]: indexed __getitem__ can
        # return stale/repeated scans on some SDK builds; sync uses iter order.
        from more_itertools import nth

        scan_set = nth(source, scan_idx)
        if scan_set is None:
            raise IndexError(f"Scan index {scan_idx} not in PCAP.")
        return _scan_from_scan_set(scan_set)

    from more_itertools import nth

    scan = nth(iter_scans(source), scan_idx)
    if scan is None:
        raise IndexError(f"Scan index {scan_idx} not in PCAP.")
    return scan


def scan_source_length(source: Any) -> Optional[int]:
    if hasattr(source, "__len__"):
        try:
            return int(len(source))
        except Exception:
            return None
    return None
