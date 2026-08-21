"""Synchronize TI radar (.h5) and Ouster lidar (.pcap) in time.

Outputs:
1) CSV of matched frame indices and timestamps
2) JSON summary with offset/error stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lib.dataset_config import add_dataset_arguments, apply_dataset_config
from lib.sync_utils import (
    compute_pair_timing_drift,
    fit_time_affine,
    fit_time_offset,
    lidar_packets_to_frames,
    nearest_neighbor_delta_ms,
    nearest_neighbor_pairs,
    print_timeline_overlap_report,
    radar_packets_to_frames,
    read_lidar_packet_timestamps_from_pcap,
    read_radar_packet_timestamps,
    try_read_ouster_scan_timestamps,
    write_pairs_csv,
    write_summary_json,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Radar/lidar timestamp synchronization.")
    p.add_argument("--radar_h5", default=None, help="Radar .h5 (or set via --dataset).")
    p.add_argument("--lidar_pcap", default=None, help="Lidar .pcap (or set via --dataset).")
    p.add_argument(
        "--lidar_metadata",
        default=None,
        help="Optional Ouster metadata .json. If provided and SDK is installed, use true scan timestamps.",
    )
    p.add_argument(
        "--cfg_file",
        default="./code/mmWaveStudio/server.lua",
        help="Radar lua config; derives packets_per_frame when --radar_packets_per_frame is unset.",
    )
    p.add_argument(
        "--radar_packets_per_frame",
        type=int,
        default=None,
        help="Packets per sync-time block for timestamp matching (default: adc_ppf/6 from --cfg_file, e.g. 45 not 270).",
    )
    p.add_argument(
        "--lidar_packets_per_frame",
        type=int,
        default=128,
        help="Number of lidar UDP packets per frame when not using Ouster SDK.",
    )
    p.add_argument(
        "--lidar_udp_port",
        type=int,
        default=7502,
        help="Lidar UDP port in pcap (OS0 default 7502). Set -1 to disable port filter.",
    )
    p.add_argument(
        "--fit_offset",
        action="store_true",
        help="Estimate constant lidar->radar clock offset before matching (scale=1).",
    )
    p.add_argument(
        "--fit_skew",
        action="store_true",
        help="Estimate affine lidar->radar map: radar ~= scale*lidar + offset (includes --fit_offset).",
    )
    p.add_argument(
        "--offset_search_s",
        type=float,
        default=5.0,
        help="Max absolute offset (seconds) searched when --fit_offset / --fit_skew is on.",
    )
    p.add_argument(
        "--skew_search_ppm",
        type=float,
        default=200.0,
        help="Max absolute clock-rate error (ppm) searched when --fit_skew is on.",
    )
    p.add_argument(
        "--max_delta_ms",
        type=float,
        default=100.0,
        help="Maximum allowed matching error in milliseconds.",
    )
    p.add_argument(
        "--output_csv",
        default="./code/sync/sync_pairs.csv",
        help="Output CSV path for matched pairs.",
    )
    p.add_argument(
        "--output_json",
        default="./code/sync/sync_summary.json",
        help="Output JSON summary path.",
    )
    add_dataset_arguments(p)
    return p


def _radar_sync_packets_per_frame(args) -> tuple[int, int]:
    """Return (sync_ppf for timestamp matching, adc_ppf from lua cfg)."""
    import sys
    from pathlib import Path

    code_root = Path(__file__).resolve().parents[2]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from utils.parse_config import radarConfig

    radar = radarConfig()
    radar.parse_radar(cfg_file=args.cfg_file)
    adc_ppf = int(radar.packets_per_frame)
    if args.radar_packets_per_frame is not None:
        return int(args.radar_packets_per_frame), adc_ppf
    # Sync-time blocks are coarser than one ADC frame (mobile cfg: 270 -> 45).
    sync_ppf = max(1, adc_ppf // 6)
    return sync_ppf, adc_ppf


def main() -> None:
    args = apply_dataset_config(
        build_argparser().parse_args(),
        required=("radar_h5", "lidar_pcap"),
    )
    radar_ppf, adc_ppf = _radar_sync_packets_per_frame(args)
    print(
        f"Using radar_packets_per_frame={radar_ppf} for sync-time matching "
        f"(adc_packets_per_frame={adc_ppf} from cfg)"
    )

    print(f"Reading radar packet timestamps from {args.radar_h5} ...")
    radar_packet_ts = read_radar_packet_timestamps(args.radar_h5)
    radar_frame_ts = radar_packets_to_frames(radar_packet_ts, packets_per_frame=radar_ppf)
    print(f"Radar timestamps loaded: {radar_packet_ts.size} packets -> {radar_frame_ts.size} sync frames")

    lidar_frame_ts = try_read_ouster_scan_timestamps(args.lidar_pcap, args.lidar_metadata)
    using_sdk = lidar_frame_ts is not None
    if lidar_frame_ts is None:
        udp_port = None if args.lidar_udp_port < 0 else args.lidar_udp_port
        lidar_packet_ts = read_lidar_packet_timestamps_from_pcap(
            args.lidar_pcap, udp_port=udp_port
        )
        lidar_frame_ts = lidar_packets_to_frames(
            lidar_packet_ts, packets_per_frame=args.lidar_packets_per_frame
        )

    if radar_frame_ts.size == 0:
        raise RuntimeError("No radar frames extracted. Check packets_per_frame and input file.")
    if lidar_frame_ts.size == 0:
        raise RuntimeError("No lidar frames extracted. Check pcap, port, and frame arguments.")

    overlap_s = print_timeline_overlap_report(radar_frame_ts, lidar_frame_ts)

    offset_s = 0.0
    scale = 1.0
    fit_median_error_s = None
    offset_at_boundary = False
    skew_at_boundary = False

    if args.fit_skew:
        if overlap_s <= 0:
            print(
                "Skipping --fit_skew because timelines do not overlap "
                f"(gap ~{max(radar_frame_ts[0], float(lidar_frame_ts[0])) - min(radar_frame_ts[-1], float(lidar_frame_ts[-1])):.0f} s)."
            )
        else:
            scale, offset_s, fit_median_error_s = fit_time_affine(
                radar_frame_ts,
                lidar_frame_ts,
                max_abs_shift_s=args.offset_search_s,
                max_skew_ppm=args.skew_search_ppm,
            )
            if abs(offset_s) >= args.offset_search_s - 1e-9:
                offset_at_boundary = True
                print(
                    "WARNING: estimated offset hit the search boundary "
                    f"(+/- {args.offset_search_s} s). Increase --offset_search_s."
                )
            skew_ppm = (scale - 1.0) * 1e6
            if abs(skew_ppm) >= args.skew_search_ppm - 1e-6:
                skew_at_boundary = True
                print(
                    "WARNING: estimated skew hit the search boundary "
                    f"(+/- {args.skew_search_ppm} ppm). Increase --skew_search_ppm."
                )

            offset_only = fit_time_offset(
                radar_frame_ts, lidar_frame_ts, max_abs_shift_s=args.offset_search_s
            )
            print(
                "Affine fit: "
                f"scale={scale:.9f} ({skew_ppm:+.3f} ppm), "
                f"offset={offset_s:.6f} s, median NN error={fit_median_error_s * 1e3:.2f} ms"
            )
            print(f"Offset-only comparison: offset={offset_only:.6f} s (scale fixed at 1.0)")
            overlap_s = print_timeline_overlap_report(
                radar_frame_ts,
                lidar_frame_ts,
                lidar_time_scale=scale,
                lidar_to_radar_offset_s=offset_s,
            )
    elif args.fit_offset:
        if overlap_s <= 0:
            print(
                "Skipping --fit_offset because timelines do not overlap "
                f"(gap ~{abs(overlap_s):.0f} s)."
            )
        else:
            offset_s = fit_time_offset(
                radar_frame_ts, lidar_frame_ts, max_abs_shift_s=args.offset_search_s
            )
            if abs(offset_s) >= args.offset_search_s - 1e-9:
                offset_at_boundary = True
                print(
                    "WARNING: estimated offset hit the search boundary "
                    f"(+/- {args.offset_search_s} s). Increase --offset_search_s."
                )
            overlap_s = print_timeline_overlap_report(
                radar_frame_ts, lidar_frame_ts, lidar_to_radar_offset_s=offset_s
            )

    pairs = nearest_neighbor_pairs(
        radar_frame_ts,
        lidar_frame_ts,
        lidar_to_radar_offset_s=offset_s,
        lidar_time_scale=scale,
        max_delta_ms=args.max_delta_ms,
    )

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    write_pairs_csv(pairs, out_csv)

    all_delta_ms = nearest_neighbor_delta_ms(
        radar_frame_ts,
        lidar_frame_ts,
        lidar_to_radar_offset_s=offset_s,
        lidar_time_scale=scale,
    )
    timing_drift = compute_pair_timing_drift(pairs, lidar_frame_ts)
    best_unmatched_med = (
        float(np.median(np.abs(all_delta_ms))) if all_delta_ms.size else None
    )

    if pairs:
        abs_delta = np.abs(np.asarray([p.delta_ms for p in pairs], dtype=np.float64))
        med: float | None = float(np.median(abs_delta))
        p95: float | None = float(np.percentile(abs_delta, 95))
    else:
        med = None
        p95 = None
        print(
            f"No pairs within --max_delta_ms={args.max_delta_ms}. "
            f"Median nearest-neighbor error (all frames): {best_unmatched_med:.1f} ms"
            if best_unmatched_med is not None
            else "No pairs matched."
        )

    skew_ppm = (scale - 1.0) * 1e6
    write_summary_json(
        out_json,
        radar_count=radar_frame_ts.size,
        lidar_count=lidar_frame_ts.size,
        matched_count=len(pairs),
        estimated_offset_s=offset_s,
        estimated_lidar_time_scale=scale,
        estimated_skew_ppm=skew_ppm,
        fit_median_error_s=fit_median_error_s,
        median_abs_delta_ms=med,
        p95_abs_delta_ms=p95,
        radar_t_start=float(radar_frame_ts[0]),
        radar_t_end=float(radar_frame_ts[-1]),
        lidar_t_start=float(lidar_frame_ts[0]),
        lidar_t_end=float(lidar_frame_ts[-1]),
        offset_at_search_boundary=offset_at_boundary,
        skew_at_search_boundary=skew_at_boundary,
        best_unmatched_median_abs_delta_ms=best_unmatched_med,
        radar_packets_per_frame=radar_ppf,
        radar_adc_packets_per_frame=adc_ppf,
        timeline_overlap_s=overlap_s,
        timing_drift=timing_drift,
    )

    print(f"Radar frames: {radar_frame_ts.size}")
    print(f"Lidar frames: {lidar_frame_ts.size} ({'Ouster SDK' if using_sdk else 'PCAP fallback'})")
    print(f"Matched pairs: {len(pairs)}")
    print(
        "Time map: "
        f"radar ~= {scale:.9f} * lidar + {offset_s:.6f} s "
        f"({skew_ppm:+.3f} ppm skew)"
    )
    if timing_drift.get("delta_ms_drift_slope_ms_per_s") is not None:
        print(
            "Post-match drift diagnostic: "
            f"slope={timing_drift['delta_ms_drift_slope_ms_per_s']:.6f} ms/s, "
            f"total={timing_drift['delta_ms_drift_total_ms']:.2f} ms over recording, "
            f"implied skew from pairs={timing_drift['implied_skew_ppm_from_pairs']:+.3f} ppm"
        )
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
