"""Extract radar/lidar timestamps for quick sanity checks."""

from __future__ import annotations

import argparse

import numpy as np

# This diagnostic lives in code/sync/dev/; pipeline modules live in code/sync/src/.
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lib.dataset_config import add_dataset_arguments, apply_dataset_config
from lib.sync_utils import (
    lidar_packets_to_frames,
    radar_packets_to_frames,
    read_lidar_packet_timestamps_from_pcap,
    read_radar_packet_timestamps,
    try_read_ouster_scan_timestamps,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract timestamp arrays from radar/lidar captures.")
    p.add_argument("--radar_h5", default=None)
    p.add_argument("--lidar_pcap", default=None)
    p.add_argument("--lidar_metadata", default=None)
    p.add_argument("--radar_packets_per_frame", type=int, default=128)
    p.add_argument("--lidar_packets_per_frame", type=int, default=128)
    p.add_argument("--lidar_udp_port", type=int, default=7502)
    p.add_argument(
        "--output_npz",
        default="./code/sync/timestamps.npz",
        help="Output npz containing packet/frame timestamps.",
    )
    add_dataset_arguments(p)
    return p


def main() -> None:
    args = apply_dataset_config(
        build_argparser().parse_args(),
        required=("radar_h5", "lidar_pcap"),
    )

    radar_packet_ts = read_radar_packet_timestamps(args.radar_h5)
    radar_frame_ts = radar_packets_to_frames(
        radar_packet_ts, packets_per_frame=args.radar_packets_per_frame
    )

    lidar_frame_ts_sdk = try_read_ouster_scan_timestamps(args.lidar_pcap, args.lidar_metadata)
    using_sdk = lidar_frame_ts_sdk is not None
    if using_sdk:
        lidar_packet_ts = np.empty((0,), dtype=np.float64)
        lidar_frame_ts = lidar_frame_ts_sdk
    else:
        udp_port = None if args.lidar_udp_port < 0 else args.lidar_udp_port
        lidar_packet_ts = read_lidar_packet_timestamps_from_pcap(
            args.lidar_pcap, udp_port=udp_port
        )
        lidar_frame_ts = lidar_packets_to_frames(
            lidar_packet_ts, packets_per_frame=args.lidar_packets_per_frame
        )

    np.savez(
        args.output_npz,
        radar_packet_t=radar_packet_ts,
        radar_frame_t=radar_frame_ts,
        lidar_packet_t=lidar_packet_ts,
        lidar_frame_t=lidar_frame_ts,
        lidar_from_sdk=np.array([int(using_sdk)], dtype=np.int32),
    )

    print(f"Radar packets: {radar_packet_ts.size}")
    print(f"Radar frames: {radar_frame_ts.size}")
    print(f"Lidar packets: {lidar_packet_ts.size}")
    print(f"Lidar frames: {lidar_frame_ts.size}")
    print(f"Lidar timestamps source: {'Ouster SDK' if using_sdk else 'PCAP fallback'}")
    print(f"Wrote: {args.output_npz}")


if __name__ == "__main__":
    main()
