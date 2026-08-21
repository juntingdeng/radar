"""Visualize radar/lidar synchronization quality side by side."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

# This diagnostic lives in code/sync/dev/; pipeline modules live in code/sync/src/.
import sys

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


def _load_pairs_csv(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    radar_idx = []
    lidar_idx = []
    delta_ms = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            radar_idx.append(int(row["radar_idx"]))
            lidar_idx.append(int(row["lidar_idx"]))
            delta_ms.append(float(row["delta_ms"]))
    return (
        np.asarray(radar_idx, dtype=np.int64),
        np.asarray(lidar_idx, dtype=np.int64),
        np.asarray(delta_ms, dtype=np.float64),
    )


def _make_video_from_pngs(
    png_dir: Path,
    out_video: Path,
    fps: int,
    pattern: str = "pair_%06d_*.png",
) -> None:
    """Create mp4 video from exported per-pair PNGs using ffmpeg."""
    out_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(png_dir / pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_video),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not installed or not in PATH. Install ffmpeg to enable video export."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed while creating video:\n{stderr}") from exc


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize radar/lidar synchronization results.")
    p.add_argument(
        "--cfg_file",
        default="./code/mmWaveStudio/server.lua",
        help="Radar lua config; used to derive packets_per_frame (must match sync run).",
    )
    p.add_argument("--sync_csv", default=None, help="CSV from sync_radar_lidar.py (or --dataset).")
    p.add_argument("--radar_h5", default=None, help="Radar .h5 (or --dataset).")
    p.add_argument("--lidar_pcap", default=None, help="Lidar .pcap (or --dataset).")
    p.add_argument("--lidar_metadata", default=None, help="Optional Ouster metadata .json.")
    p.add_argument(
        "--radar_packets_per_frame",
        type=int,
        default=None,
        help="Override packets/frame; default: from --cfg_file via radarConfig.",
    )
    p.add_argument("--lidar_packets_per_frame", type=int, default=128)
    p.add_argument("--lidar_udp_port", type=int, default=7502)
    p.add_argument(
        "--estimated_offset_s",
        type=float,
        default=0.0,
        help="Offset applied during matching (from sync summary json), lidar->radar.",
    )
    p.add_argument(
        "--out_png",
        default="./code/sync/sync_visualization.png",
        help="Path to save visualization image.",
    )
    p.add_argument(
        "--export_all_pairs_dir",
        default=None,
        help="If set, export one zoomed PNG per matched pair to this directory.",
    )
    p.add_argument(
        "--export_max_pairs",
        type=int,
        default=-1,
        help="Maximum number of pair PNGs to export (-1 means all).",
    )
    p.add_argument(
        "--zoom_window_s",
        type=float,
        default=0.2,
        help="Half window (seconds) for per-pair zoomed timeline export.",
    )
    p.add_argument(
        "--out_video",
        default=None,
        help="If set with --export_all_pairs_dir, create an MP4 from exported pair PNGs.",
    )
    p.add_argument(
        "--video_fps",
        type=int,
        default=12,
        help="FPS for output video when --out_video is enabled.",
    )
    p.add_argument(
        "--no_show",
        action="store_true",
        help="Only save figure, do not open an interactive window.",
    )
    add_dataset_arguments(p)
    return p


def _packets_per_frame_from_cfg(cfg_file: str, override: int | None) -> int:
    if override is not None:
        return override
    import sys
    from pathlib import Path

    # dev/ is code/sync/dev/, so code/ is parents[2].
    code_root = Path(__file__).resolve().parents[2]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from utils.parse_config import radarConfig

    radar = radarConfig()
    radar.parse_radar(cfg_file=cfg_file)
    return int(radar.packets_per_frame)


def main() -> None:
    args = apply_dataset_config(
        build_argparser().parse_args(),
        required=("radar_h5", "lidar_pcap", "sync_csv"),
    )

    radar_idx, lidar_idx, delta_ms = _load_pairs_csv(args.sync_csv)
    if radar_idx.size == 0:
        raise RuntimeError("No matched pairs in CSV. Check sync parameters first.")

    radar_ppf = _packets_per_frame_from_cfg(args.cfg_file, args.radar_packets_per_frame)
    print(f"Using radar_packets_per_frame={radar_ppf}")

    radar_packet_t = read_radar_packet_timestamps(args.radar_h5)
    radar_frame_t = radar_packets_to_frames(radar_packet_t, packets_per_frame=radar_ppf)

    if radar_idx.max() >= radar_frame_t.size:
        raise RuntimeError(
            f"CSV radar_idx max={radar_idx.max()} but only {radar_frame_t.size} radar frames. "
            "Re-run sync with the same --cfg_file / packets_per_frame."
        )

    lidar_frame_t = try_read_ouster_scan_timestamps(args.lidar_pcap, args.lidar_metadata)
    using_sdk = lidar_frame_t is not None
    if lidar_frame_t is None:
        udp_port = None if args.lidar_udp_port < 0 else args.lidar_udp_port
        lidar_packet_t = read_lidar_packet_timestamps_from_pcap(args.lidar_pcap, udp_port=udp_port)
        lidar_frame_t = lidar_packets_to_frames(
            lidar_packet_t, packets_per_frame=args.lidar_packets_per_frame
        )

    lidar_frame_t = lidar_frame_t + float(args.estimated_offset_s)

    radar_rel = radar_frame_t - radar_frame_t[0]
    lidar_rel = lidar_frame_t - lidar_frame_t[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Side-by-side timeline view.
    axes[0, 0].plot(np.arange(radar_rel.size), radar_rel, ".", ms=2, label="Radar frames")
    axes[0, 0].set_title("Radar Timeline")
    axes[0, 0].set_xlabel("Radar frame index")
    axes[0, 0].set_ylabel("Time since start [s]")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(np.arange(lidar_rel.size), lidar_rel, ".", ms=2, color="tab:orange", label="Lidar frames")
    axes[0, 1].set_title(f"Lidar Timeline ({'Ouster SDK' if using_sdk else 'PCAP fallback'})")
    axes[0, 1].set_xlabel("Lidar frame index")
    axes[0, 1].set_ylabel("Time since start [s]")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(loc="best")

    # Pair mapping plot.
    axes[1, 0].plot(lidar_idx, radar_idx, ".", ms=3, color="tab:green")
    axes[1, 0].set_title("Matched Index Pairs")
    axes[1, 0].set_xlabel("Lidar frame index")
    axes[1, 0].set_ylabel("Matched radar frame index")
    axes[1, 0].grid(True, alpha=0.25)

    # Timing error plot + histogram.
    pair_order = np.arange(delta_ms.size)
    axes[1, 1].plot(pair_order, delta_ms, ".", ms=2, label="delta_ms")
    axes[1, 1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1, 1].set_title("Per-Pair Time Error (radar_t - lidar_t)")
    axes[1, 1].set_xlabel("Matched pair id")
    axes[1, 1].set_ylabel("Delta [ms]")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(loc="best")

    med = float(np.median(np.abs(delta_ms)))
    p95 = float(np.percentile(np.abs(delta_ms), 95))
    fig.suptitle(
        f"Radar/Lidar Sync Inspection | matches={delta_ms.size}, median|delta|={med:.2f} ms, p95|delta|={p95:.2f} ms",
        fontsize=11,
    )
    fig.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"Wrote: {out_png}")

    if args.export_all_pairs_dir:
        export_dir = Path(args.export_all_pairs_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        max_pairs = delta_ms.size if args.export_max_pairs < 0 else min(
            delta_ms.size, args.export_max_pairs
        )
        half_win = float(args.zoom_window_s)

        for i in range(max_pairs):
            ridx = int(radar_idx[i])
            lidx = int(lidar_idx[i])
            dt = float(delta_ms[i])

            rt = float(radar_frame_t[ridx])
            lt = float(lidar_frame_t[lidx])

            start_t = min(rt, lt) - half_win
            end_t = max(rt, lt) + half_win

            rmask = (radar_frame_t >= start_t) & (radar_frame_t <= end_t)
            lmask = (lidar_frame_t >= start_t) & (lidar_frame_t <= end_t)

            fig_i, ax_i = plt.subplots(1, 2, figsize=(12, 4))

            # Left: local timeline around this pair in absolute time.
            if np.any(rmask):
                ax_i[0].plot(
                    radar_frame_t[rmask],
                    np.zeros(np.count_nonzero(rmask)),
                    ".",
                    ms=3,
                    label="Radar frames",
                )
            if np.any(lmask):
                ax_i[0].plot(
                    lidar_frame_t[lmask],
                    np.ones(np.count_nonzero(lmask)),
                    ".",
                    ms=3,
                    color="tab:orange",
                    label="Lidar frames",
                )
            ax_i[0].plot(rt, 0.0, "o", color="tab:blue", ms=8)
            ax_i[0].plot(lt, 1.0, "o", color="tab:red", ms=8)
            ax_i[0].set_yticks([0.0, 1.0])
            ax_i[0].set_yticklabels(["Radar", "Lidar"])
            ax_i[0].set_xlim(start_t, end_t)
            ax_i[0].set_xlabel("Time [s]")
            ax_i[0].set_title("Local Timeline")
            ax_i[0].grid(True, alpha=0.25)
            ax_i[0].legend(loc="best")

            # Right: this pair's metadata for quick inspection.
            ax_i[1].axis("off")
            text = (
                f"Pair ID: {i}\n"
                f"Radar idx: {ridx}\n"
                f"Lidar idx: {lidx}\n"
                f"Radar t: {rt:.6f} s\n"
                f"Lidar t: {lt:.6f} s\n"
                f"delta_ms (radar-lidar): {dt:.3f} ms\n"
            )
            ax_i[1].text(0.05, 0.95, text, va="top", fontsize=11)
            fig_i.suptitle("Synchronized Pair Inspection", fontsize=11)
            fig_i.tight_layout()

            out_i = export_dir / f"pair_{i:06d}_r{ridx:06d}_l{lidx:06d}.png"
            fig_i.savefig(out_i, dpi=140)
            plt.close(fig_i)

        print(f"Wrote {max_pairs} per-pair PNGs to: {export_dir}")
        if args.out_video:
            out_video = Path(args.out_video)
            _make_video_from_pngs(export_dir, out_video, fps=max(1, args.video_fps))
            print(f"Wrote video: {out_video}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
