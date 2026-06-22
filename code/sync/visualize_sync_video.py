"""Render synchronized radar range-azimuth + lidar BEV side-by-side as MP4."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

_SYNC_DIR = Path(__file__).resolve().parent
if str(_SYNC_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNC_DIR))

_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from utils.parse_config import radarConfig  # noqa: E402

from dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402
from sync_utils import read_radar_packet_timestamps  # noqa: E402
from sync_viz_data import (  # noqa: E402
    CANONICAL_SCENE_FORWARD_RANGE,
    CANONICAL_SCENE_LATERAL_RANGE,
    DEFAULT_SCENE_FORWARD_RANGE,
    DEFAULT_SCENE_GRID_RES,
    DEFAULT_SCENE_LATERAL_RANGE,
    LidarScanReader,
    VizFrameCache,
    canonical_scene_bounds,
    clamp_display_bounds,
    compute_range_azimuth,
    crop_topdown_panel,
    infer_sync_packets_per_frame,
    lidar_panel_axis,
    lidar_panel_color_limits,
    lidar_panel_for_imshow,
    lidar_view_is_topdown,
    load_radar_adc_frame,
    load_sync_pairs_full,
    print_lidar_diagnosis,
    radar_adc_bounds,
    radar_panel_axis,
    radar_samples_per_frame,
    radar_t_to_adc_frame,
    range_azimuth_to_bev,
    scene_bounds_from_args,
    sync_radar_idx_to_adc_frame,
    use_radar_bev_panel,
)


def _load_sync_packets_per_frame(args, radar_h5: str) -> int:
    if args.sync_packets_per_frame is not None:
        return int(args.sync_packets_per_frame)
    if args.sync_summary:
        with open(args.sync_summary, "r", encoding="utf-8") as f:
            summary = json.load(f)
        ppf = summary.get("radar_packets_per_frame")
        if ppf is not None:
            return int(ppf)
        radar_frames = int(summary.get("radar_frames", 0))
        if radar_frames > 0:
            return infer_sync_packets_per_frame(radar_h5, radar_frames)
    raise ValueError(
        "Need --sync_packets_per_frame or --sync_summary with radar_packets_per_frame."
    )


def _resolve_cfg_file(cfg_file: str) -> str:
    candidates = [
        Path(cfg_file),
        _CODE_ROOT / "mmWaveStudio" / "server.lua",
        Path.cwd() / "code" / "mmWaveStudio" / "server.lua",
    ]
    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    raise FileNotFoundError(
        f"Radar cfg not found: {cfg_file!r}. Tried: {[str(c) for c in candidates]}"
    )


def _percentile_limits(arr: np.ndarray) -> Optional[Tuple[float, float]]:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return None
    nz = v[v > 0]
    if nz.size > 0:
        v = nz
    vmin = float(np.percentile(v, 2))
    vmax = float(np.percentile(v, 98))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = max(vmin + 1.0, float(np.max(v)))
    return vmin, vmax


def _n_lidar_hint(args, frame_cache: VizFrameCache) -> Optional[int]:
    extra = frame_cache.load_disk_manifest_extra()
    n = extra.get("n_lidar_scans")
    if n is not None:
        return int(n)
    if args.sync_summary:
        with open(args.sync_summary, "r", encoding="utf-8") as f:
            summary = json.load(f)
        lf = summary.get("lidar_frames")
        if lf is not None:
            return int(lf)
    return None


def _filter_pair_ids_by_lidar_count(
    lidar_idx: np.ndarray, pair_ids: List[int], n_lidar: Optional[int]
) -> List[int]:
    if n_lidar is None:
        return list(pair_ids)
    valid: List[int] = []
    skipped = 0
    for pid in pair_ids:
        if int(lidar_idx[pid]) >= n_lidar:
            skipped += 1
            continue
        valid.append(pid)
    if skipped:
        csv_max = int(np.max(lidar_idx[pair_ids]))
        print(
            f"WARNING: skipped {skipped}/{len(pair_ids)} pairs with "
            f"lidar_idx >= {n_lidar} (max lidar_idx in selection: {csv_max})."
        )
    return valid


def _filter_pair_ids_by_lidar_bounds(renderer: "SyncVideoRenderer", pair_ids: List[int]) -> List[int]:
    return _filter_pair_ids_by_lidar_count(renderer.lidar_idx, pair_ids, renderer.n_lidar)


class RadarLoader:
    """Radar-only loader and validation (no lidar)."""

    def __init__(self, args):
        self.args = args
        self.radar_idx, self.lidar_idx, self.delta_ms, self.radar_t, _ = load_sync_pairs_full(
            args.sync_csv
        )

        cfg_path = _resolve_cfg_file(args.cfg_file)
        self.radar = radarConfig()
        self.radar.parse_radar(cfg_file=cfg_path)
        self.cfg_path = cfg_path
        self.sync_ppf = _load_sync_packets_per_frame(args, args.radar_h5)
        self.radar_processing = str(getattr(args, "radar_processing", "legacy"))
        self.use_radar_bev = use_radar_bev_panel(args)
        self.canonical_lateral_range, self.canonical_forward_range = canonical_scene_bounds()
        self.lateral_range, self.forward_range = clamp_display_bounds(
            *scene_bounds_from_args(args),
            cache_lateral=self.canonical_lateral_range,
            cache_forward=self.canonical_forward_range,
        )
        self.display_bev_kw = {
            "lateral_range": self.lateral_range,
            "forward_range": self.forward_range,
            "res": DEFAULT_SCENE_GRID_RES,
        }
        self.cache_lidar_kw = {
            "x_range": self.canonical_lateral_range,
            "y_range": self.canonical_forward_range,
            "res": DEFAULT_SCENE_GRID_RES,
        }

        print("Checking radar data...")
        self.packet_ts = read_radar_packet_timestamps(args.radar_h5)
        n_packets, spf, self.max_adc_frame, _ = radar_adc_bounds(args.radar_h5, self.radar)
        print(f"  cfg: {cfg_path}")
        print(
            f"  packets: {n_packets} | samples/frame: {spf} "
            f"({self.radar.num_adc_samples}x{self.radar.num_chirps}x{self.radar.num_rx}) "
            f"| ADC frames 0..{self.max_adc_frame} | sync_ppf={self.sync_ppf}"
        )

    def adc_frame_for_pair(self, pid: int) -> int:
        ridx = int(self.radar_idx[pid])
        rt = float(self.radar_t[pid])
        if self.args.radar_load_mode == "timestamp":
            return radar_t_to_adc_frame(rt, self.packet_ts, self.radar, self.max_adc_frame)
        return sync_radar_idx_to_adc_frame(
            ridx, self.sync_ppf, self.radar, self.max_adc_frame
        )

    def load_radar_panel(self, pid: int) -> Tuple[np.ndarray, int]:
        adc_frame_idx = self.adc_frame_for_pair(pid)
        cache = getattr(self, "frame_cache", None)
        range_az = None
        if cache is not None:
            range_az = cache.get_radar(adc_frame_idx)
        if range_az is None:
            adc_frame = load_radar_adc_frame(self.args.radar_h5, adc_frame_idx, self.radar)
            range_az = compute_range_azimuth(
                adc_frame,
                self.radar,
                angle="Azimuth",
                processing=self.radar_processing,
            )
            if cache is not None:
                cache.put_radar(adc_frame_idx, range_az)
        if self.use_radar_bev:
            left = range_azimuth_to_bev(range_az, self.radar, **self.display_bev_kw)
        else:
            left = range_az
        return left, adc_frame_idx

    def validate_endpoints(self, first_pid: int, last_pid: int) -> None:
        first_adc = self.adc_frame_for_pair(first_pid)
        last_adc = self.adc_frame_for_pair(last_pid)
        left_first, _ = self.load_radar_panel(first_pid)
        left_last, _ = self.load_radar_panel(last_pid)
        diff = float(np.mean(np.abs(left_first - left_last)))

        print(
            "Radar check (first vs last pair):\n"
            f"  radar_idx {int(self.radar_idx[first_pid])} -> {int(self.radar_idx[last_pid])}\n"
            f"  adc_frame {first_adc} -> {last_adc}\n"
            f"  range-azimuth mean|diff|={diff:.4f}"
        )

        if first_adc == last_adc:
            raise RuntimeError(
                f"Radar mapping failed: first and last pairs both map to adc_frame {first_adc}. "
                f"Check --cfg_file and --sync_summary (sync_ppf={self.sync_ppf})."
            )
        if diff < 1e-6:
            raise RuntimeError(
                "Radar range-azimuth is identical at first and last pair. "
                "Check --cfg_file matches capture (server.lua) and --radar_load_mode (use sync_idx)."
            )
        print("Radar check passed.")


class SyncVideoRenderer:
    """Full radar + lidar renderer (open lidar only after radar check)."""

    def __init__(self, args, radar: RadarLoader, frame_cache: VizFrameCache):
        self.args = args
        self.radar_loader = radar
        self.frame_cache = frame_cache
        self.radar_idx = radar.radar_idx
        self.lidar_idx = radar.lidar_idx
        self.delta_ms = radar.delta_ms
        self.radar_t = radar.radar_t
        self.radar = radar.radar
        self.sync_ppf = radar.sync_ppf
        self.max_adc_frame = radar.max_adc_frame
        self.packet_ts = radar.packet_ts
        self.radar_processing = radar.radar_processing
        self.use_radar_bev = radar.use_radar_bev
        self.canonical_lateral_range = radar.canonical_lateral_range
        self.canonical_forward_range = radar.canonical_forward_range
        self.lateral_range = radar.lateral_range
        self.forward_range = radar.forward_range
        self.display_bev_kw = radar.display_bev_kw
        self.cache_lidar_kw = radar.cache_lidar_kw

        self.lidar_reader: Optional[LidarScanReader] = None
        self.n_lidar: Optional[int] = None
        self.left_vlim: Optional[Tuple[float, float]] = None
        self.lidar_vlim: Tuple[float, float] = (0.0, 1.0)
        self._warned_shape = False
        self._pair_cache: Dict[int, Tuple[np.ndarray, np.ndarray, str]] = {}
        self._diag_reader: Optional[LidarScanReader] = None

    def open_lidar(self, *, needed_lidar_indices: Optional[List[int]] = None) -> None:
        needed = needed_lidar_indices or []
        if (
            needed
            and self.frame_cache.enabled
            and self.frame_cache.all_lidar_cached(needed)
        ):
            extra = self.frame_cache.load_disk_manifest_extra()
            self.n_lidar = int(extra.get("n_lidar_scans", max(needed) + 1))
            print(
                f"Using lidar disk cache ({len(needed)} unique scans); "
                "skipping PCAP open."
            )
            return

        print("Opening lidar PCAP (index=True; may take 1–2 min)...")
        self.lidar_reader = LidarScanReader(
            self.args.lidar_pcap,
            self.args.lidar_metadata,
            frame_cache=self.frame_cache,
        )
        self.n_lidar = len(self.lidar_reader) if self.lidar_reader._len is not None else None
        if self.n_lidar is not None:
            print(f"  lidar scans in PCAP: {self.n_lidar}")

    def close(self) -> None:
        if self.lidar_reader is not None:
            self.lidar_reader.close()
        if self._diag_reader is not None:
            self._diag_reader.close()
            self._diag_reader = None

    def _raw_fingerprint_reader(self) -> Optional[LidarScanReader]:
        if self.lidar_reader is not None:
            return self.lidar_reader
        if self._diag_reader is not None:
            return self._diag_reader
        return None

    def _ensure_diag_reader(self) -> LidarScanReader:
        """Temporary PCAP reader for --diagnose_lidar when cache skipped PCAP open."""
        if self.lidar_reader is not None:
            return self.lidar_reader
        if self._diag_reader is None:
            print("  (--diagnose_lidar: opening PCAP for raw scan verification)")
            self._diag_reader = LidarScanReader(
                self.args.lidar_pcap,
                self.args.lidar_metadata,
                frame_cache=None,
            )
        return self._diag_reader

    def diagnose_lidar(self, valid_ids: List[int], *, full: bool = False) -> dict:
        """Separate PCAP read issues from BEV/display issues."""
        unique_lidar = sorted({int(self.lidar_idx[p]) for p in valid_ids})
        view = str(self.args.lidar_view)
        load_bev_display = lambda lidx: lidar_panel_for_imshow(
            self._load_lidar_panel(lidx), view
        )

        raw_fn = None
        fresh_fn = None
        reader = self._raw_fingerprint_reader()
        if full and reader is None:
            reader = self._ensure_diag_reader()
        if reader is not None:
            raw_fn = reader.raw_fingerprint
            if self.frame_cache.enabled and any(
                self.frame_cache.has_lidar(i) for i in unique_lidar
            ):
                fresh_fn = lambda lidx, r=reader, v=view: lidar_panel_for_imshow(
                    r.get_panel(lidx, view=v), v
                )

        return print_lidar_diagnosis(
            unique_lidar,
            load_bev_display=load_bev_display,
            raw_fingerprint=raw_fn,
            fresh_bev=fresh_fn,
            full=full,
        )

    def _lidar_panel_title(self) -> str:
        titles = {
            "bev": "Lidar BEV (reflectivity)",
            "pointcloud": "Lidar point cloud (XY density)",
            "range": "Lidar range image (reflectivity H×W)",
        }
        return titles.get(str(self.args.lidar_view), "Lidar")

    def _load_lidar_panel(self, lidx: int) -> np.ndarray:
        view = str(self.args.lidar_view)
        if self.lidar_reader is not None:
            panel = self.lidar_reader.get_panel(lidx, view=view, **self.cache_lidar_kw)
        elif self.frame_cache.lidar_view == view:
            cached = self.frame_cache.get_lidar(lidx)
            if cached is None:
                raise RuntimeError(
                    f"Lidar scan {lidx} ({view}) not in cache and PCAP is not open."
                )
            panel = cached
        else:
            raise RuntimeError(
                f"Lidar scan {lidx} ({view}) not in cache and PCAP is not open."
            )
        if lidar_view_is_topdown(view):
            return crop_topdown_panel(
                panel,
                panel_lateral=self.canonical_lateral_range,
                panel_forward=self.canonical_forward_range,
                display_lateral=self.lateral_range,
                display_forward=self.forward_range,
                res=DEFAULT_SCENE_GRID_RES,
            )
        return panel

    def load_pair(self, pid: int) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
        if pid in self._pair_cache:
            left, right, title = self._pair_cache[pid]
            if self.args.fixed_color_scale and self.left_vlim is None:
                self.left_vlim = _percentile_limits(left)
            self.lidar_vlim = lidar_panel_color_limits(
                right, str(self.args.lidar_view)
            )
            return left, right, title

        ridx = int(self.radar_idx[pid])
        lidx = int(self.lidar_idx[pid])
        dt = float(self.delta_ms[pid])

        if self.n_lidar is not None and lidx >= self.n_lidar:
            return None

        try:
            left, adc_frame_idx = self.radar_loader.load_radar_panel(pid)
        except (IndexError, ValueError):
            return None

        try:
            lidar_panel = self._load_lidar_panel(lidx)
        except (IndexError, RuntimeError):
            return None

        view = str(self.args.lidar_view)
        right = lidar_panel_for_imshow(lidar_panel, view)
        self.lidar_vlim = lidar_panel_color_limits(lidar_panel, view)

        if self.args.fixed_color_scale and self.left_vlim is None:
            self.left_vlim = _percentile_limits(left)

        title = (
            f"pair={pid}  radar_idx={ridx}  adc_frame={adc_frame_idx}  "
            f"lidar_idx={lidx}  delta={dt:.1f} ms"
        )
        reader = self._raw_fingerprint_reader()
        if reader is not None and str(self.args.lidar_view) != "bev":
            try:
                fp = reader.raw_fingerprint(lidx)
                title += f"  cksum={fp['range_checksum']}"
            except Exception:
                pass
        return left, right, title

    def precache(self, valid_ids: List[int]) -> None:
        """Load all unique radar/lidar frames once; reuse during ffmpeg encode."""
        self.diagnose_lidar(valid_ids, full=bool(self.args.diagnose_lidar))

        unique_lidar = sorted({int(self.lidar_idx[p]) for p in valid_ids})
        unique_adc = sorted({self.radar_loader.adc_frame_for_pair(p) for p in valid_ids})

        lidar_hits = sum(1 for i in unique_lidar if self.frame_cache.has_lidar(i))
        radar_hits = sum(1 for a in unique_adc if self.frame_cache.has_radar(a))
        lidar_miss = len(unique_lidar) - lidar_hits
        radar_miss = len(unique_adc) - radar_hits

        if self.frame_cache.enabled:
            print(
                f"Frame cache ({self.frame_cache.cache_dir}): "
                f"radar {radar_hits}/{len(unique_adc)} hit, "
                f"lidar {lidar_hits}/{len(unique_lidar)} hit"
            )
        if lidar_miss > 0 and self.lidar_reader is None:
            raise RuntimeError("Need lidar PCAP open to build missing lidar cache entries.")

        if lidar_miss == 0 and radar_miss == 0:
            print("All unique radar/lidar frames already on disk; assembling pair cache...")
        else:
            print(
                f"Pre-caching {len(valid_ids)} video frames "
                f"({radar_miss} radar + {lidar_miss} lidar to compute)..."
            )

        if lidar_miss > 0:
            built = 0
            t_lidar = time.perf_counter()
            pending = [i for i in unique_lidar if not self.frame_cache.has_lidar(i)]
            print(
                f"  Building {len(pending)} lidar panels "
                f"(lidar_idx {pending[0]}..{pending[-1]})..."
            )
            for lidx in pending:
                t0 = time.perf_counter()
                self._load_lidar_panel(lidx)
                built += 1
                dt = time.perf_counter() - t0
                # Progress every 50, last 30, or slow frames (>3s on USB).
                tail = built > lidar_miss - 30
                if built % 50 == 0 or tail or dt > 3.0 or built == lidar_miss:
                    elapsed = time.perf_counter() - t_lidar
                    print(
                        f"  lidar cache built {built}/{lidar_miss}  "
                        f"lidar_idx={lidx}  ({dt:.1f}s this, {elapsed:.0f}s total)",
                        flush=True,
                    )
            print(
                f"  Lidar cache done ({built}/{lidar_miss} built in "
                f"{time.perf_counter() - t_lidar:.0f}s).",
                flush=True,
            )

        if radar_miss > 0:
            for adc in unique_adc:
                if self.frame_cache.has_radar(adc):
                    continue
                adc_frame = load_radar_adc_frame(self.args.radar_h5, adc, self.radar)
                range_az = compute_range_azimuth(
                    adc_frame,
                    self.radar,
                    angle="Azimuth",
                    processing=self.radar_processing,
                )
                self.frame_cache.put_radar(adc, range_az)

        print(
            f"  Assembling {len(valid_ids)} synced pairs from cache...",
            flush=True,
        )
        t_pairs = time.perf_counter()
        for i, pid in enumerate(valid_ids):
            loaded = self.load_pair(pid)
            if loaded is None:
                continue
            left, right, title = loaded
            self._pair_cache[pid] = (left, right, title)
            if (i + 1) % 50 == 0 or (i + 1) == len(valid_ids):
                print(
                    f"  pair cache {i + 1}/{len(valid_ids)}  "
                    f"({time.perf_counter() - t_pairs:.0f}s)",
                    flush=True,
                )

        extra = {}
        if self.n_lidar is not None:
            extra["n_lidar_scans"] = int(self.n_lidar)
        self.frame_cache.save_manifest(extra)
        print(f"Pre-cache done ({len(self._pair_cache)} pairs in memory).")

    def set_panel(
        self,
        ax,
        im,
        data,
        vlim,
        panel_title: str,
        *,
        origin: str = "lower",
        extent: Optional[Tuple[float, float, float, float]] = None,
        aspect: str = "auto",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ):
        data = np.nan_to_num(data, nan=0.0)
        if im is None or im.get_array().shape != data.shape:
            if im is not None and im.get_array().shape != data.shape:
                if not self._warned_shape:
                    print("WARNING: panel shape changed; re-drawing panel.")
                    self._warned_shape = True
            ax.clear()
            vmin, vmax = vlim if vlim else (None, None)
            out = ax.imshow(
                data,
                aspect=aspect,
                origin=origin,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            if extent is not None:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.set_title(panel_title)
            return out
        im.set_data(data)
        if vlim:
            im.set_clim(*vlim)
        ax.set_title(panel_title)
        return im

    def radar_axis(self, panel: np.ndarray) -> dict:
        return radar_panel_axis(
            panel,
            self.radar,
            use_bev=self.use_radar_bev,
            lateral_range=self.lateral_range,
            forward_range=self.forward_range,
        )

    def lidar_axis(self) -> dict:
        return lidar_panel_axis(
            self.args.lidar_view,
            lateral_range=self.lateral_range,
            forward_range=self.forward_range,
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create MP4: radar range-azimuth + lidar BEV for each synced pair."
    )
    p.add_argument("--sync_csv", default=None, help="Matched pairs CSV (or set via --dataset).")
    p.add_argument("--sync_summary", default=None)
    p.add_argument("--sync_packets_per_frame", type=int, default=None)
    p.add_argument("--radar_h5", default=None, help="Radar .h5 (or set via --dataset).")
    p.add_argument("--lidar_pcap", default=None, help="Lidar .pcap (or set via --dataset).")
    p.add_argument("--lidar_metadata", default=None)
    p.add_argument("--cfg_file", default="./code/mmWaveStudio/server.lua")
    p.add_argument(
        "--radar_load_mode",
        choices=["sync_idx", "timestamp"],
        default="sync_idx",
        help="sync_idx: map CSV radar_idx to ADC frame (recommended).",
    )
    p.add_argument("--out_video", default="./code/sync/res/sync_data.mp4")
    p.add_argument("--video_fps", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=-1)
    p.add_argument("--show_radar_bev", action="store_true")
    p.add_argument(
        "--align_scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When lidar view is bev/pointcloud, show radar BEV on the same lateral/forward "
        "meter grid (default: on). Use --no-align_scene to keep range-azimuth on the left.",
    )
    p.add_argument(
        "--scene_lateral",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=list(DEFAULT_SCENE_LATERAL_RANGE),
        help="Shared top-down lateral Y bounds in meters (default: -40 40).",
    )
    p.add_argument(
        "--scene_forward",
        type=float,
        nargs=2,
        metavar=("MIN_M", "MAX_M"),
        default=list(DEFAULT_SCENE_FORWARD_RANGE),
        help="Shared top-down forward X bounds in meters (default: 0 20). "
        "Display-only; disk cache is built at 0..80 m and cropped at render time.",
    )
    p.add_argument(
        "--radar_processing",
        choices=["legacy", "ti"],
        default="legacy",
        help="Radar range-azimuth chain: legacy (raw FFT+Capon) or ti "
        "(clutter removal + Hamming windows + zero-Doppler notch).",
    )
    p.add_argument(
        "--lidar_view",
        choices=["bev", "pointcloud", "range"],
        default="bev",
        help="Right panel: bev (histogram), pointcloud (XY point density from PCAP), "
        "or range (native H×W reflectivity scan).",
    )
    p.add_argument("--fixed_color_scale", action="store_true")
    p.add_argument(
        "--radar_only",
        action="store_true",
        help="Run radar validation only; do not open lidar or encode video.",
    )
    p.add_argument("--debug_png_dir", default=None, help="Save radar/lidar debug PNGs.")
    p.add_argument(
        "--cache_dir",
        default=str(_SYNC_DIR / "res" / "viz_cache"),
        help="Disk cache for radar/lidar rendered frames (speeds up re-runs).",
    )
    p.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable disk cache; always read H5/PCAP fresh.",
    )
    p.add_argument(
        "--refresh_cache",
        action="store_true",
        help="Ignore existing cache files and rebuild them.",
    )
    p.add_argument(
        "--diagnose_lidar",
        action="store_true",
        help="Full lidar diagnostic: raw PCAP checksums vs BEV, cache-vs-fresh.",
    )
    add_dataset_arguments(p)
    return p


def main() -> None:
    args = apply_dataset_config(
        build_argparser().parse_args(),
        required=("radar_h5", "lidar_pcap", "sync_csv"),
    )

    radar_idx, _, _, _, _ = load_sync_pairs_full(args.sync_csv)
    if radar_idx.size == 0:
        raise RuntimeError("sync_csv has no matched pairs.")

    pair_ids: List[int] = list(np.arange(radar_idx.size)[:: max(1, args.stride)])
    if args.max_frames > 0:
        pair_ids = pair_ids[: args.max_frames]
    if len(pair_ids) < 2:
        raise RuntimeError(f"Only {len(pair_ids)} pairs; need at least 2.")

    first_pid, last_pid = pair_ids[0], pair_ids[-1]

    # --- Phase 1: radar only (fast) ---
    radar_loader = RadarLoader(args)
    frame_cache = VizFrameCache.from_render_args(
        args, cfg_path=radar_loader.cfg_path, sync_ppf=radar_loader.sync_ppf
    )
    radar_loader.frame_cache = frame_cache
    radar_loader.validate_endpoints(first_pid, last_pid)

    if args.radar_only:
        print("--radar_only set; skipping lidar and video encoding.")
        return

    # --- Phase 2: lidar + video ---
    renderer = SyncVideoRenderer(args, radar_loader, frame_cache)
    try:
        n_lidar_hint = _n_lidar_hint(args, frame_cache)
        valid_ids = _filter_pair_ids_by_lidar_count(
            renderer.lidar_idx, pair_ids, n_lidar_hint
        )
        needed_lidar = sorted({int(renderer.lidar_idx[p]) for p in valid_ids})
        renderer.open_lidar(needed_lidar_indices=needed_lidar)
        if n_lidar_hint is None and renderer.n_lidar is not None:
            valid_ids = _filter_pair_ids_by_lidar_bounds(renderer, valid_ids)
            needed_lidar = sorted({int(renderer.lidar_idx[p]) for p in valid_ids})
        if len(valid_ids) < 2:
            csv_max = int(np.max(renderer.lidar_idx))
            raise RuntimeError(
                f"Only {len(valid_ids)} pair(s) have lidar_idx < PCAP length "
                f"({renderer.n_lidar} scans; CSV lidar_idx up to {csv_max}). "
                "Re-run sync_radar_lidar.py with this PCAP, or pass the PCAP "
                "that was used when sync_pairs.csv was created."
            )

        first = renderer.load_pair(valid_ids[0])
        last = renderer.load_pair(valid_ids[-1])
        if not first or not last:
            bad = []
            for tag, pid in (("first", valid_ids[0]), ("last", valid_ids[-1])):
                if renderer.load_pair(pid) is None:
                    bad.append(
                        f"{tag} pair={pid} radar_idx={int(renderer.radar_idx[pid])} "
                        f"lidar_idx={int(renderer.lidar_idx[pid])}"
                    )
            raise RuntimeError(
                "Failed to load endpoint pairs after lidar bounds filter:\n  "
                + "\n  ".join(bad)
            )
        first_lidx = int(renderer.lidar_idx[valid_ids[0]])
        last_lidx = int(renderer.lidar_idx[valid_ids[-1]])
        print(
            "Full pair diversity (first vs last valid pair):\n"
            f"  radar  mean|diff|={float(np.mean(np.abs(first[0] - last[0]))):.4f}\n"
            f"  lidar  mean|diff|={float(np.mean(np.abs(first[1] - last[1]))):.4f}\n"
            f"  lidar  lidar_idx {first_lidx} -> {last_lidx} | "
            f"first max={float(np.max(first[1])):.1f} "
            f"last max={float(np.max(last[1])):.1f}"
        )
        lidar_diff = float(np.mean(np.abs(first[1] - last[1])))
        if float(np.max(first[1])) <= 0:
            raise RuntimeError(
                "Lidar BEV is empty (all zeros). Check PCAP/metadata paths and "
                "that ouster-sdk is installed (pip install ouster-sdk)."
            )
        if lidar_diff < 1e-3:
            print(
                f"WARNING: lidar {args.lidar_view} panel barely changes between first "
                f"and last pair (mean|diff|={lidar_diff:.6f}). "
                "Video may look static on the right panel."
            )

        renderer.precache(valid_ids)

        if args.debug_png_dir:
            dbg = Path(args.debug_png_dir)
            dbg.mkdir(parents=True, exist_ok=True)
            for tag, pid in [
                ("first", valid_ids[0]),
                ("mid", valid_ids[len(valid_ids) // 2]),
                ("last", valid_ids[-1]),
            ]:
                loaded = renderer.load_pair(pid)
                if not loaded:
                    continue
                left, right, title = loaded
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                radar_ax = renderer.radar_axis(left)
                lidar_ax = renderer.lidar_axis()
                lidar_vlim = lidar_panel_color_limits(right, args.lidar_view)
                axes[0].imshow(
                    left,
                    origin=radar_ax["origin"],
                    extent=radar_ax["extent"],
                    aspect=radar_ax["aspect"],
                    cmap="viridis",
                )
                axes[0].set_xlabel(radar_ax["xlabel"])
                axes[0].set_ylabel(radar_ax["ylabel"])
                axes[1].imshow(
                    right,
                    origin=lidar_ax["origin"],
                    extent=lidar_ax["extent"],
                    aspect=lidar_ax["aspect"],
                    cmap="viridis",
                    vmin=lidar_vlim[0],
                    vmax=lidar_vlim[1],
                )
                axes[1].set_xlabel(lidar_ax["xlabel"])
                axes[1].set_ylabel(lidar_ax["ylabel"])
                fig.suptitle(title)
                fig.savefig(dbg / f"debug_{tag}_pair{pid}.png", dpi=120)
                plt.close(fig)
            print(f"Wrote debug PNGs to {dbg}")

        left0, right0, title0 = renderer.load_pair(valid_ids[0])
        assert left0 is not None
        left_title = "Radar BEV" if renderer.use_radar_bev else "Radar Range-Azimuth (dB)"
        radar_ax = renderer.radar_axis(left0)
        lidar_ax = renderer.lidar_axis()
        if renderer.use_radar_bev:
            print(
                "Display scene (m): lateral "
                f"{renderer.lateral_range[0]:.0f}..{renderer.lateral_range[1]:.0f}, "
                f"forward {renderer.forward_range[0]:.0f}..{renderer.forward_range[1]:.0f} "
                f"(cache grid forward 0..{CANONICAL_SCENE_FORWARD_RANGE[1]:.0f})"
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        panels = {
            "im0": renderer.set_panel(
                axes[0],
                None,
                left0,
                renderer.left_vlim,
                left_title,
                origin=radar_ax["origin"],
                extent=radar_ax["extent"],
                aspect=radar_ax["aspect"],
                xlabel=radar_ax["xlabel"],
                ylabel=radar_ax["ylabel"],
            ),
            "im1": renderer.set_panel(
                axes[1],
                None,
                right0,
                renderer.lidar_vlim,
                renderer._lidar_panel_title(),
                origin=lidar_ax["origin"],
                extent=lidar_ax["extent"],
                aspect=lidar_ax["aspect"],
                xlabel=lidar_ax["xlabel"],
                ylabel=lidar_ax["ylabel"],
            ),
        }
        supt = fig.suptitle(title0)

        def update(i: int):
            loaded = renderer.load_pair(valid_ids[i])
            if loaded is None:
                return panels["im0"], panels["im1"], supt
            left, right, title = loaded
            panels["im0"] = renderer.set_panel(
                axes[0],
                panels["im0"],
                left,
                renderer.left_vlim,
                left_title,
                origin=radar_ax["origin"],
                extent=radar_ax["extent"],
                aspect=radar_ax["aspect"],
                xlabel=radar_ax["xlabel"],
                ylabel=radar_ax["ylabel"],
            )
            panels["im1"] = renderer.set_panel(
                axes[1],
                panels["im1"],
                right,
                renderer.lidar_vlim,
                renderer._lidar_panel_title(),
                origin=lidar_ax["origin"],
                extent=lidar_ax["extent"],
                aspect=lidar_ax["aspect"],
                xlabel=lidar_ax["xlabel"],
                ylabel=lidar_ax["ylabel"],
            )
            supt.set_text(title)
            if i > 0 and i % 50 == 0:
                print(f"  anim {i}/{len(valid_ids)} | {title}")
            return panels["im0"], panels["im1"], supt

        out_video = Path(args.out_video)
        out_video.parent.mkdir(parents=True, exist_ok=True)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(valid_ids),
            interval=1000 / max(1, args.video_fps),
            blit=False,
        )
        print(
            f"Encoding {len(valid_ids)} frames -> {out_video}\n"
            f"  (frames pre-cached; ffmpeg progress every 50 steps)"
        )
        ani.save(str(out_video), writer="ffmpeg", fps=max(1, args.video_fps), dpi=120)
        plt.close(fig)

        print(
            f"Done. ~{len(valid_ids) / max(1, args.video_fps):.1f}s @ {args.video_fps} fps -> {out_video}"
        )
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
