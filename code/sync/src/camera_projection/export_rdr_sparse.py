"""Export radar sparse tensors (``sprdr_<idx>.npy``) for the K-Radar model, from the sync .h5.

Companion to ``export_kradar.py`` (which writes labels + calib). This writes the radar side:
per frame it reuses the VERIFIED sync radar chain (correct 3TX / 12-virtual-channel AWR2944
config -- ``load_radar_adc_frame`` + ``compute_range_azimuth``), builds a Cartesian BEV,
keeps the top-N% highest-power cells (K-Radar RTNH "quantile" sparsification), normalizes the
power to ~O(1), clips to the model ROI, and saves ``[x, y, z, power]`` float32 point clouds.

The K-Radar generative pipeline consumes rdr_sparse as ``[x, y, z, power]`` with
``input_dim=4`` (Doppler unused; see models/generatives/unet.py), so 4 columns is exactly right.

IMPORTANT -- no elevation: the sync capture fires 3 azimuth-collinear TX (``tx3``), so there is
NO elevation aperture. ``z`` is therefore not resolvable and every point is emitted at ``z=0``
(a 2D/BEV radar cloud stored in the 3D format). This is a property of the capture config, not a
bug; the model's ``[x,y,z,power]`` input still fits.

    python code/sync/src/camera_projection/export_rdr_sparse.py -d 2026.05.08/17-34-27 \
        --pairs 5000 5508 12720 --out /path/to/kradar_export --seq 100 --quantile_rate 0.01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

_THIS_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _THIS_DIR.parent
_CODE_ROOT = _SYNC_DIR.parent
for _p in (_SYNC_DIR, _CODE_ROOT, _SYNC_DIR / "calibration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402


# --- power normalization: dB power -> stable ~[0,1] (fixed bounds; see NOTE) ------------------
def normalize_power(pw: np.ndarray, method: str, lo: float, hi: float, div: float) -> np.ndarray:
    """Scale the radar power channel to a stable ~O(1) range for the model.

    Our power is a range-azimuth value in dB (compute_range_azimuth returns 10*log10). ``minmax``
    maps dB -> [0,1] with FIXED bounds -- fixed (not per-frame) so the feature means the same
    thing in every frame. Calibrate ``lo``/``hi`` once to ~the 1st/99th percentile of the kept
    (top-N%) power over your data (the script prints per-frame stats to help).
    """
    pw = np.asarray(pw, dtype=np.float32)
    if pw.size == 0 or method == "none":
        return pw
    if method == "minmax":
        return np.clip((pw - lo) / (hi - lo + 1e-9), 0.0, 1.0).astype(np.float32)
    if method == "zscore":  # per-frame; not frame-consistent -- prefer minmax with fixed bounds
        return ((pw - pw.mean()) / (pw.std() + 1e-9)).astype(np.float32)
    if method == "divide":  # linear-power / const (K-Radar style; NOT for dB power)
        return (pw / div).astype(np.float32)
    raise ValueError(f"unknown power norm method {method!r}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arguments(p)
    p.add_argument("--pairs", type=int, nargs="+", required=True, help="Sync pair indices.")
    p.add_argument("--radar_h5", default=None)
    p.add_argument("--cfg_file", default=None)
    p.add_argument("--sync_csv", default=None)
    p.add_argument("--sync_summary", default=None)
    p.add_argument("--lidar_pcap", default=None)
    p.add_argument("--lidar_metadata", default=None)
    p.add_argument("--camera_sync_csv", default=None)
    p.add_argument("--calibration_json", default=None)
    p.add_argument("--warp_to_lidar", action=argparse.BooleanOptionalAction, default=True,
                   help="Rotate+translate the radar points from the radar frame into the LIDAR frame "
                   "(radar_to_lidar extrinsic) so they share the label/lpc frame. ON by default -- a "
                   "radar detector needs its input in the same frame as the boxes. --no-warp_to_lidar "
                   "keeps raw radar-frame points.")
    p.add_argument("--out", required=True, help="Output root; writes <out>/<seq>/sprdr_<id>.npy "
                   "(set the model's rdr_sparse.dir = <out>).")
    p.add_argument("--seq", default="100", help="Sequence id/folder (match export_kradar --seq).")
    # sparsification (K-Radar RTNH: 'quantile', top-N%)
    p.add_argument("--quantile_rate", type=float, default=0.01,
                   help="Keep the top fraction of power cells (0.01 = top 1%, matches rtnh_wider_1p).")
    p.add_argument("--bev_res_m", type=float, default=0.4, help="Cartesian BEV cell size (m).")
    # ROI (K-Radar wide range from cfg_rdr_ldr.yml): x fwd, y lat, z up
    p.add_argument("--roi", type=float, nargs=6, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                   default=[0.0, -16.0, -2.0, 72.0, 16.0, 7.6], help="x,y,z min/max (m).")
    p.add_argument("--z_value", type=float, default=0.0,
                   help="z assigned to every point (no elevation in tx3). Add calib z_offset if desired.")
    # radar processing chain
    p.add_argument("--radar_processing", default="ti", choices=["legacy", "ti"])
    p.add_argument("--radar_doppler_mode", default="all", choices=["all", "moving", "static"],
                   help="'all' keeps every return (K-Radar-style full-cube sparsify).")
    p.add_argument("--radar_aoa", default="dbf_nci", choices=["capon", "dbf_nci", "dbf_coh"])
    # power normalization
    p.add_argument("--power_norm", default="auto", choices=["auto", "minmax", "zscore", "divide", "none"],
                   help="'auto': minmax with lo/hi = 1st/99th pct of kept power over ALL --pairs "
                        "(frame-consistent, self-calibrating; prints the values to pin later).")
    p.add_argument("--power_db_lo", type=float, default=-30.0, help="minmax: dB mapped to 0 (calibrate).")
    p.add_argument("--power_db_hi", type=float, default=20.0, help="minmax: dB mapped to 1 (calibrate).")
    p.add_argument("--power_div_val", type=float, default=1.0e13, help="divide mode only.")
    return p


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())
    from lib.bev_render import (
        compute_range_azimuth, range_azimuth_to_bev, load_radar_adc_frame,
    )
    import make_video as mv
    import csv as _csv

    # RadarLoader parses the .lua (correct 3TX config) and gives the radar object + frame mapping.
    rargs = mv.build_argparser().parse_args([])
    for k in ("radar_h5", "cfg_file", "sync_csv", "sync_summary", "lidar_pcap", "lidar_metadata"):
        v = getattr(args, k, None)
        if v:
            setattr(rargs, k, v)
    rargs.radar_processing = args.radar_processing
    rargs.radar_doppler_mode = args.radar_doppler_mode
    rargs.radar_aoa = args.radar_aoa
    loader = mv.RadarLoader(rargs)
    radar = loader.radar

    # radar->lidar warp (so radar points share the label/lpc frame)
    R2L = None
    if args.warp_to_lidar:
        from extrinsic_io import get_extrinsic
        calib = json.loads(Path(args.calibration_json).read_text())
        R2L, r_src = get_extrinsic(calib, "radar_to_lidar")
        R2L = np.asarray(R2L, dtype=np.float64)
        print(f"warp_to_lidar: using '{r_src}' (yaw {np.degrees(np.arctan2(R2L[1,0],R2L[0,0])):.2f} deg, "
              f"t=({R2L[0,3]:.3f},{R2L[1,3]:.3f}))")

    rows = {}
    with open(args.camera_sync_csv, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rows[int(r["pair_idx"])] = r

    x_min, y_min, z_min, x_max, y_max, z_max = args.roi
    res = float(args.bev_res_m)
    # BEV grid over the ROI (radar returns only populate 0..~21 m; farther cells stay NaN).
    lat_ax = np.arange(y_min, y_max + res, res)
    fwd_ax = np.arange(x_min, x_max + res, res)

    # K-Radar loader reads osp.join(rdr_sparse.dir, seq, f'sprdr_{idx}.npy'), so write
    # <out>/<seq>/sprdr_<id>.npy and set the model's rdr_sparse.dir = <out>.
    out_dir = Path(args.out) / str(args.seq)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- pass 1: build the sparse [x,y,z,power_dB] per frame (collect all power for auto-calib) ---
    frames = []  # (pid, x, y, z, pw_db)
    for pid in sorted(args.pairs):
        if pid not in rows:
            print(f"  pair {pid}: not in sync CSV; skip.")
            continue
        adc = load_radar_adc_frame(args.radar_h5, loader.adc_frame_for_pair(pid), radar)
        ra = compute_range_azimuth(adc, radar, processing=args.radar_processing,
                                   doppler_mode=args.radar_doppler_mode, aoa_method=args.radar_aoa)
        bev = range_azimuth_to_bev(ra, radar, lateral_range=(y_min, y_max),
                                   forward_range=(x_min, x_max), res=res)  # (n_fwd, n_lat) dB
        valid = np.isfinite(bev)
        if not valid.any():
            print(f"  pair {pid}: empty BEV; skip.")
            continue
        thr = np.nanquantile(bev, 1.0 - args.quantile_rate)   # top-N% power cells
        fi, li = np.where(valid & (bev >= thr))
        x, y = fwd_ax[fi], lat_ax[li]
        pw_db = bev[fi, li].astype(np.float32)
        z = np.full_like(x, float(args.z_value), dtype=np.float32)
        if R2L is not None:   # radar-frame -> lidar frame (so points share the label/lpc frame)
            xyz = np.column_stack([x, y, z, np.ones_like(x)]) @ R2L.T
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        m = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
        frames.append((pid, x[m], y[m], z[m], pw_db[m]))

    if not frames:
        print("No usable frames.")
        return 2

    # --- resolve power-norm bounds: 'auto' calibrates lo/hi from ALL kept power (frame-consistent) ---
    method, lo, hi = args.power_norm, args.power_db_lo, args.power_db_hi
    if method == "auto":
        all_pw = np.concatenate([f[4] for f in frames])
        lo, hi = float(np.percentile(all_pw, 1)), float(np.percentile(all_pw, 99))
        method = "minmax"
        print(f"power_norm=auto -> minmax bounds from data: lo={lo:.2f} hi={hi:.2f} dB "
              f"(pin with --power_norm minmax --power_db_lo {lo:.1f} --power_db_hi {hi:.1f})")

    # --- pass 2: normalize power + save ---
    n_ok = 0
    for pid, x, y, z, pw_db in frames:
        pw = normalize_power(pw_db, method, lo, hi, args.power_div_val)
        pts = np.stack([x, y, z, pw], axis=1).astype(np.float32)   # (N, 4) [x, y, z, power]
        np.save(out_dir / f"sprdr_{pid:05d}.npy", pts)
        n_ok += 1
        print(f"  pair {pid}: {pts.shape[0]:4d} pts  norm {pw.min():.2f}..{pw.max():.2f}")

    print(f"\nWrote {n_ok} sprdr_*.npy ([x,y,z,power], z={args.z_value}) -> {out_dir}")
    print("NOTE: z is constant (tx3 = no elevation). Point count is low vs K-Radar's imaging radar "
          "(lower-res sensor) -- raise --quantile_rate or lower --bev_res_m for more points. Match "
          "--quantile_rate to your K-Radar density (rtnh_wider_1p = 0.01).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
