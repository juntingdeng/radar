"""Export lidar point clouds (``lpc_<idx>.npy``) for the K-Radar model, from the sync pcap.

Third companion to ``export_kradar.py`` (labels+calib) and ``export_rdr_sparse.py`` (radar).
This writes the lidar side: per frame it reads the Ouster scan (sync ``LidarScanReader``), takes
each valid return's ``[x, y, z, intensity]`` in the LIDAR sensor frame (X fwd, Y left, Z up) --
the SAME frame as your labels and rdr_sparse -- and saves a float32 ``.npy``.

Load it with the K-Radar dataloader's ``processed: True`` path (``get_ldr64`` reads
``osp.join(ldr64.dir, seq, f'lpc_{idx}.npy')`` and skips calib/parsing). The model uses
``n_used = 4`` = ``[x, y, z, intensity]`` (extra columns ignored); ROI is applied by the loader.

Frame note: your labels (export_kradar) + rdr_sparse are already in the lidar frame, so no
calib shift is applied here (K-Radar's ``z_offset``/``calib_radar_lidar`` corrections are for
their own capture rig). Set ``ldr64.calib: False`` (or use ``processed: True``, which skips it).

    python code/sync/src/camera_projection/export_ldr64.py -d 2026.05.08/17-34-27 \
        --pairs 5000 5508 12720 --out /path/to/kradar_export --seq 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

_THIS_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _THIS_DIR.parent
_CODE_ROOT = _SYNC_DIR.parent
for _p in (_SYNC_DIR, _CODE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arguments(p)
    p.add_argument("--pairs", type=int, nargs="+", required=True, help="Sync pair indices.")
    p.add_argument("--lidar_pcap", default=None)
    p.add_argument("--lidar_metadata", default=None)
    p.add_argument("--camera_sync_csv", default=None)
    p.add_argument("--out", required=True, help="Output root; writes <out>/<seq>/lpc_<id>.npy "
                   "(set the model's ldr64.dir = <out>, ldr64.processed = True).")
    p.add_argument("--seq", default="100", help="Sequence id/folder (match export_kradar --seq).")
    p.add_argument("--min_range_mm", type=float, default=500.0, help="Drop returns nearer than this.")
    p.add_argument("--intensity_norm", default="div", choices=["div", "auto", "none"],
                   help="'div': intensity/--intensity_div (default, Ouster reflectivity is 8-bit). "
                        "'auto': /99th-pct over all --pairs. 'none': raw.")
    p.add_argument("--intensity_div", type=float, default=255.0, help="'div' mode divisor.")
    return p


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())
    from lib.lidar_io import (
        open_pcap_scan_source, sensor_info_from_source, _scan_from_scan_set,
        scan_to_xyz, scan_range_flat, scan_reflectivity_staggered,
    )

    rows = {}
    with open(args.camera_sync_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[int(r["pair_idx"])] = r

    src = open_pcap_scan_source(args.lidar_pcap, args.lidar_metadata, index=True)
    meta = sensor_info_from_source(src)

    out_dir = Path(args.out) / str(args.seq)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- pass 1: build [x,y,z,intensity] per frame (collect intensity for optional auto-norm) ---
    frames = []  # (pid, xyzi)
    for pid in sorted(args.pairs):
        if pid not in rows:
            print(f"  pair {pid}: not in sync CSV; skip.")
            continue
        l_idx = int(rows[pid]["lidar_idx"])
        scan = _scan_from_scan_set(src[l_idx])
        xyz = scan_to_xyz(meta, scan).reshape(-1, 3)
        refl = scan_reflectivity_staggered(scan).reshape(-1).astype(np.float32)  # native layout matches xyz
        rng = scan_range_flat(scan).reshape(-1)
        valid = np.isfinite(xyz).all(axis=1) & (rng >= float(args.min_range_mm))
        xyzi = np.column_stack([xyz[valid], refl[valid]]).astype(np.float32)  # (N,4) [x,y,z,intensity]
        frames.append((pid, xyzi))

    if not frames:
        print("No usable frames.")
        return 2

    div = args.intensity_div
    if args.intensity_norm == "auto":
        all_i = np.concatenate([f[1][:, 3] for f in frames])
        div = float(np.percentile(all_i, 99)) or 1.0
        print(f"intensity_norm=auto -> divide by 99th pct = {div:.1f} "
              f"(pin with --intensity_norm div --intensity_div {div:.1f})")

    n_ok = 0
    for pid, xyzi in frames:
        if args.intensity_norm != "none":
            xyzi = xyzi.copy()
            xyzi[:, 3] = xyzi[:, 3] / (div + 1e-9)
        np.save(out_dir / f"lpc_{pid:05d}.npy", xyzi)
        n_ok += 1
        print(f"  pair {pid}: {xyzi.shape[0]:6d} pts  "
              f"x[{xyzi[:,0].min():.0f},{xyzi[:,0].max():.0f}] "
              f"intensity[{xyzi[:,3].min():.2f},{xyzi[:,3].max():.2f}]")

    print(f"\nWrote {n_ok} lpc_*.npy ([x,y,z,intensity], lidar frame) -> {out_dir}")
    print("Set model: ldr64.dir=<out>, ldr64.processed=True, ldr64.calib=False (points already "
          "in the label/rdr_sparse lidar frame). n_used=4 uses [x,y,z,intensity].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
