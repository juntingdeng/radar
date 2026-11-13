"""Benchmark exporters (merged). Subcommands: kradar | ldr64 | rdr_sparse"""
from __future__ import annotations
import sys

# ======== src/camera_projection/export_kradar.py ========
# Export our annotations (with oriented box3d) to K-Radar label + calib format.
# 
# Writes, under ``<out>/<seq>/``:
#   info_label/<idx>_<idx>.txt   — one file per pair, K-Radar v1_0 label format
#   info_calib/calib_radar_lidar.txt — radar->lidar planar (x,y) offset from our calibration
# 
# K-Radar v1_0 label format (matched to datasets/kradar_detection_v2_0.py::get_label):
#   header : ``* idx(tesseract_os2-64_cam-front_os1-128_cam-lrr)=r_l_cf_l128_cr, timestamp=<t>``
#   object : ``*, idx_p, idx_b4, Class, x, y, z, theta_DEG, l/2, w/2, h/2``
#     - theta is in DEGREES (loader multiplies by pi/180),
#     - dimensions are HALF-extents (loader does l = 2*value),
#     - separator is ", " (comma-space).
# Frame: our sensor frame (X fwd, Y left, Z up) already matches K-Radar's lidar frame.
# 
# NOTE: this exports LABELS + CALIB only. The point clouds (ldr64) and sparse radar
# (rdr_sparse ``sprdr_<idx>.npy``) still need exporting with MATCHING per-frame indices
# before the K-Radar dataloader can train on it (see --help / project notes).
# 
#     python code/sync/src/camera_projection/export_kradar.py -d 2026.05.08/17-34-27         --annotations res/.../annotations/yolo.json --out /path/to/kradar_export --seq 100
def kradar_main():
    import argparse
    import json
    import math
    import os
    import sys
    from pathlib import Path
    import numpy as np
    _THIS_DIR = Path(__file__).resolve().parent
    _SYNC_DIR = _THIS_DIR.parent
    _CALIB_DIR = _SYNC_DIR / "calibration"
    for _p in (_SYNC_DIR, _THIS_DIR, _CALIB_DIR):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402
    from annotations import load_annotations  # noqa: E402
    DEFAULT_CLASS_MAP = {
        "person": "Pedestrian",
        "car": "Sedan",
        "bus": "Bus or Truck",
        "truck": "Bus or Truck",
        "bicycle": "Bicycle",
        "motorcycle": "Motorcycle",
    }
    def build_argparser():
        p = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        p.add_argument("--calibration_json", default=None)
        p.add_argument("--camera_sync_csv", default=None)
        p.add_argument("--annotations", default=None, help="annotations JSON with box3d (default: <calib>/annotations/yolo.json).")
        p.add_argument("--out", required=True, help="Output root; a <seq>/ folder is created under it.")
        p.add_argument("--seq", default="100", help="Sequence id/name for the exported folder.")
        p.add_argument("--min_points", type=int, default=15,
                       help="A box needs at least this many oriented-cluster points to be a "
                       "positive label. 0 disables the point check.")
        p.add_argument("--include_unreliable", action="store_true",
                       help="Export unreliable boxes as normal (positive) labels anyway.")
        p.add_argument("--unreliable_as_dontcare", action="store_true",
                       help="Emit detected-but-unreliable boxes (occluded/single-face/implausible) "
                       "as class 'DontCare' instead of dropping them, so a detector isn't penalized "
                       "for finding these real-but-hard objects. The KITTI eval excludes DontCare "
                       "regions from false positives. NOTE: add 'DontCare':[False,-1,...] to your "
                       "K-Radar label config so the training loader drops (not errors on) them.")
        p.add_argument("--radar_unseen_dontcare", action=argparse.BooleanOptionalAction, default=True,
                       help="Radar is the target modality: a lidar-reliable box that radar did NOT "
                       "confirm (radar_confirmed=False) is emitted as 'DontCare' instead of a positive "
                       "label, so the radar detector isn't penalized for missing objects radar can't "
                       "physically sense. ON by default; only affects annotations generated WITH radar "
                       "(boxes carrying radar_confirmed). Use --no-radar_unseen_dontcare to keep them "
                       "positive regardless of radar.")
        p.add_argument("--max_range_m", type=float, default=None,
                       help="Drop boxes beyond this range (m). Keeps labels in the regime the sensor "
                       "actually supports 3D; pair with a matching eval ROI.")
        p.add_argument("--frame_diff", type=int, default=0,
                       help="radar-lidar frame offset written into calib (0: already synced).")
        add_dataset_arguments(p)
        return p
    def _load_pair_rows(csv_path):
        import csv as _csv
        rows = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                rows[int(r["pair_idx"])] = r
        return rows
    def _kradar_line(idx_p, cls, o):
        """One v1_0 object line: half-extents, theta in degrees, ', ' separated."""
        x, y, z = o["cx"], o["cy"], o["cz"]
        th_deg = math.degrees(float(o["yaw"]))
        hl, hw, hh = o["length"] / 2.0, o["width"] / 2.0, o["height"] / 2.0
        return (f"*, {idx_p}, -1, {cls}, {x:.4f}, {y:.4f}, {z:.4f}, "
                f"{th_deg:.4f}, {hl:.4f}, {hw:.4f}, {hh:.4f}")
    def main() -> int:
        args = apply_dataset_config(build_argparser().parse_args())
        if not args.annotations:
            args.annotations = str(Path(args.calibration_json).parent / "annotations" / "yolo.json")

        calib = json.loads(Path(args.calibration_json).read_text())
        # prefer the auto radar_to_lidar; planar translation (forward=x, lateral=y)
        try:
            from extrinsic_io import get_extrinsic
            R2L, r_src = get_extrinsic(calib, "radar_to_lidar")
        except Exception:  # noqa: BLE001
            R2L, r_src = np.asarray(calib["radar_to_lidar"], float), "radar_to_lidar"
        rx, ry = float(R2L[0, 3]), float(R2L[1, 3])

        objects = load_annotations(args.annotations)["objects"]
        rows = _load_pair_rows(Path(args.camera_sync_csv))
        class_map = DEFAULT_CLASS_MAP

        seq_dir = Path(args.out) / str(args.seq)
        (seq_dir / "info_label").mkdir(parents=True, exist_ok=True)
        (seq_dir / "info_calib").mkdir(parents=True, exist_ok=True)

        # calib (radar->lidar planar): "frame difference,x,y" then "<diff>,<x>,<y>"
        with open(seq_dir / "info_calib" / "calib_radar_lidar.txt", "w") as f:
            f.write("frame difference,x,y\n")
            f.write(f"{args.frame_diff},{rx:.4f},{ry:.4f}\n")

        n_files = n_pos = n_dc = n_dc_radar = n_far = n_skip = 0
        for pid_str, boxes in sorted(objects.items(), key=lambda kv: int(kv[0])):
            pid = int(pid_str)
            r = rows.get(pid, {})
            ts = r.get("lidar_t") or r.get("radar_t") or "0.0"
            fid = pid  # single frame id across the 5 sensor slots (name point clouds to match)
            idx5 = "_".join([f"{fid:05d}"] * 5)
            lines = [f"* idx(tesseract_os2-64_cam-front_os1-128_cam-lrr)={idx5}, timestamp={ts}"]
            kept = 0
            for b in boxes:
                o = b.get("box3d")
                if o is None:                                   # no 3D geometry at all
                    n_skip += 1
                    continue
                if args.max_range_m is not None and np.hypot(o["cx"], o["cy"]) > args.max_range_m:
                    n_far += 1
                    continue
                veh_cls = class_map.get(b.get("label", ""), None)
                if veh_cls is None:                             # unmapped class
                    n_skip += 1
                    continue
                enough = (not args.min_points) or o.get("n_points", 0) >= args.min_points
                reliable = (o.get("reliable", True) and enough) or args.include_unreliable
                # radar is the target modality: radar_confirmed is explicitly False only when the
                # box was cross-checked (annotations generated with radar) and no radar return landed
                # on it. None = not radar-checked -> don't gate.
                radar_unseen = args.radar_unseen_dontcare and (b.get("radar_confirmed") is False)
                if reliable and not radar_unseen:               # real + radar sees it -> positive
                    lines.append(_kradar_line(kept, veh_cls, o))
                    kept += 1
                    n_pos += 1
                elif reliable and radar_unseen:                 # real but radar-blind -> DontCare
                    lines.append(_kradar_line(kept, "DontCare", o))
                    kept += 1
                    n_dc_radar += 1
                elif args.unreliable_as_dontcare:               # detected but geometry uncertain
                    lines.append(_kradar_line(kept, "DontCare", o))
                    kept += 1
                    n_dc += 1
                else:
                    n_skip += 1
            if kept == 0:
                continue
            fname = f"{fid:05d}_{fid:05d}.txt"
            with open(seq_dir / "info_label" / fname, "w") as f:
                f.write("\n".join(lines) + "\n")
            n_files += 1

        print(f"radar_to_lidar source: {r_src}; calib x,y = ({rx:.3f},{ry:.3f})")
        print(f"Wrote {n_files} label file(s) -> {seq_dir}/info_label")
        print(f"  positive labels      : {n_pos}  (reliable AND radar-confirmed)")
        print(f"  DontCare (radar-unseen): {n_dc_radar}" + ("" if args.radar_unseen_dontcare
              else "  (--no-radar_unseen_dontcare: kept positive)"))
        print(f"  DontCare (unreliable): {n_dc}" + ("" if args.unreliable_as_dontcare
                                                    else "  (use --unreliable_as_dontcare to emit these)"))
        print(f"  dropped >range       : {n_far}")
        print(f"  dropped other        : {n_skip}  (no box3d / unreliable+not-dontcare / unmapped class)")
        print("NOTE: still need ldr64 point clouds + rdr_sparse/sprdr_<idx>.npy named with the "
              "same 5-digit frame id before the K-Radar dataloader can train on this sequence.")
        return 0
    if True:
        raise SystemExit(main())

# ======== src/camera_projection/export_ldr64.py ========
# Export lidar point clouds (``lpc_<idx>.npy``) for the K-Radar model, from the sync pcap.
# 
# Third companion to ``export_kradar.py`` (labels+calib) and ``export_rdr_sparse.py`` (radar).
# This writes the lidar side: per frame it reads the Ouster scan (sync ``LidarScanReader``), takes
# each valid return's ``[x, y, z, intensity]`` in the LIDAR sensor frame (X fwd, Y left, Z up) --
# the SAME frame as your labels and rdr_sparse -- and saves a float32 ``.npy``.
# 
# Load it with the K-Radar dataloader's ``processed: True`` path (``get_ldr64`` reads
# ``osp.join(ldr64.dir, seq, f'lpc_{idx}.npy')`` and skips calib/parsing). The model uses
# ``n_used = 4`` = ``[x, y, z, intensity]`` (extra columns ignored); ROI is applied by the loader.
# 
# Frame note: your labels (export_kradar) + rdr_sparse are already in the lidar frame, so no
# calib shift is applied here (K-Radar's ``z_offset``/``calib_radar_lidar`` corrections are for
# their own capture rig). Set ``ldr64.calib: False`` (or use ``processed: True``, which skips it).
# 
#     python code/sync/src/camera_projection/export_ldr64.py -d 2026.05.08/17-34-27         --pairs 5000 5508 12720 --out /path/to/kradar_export --seq 100
def ldr64_main():
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
    if True:
        raise SystemExit(main())

# ======== src/camera_projection/export_rdr_sparse.py ========
# Export radar sparse tensors (``sprdr_<idx>.npy``) for the K-Radar model, from the sync .h5.
# 
# Companion to ``export_kradar.py`` (which writes labels + calib). This writes the radar side:
# per frame it reuses the VERIFIED sync radar chain (correct 3TX / 12-virtual-channel AWR2944
# config -- ``load_radar_adc_frame`` + ``compute_range_azimuth``), builds a Cartesian BEV,
# keeps the top-N% highest-power cells (K-Radar RTNH "quantile" sparsification), normalizes the
# power to ~O(1), clips to the model ROI, and saves ``[x, y, z, power]`` float32 point clouds.
# 
# The K-Radar generative pipeline consumes rdr_sparse as ``[x, y, z, power]`` with
# ``input_dim=4`` (Doppler unused; see models/generatives/unet.py), so 4 columns is exactly right.
# 
# IMPORTANT -- no elevation: the sync capture fires 3 azimuth-collinear TX (``tx3``), so there is
# NO elevation aperture. ``z`` is therefore not resolvable and every point is emitted at ``z=0``
# (a 2D/BEV radar cloud stored in the 3D format). This is a property of the capture config, not a
# bug; the model's ``[x,y,z,power]`` input still fits.
# 
#     python code/sync/src/camera_projection/export_rdr_sparse.py -d 2026.05.08/17-34-27         --pairs 5000 5508 12720 --out /path/to/kradar_export --seq 100 --quantile_rate 0.01
def rdr_sparse_main():
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
    if True:
        raise SystemExit(main())

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'kradar': kradar_main, 'ldr64': ldr64_main, 'rdr_sparse': rdr_sparse_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
