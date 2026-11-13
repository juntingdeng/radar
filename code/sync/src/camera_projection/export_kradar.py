"""Export our annotations (with oriented box3d) to K-Radar label + calib format.

Writes, under ``<out>/<seq>/``:
  info_label/<idx>_<idx>.txt   — one file per pair, K-Radar v1_0 label format
  info_calib/calib_radar_lidar.txt — radar->lidar planar (x,y) offset from our calibration

K-Radar v1_0 label format (matched to datasets/kradar_detection_v2_0.py::get_label):
  header : ``* idx(tesseract_os2-64_cam-front_os1-128_cam-lrr)=r_l_cf_l128_cr, timestamp=<t>``
  object : ``*, idx_p, idx_b4, Class, x, y, z, theta_DEG, l/2, w/2, h/2``
    - theta is in DEGREES (loader multiplies by pi/180),
    - dimensions are HALF-extents (loader does l = 2*value),
    - separator is ", " (comma-space).
Frame: our sensor frame (X fwd, Y left, Z up) already matches K-Radar's lidar frame.

NOTE: this exports LABELS + CALIB only. The point clouds (ldr64) and sparse radar
(rdr_sparse ``sprdr_<idx>.npy``) still need exporting with MATCHING per-frame indices
before the K-Radar dataloader can train on it (see --help / project notes).

    python code/sync/src/camera_projection/export_kradar.py -d 2026.05.08/17-34-27 \
        --annotations res/.../annotations/yolo.json --out /path/to/kradar_export --seq 100
"""
from __future__ import annotations

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

# our label -> K-Radar class name (K-Radar detection classes)
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


if __name__ == "__main__":
    raise SystemExit(main())
