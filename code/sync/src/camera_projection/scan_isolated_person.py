"""Scan the capture for ISOLATED-person frames (radar-calibration candidates).

Samples color frames across the capture (seeking via camera_index so it doesn't stream
the whole 113 GB bag), runs YOLO, and flags pairs with exactly one `person` and no
vehicles -- a clean, unambiguous moving radar target. Writes a CSV and prints contiguous
"walking-person" scenes you can hand to calibrate_radar_lidar.py.

    python code/sync/src/camera_projection/scan_isolated_person.py \
        -d 2026.05.08/17-34-27 --stride 30 --model yolov8n.pt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

_THIS_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _THIS_DIR.parent
for _p in (_SYNC_DIR, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402
from lib.camera_io import DEFAULT_COLOR_TOPIC, iter_topic_messages, parse_ros1_image  # noqa: E402

VEHICLES = ("bicycle", "car", "motorcycle", "bus", "truck")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--camera_bag", default=None)
    p.add_argument("--camera_index", default=None)
    p.add_argument("--camera_sync_csv", default=None)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--stride", type=int, default=30, help="Sample every Nth sync pair.")
    p.add_argument("--max_samples", type=int, default=800)
    p.add_argument("--pair_lo", type=int, default=0)
    p.add_argument("--pair_hi", type=int, default=-1)
    p.add_argument("--frame_cache", default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--min_person_conf", type=float, default=0.45,
                   help="Min confidence for the single person to count as isolated.")
    add_dataset_arguments(p)
    return p


def _cache_dir(args) -> Path:
    if args.frame_cache:
        base = Path(args.frame_cache)
    else:
        tag = hashlib.md5(str(Path(args.camera_bag).resolve()).encode()).hexdigest()[:10]
        base = Path.home() / ".cache" / "ti_setup_frames" / tag
    base.mkdir(parents=True, exist_ok=True)
    return base


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())

    rows: List[dict] = []
    with open(args.camera_sync_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    pair_hi = args.pair_hi if args.pair_hi >= 0 else len(rows)
    sample = [r for i, r in enumerate(rows)
              if args.pair_lo <= int(r["pair_idx"]) < pair_hi and i % args.stride == 0]
    if len(sample) > args.max_samples:
        keep = np.linspace(0, len(sample) - 1, args.max_samples).astype(int)
        sample = [sample[i] for i in keep]
    print(f"Sampling {len(sample)} frames (stride {args.stride}) across pairs "
          f"{args.pair_lo}..{pair_hi}", flush=True)

    # seek table
    color_bag_ts = None
    if args.camera_index and Path(args.camera_index).is_file():
        idx = np.load(args.camera_index, allow_pickle=False)
        color_bag_ts = idx.get("color_bag_ts_ns") if hasattr(idx, "get") else (
            idx["color_bag_ts_ns"] if "color_bag_ts_ns" in idx else None)
    if color_bag_ts is None:
        raise SystemExit("Need camera_index.npz with color_bag_ts_ns for seeking.")

    cache = _cache_dir(args)
    from ultralytics import YOLO
    model = YOLO(args.model)
    name_to_id = {v: k for k, v in model.names.items()}
    person_id = name_to_id["person"]
    veh_ids = {name_to_id[v] for v in VEHICLES if v in name_to_id}

    results: List[dict] = []
    for n, r in enumerate(sample):
        pid = int(r["pair_idx"])
        c_idx = int(r.get("camera_color_idx", r.get("camera_idx", -1)))
        if c_idx < 0 or c_idx >= len(color_bag_ts):
            continue
        fpath = cache / f"color_{c_idx:06d}.npy"
        if fpath.is_file():
            color = np.load(fpath)
        else:
            _ts, raw = next(iter(iter_topic_messages(
                args.camera_bag, DEFAULT_COLOR_TOPIC, start_ns=int(color_bag_ts[c_idx]))))
            color = np.array(parse_ros1_image(raw).data, copy=True)
            np.save(fpath, color)
        res = model.predict(color[..., ::-1], conf=args.conf, verbose=False)[0]
        cls = [int(b.cls[0]) for b in res.boxes]
        confs = [float(b.conf[0]) for b in res.boxes]
        persons = [(c, cf, b) for c, cf, b in zip(cls, confs, res.boxes) if c == person_id]
        n_person = len(persons)
        n_veh = sum(1 for c in cls if c in veh_ids)
        pcx = pcy = pconf = -1.0
        if n_person >= 1:
            best = max(persons, key=lambda t: t[1])
            x0, y0, x1, y1 = [float(v) for v in best[2].xyxy[0].tolist()]
            pcx, pcy, pconf = 0.5 * (x0 + x1), 0.5 * (y0 + y1), best[1]
        isolated = (n_person == 1 and n_veh == 0 and pconf >= args.min_person_conf)
        results.append(dict(pair_idx=pid, camera_color_idx=c_idx,
                            camera_t=r.get("camera_t", ""), n_person=n_person, n_vehicle=n_veh,
                            person_cx=round(pcx, 1), person_conf=round(pconf, 2),
                            isolated=int(isolated)))
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(sample)} scanned, {sum(x['isolated'] for x in results)} isolated so far", flush=True)

    out_csv = args.out_csv or str(Path(args.camera_sync_csv).parent / "isolated_person_scan.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    iso = [x for x in results if x["isolated"]]
    print(f"\nScanned {len(results)}, isolated-person frames: {len(iso)} -> {out_csv}")

    # group contiguous isolated hits (by sample order) into scenes
    scenes = []
    cur = []
    iso_pairs = sorted(x["pair_idx"] for x in iso)
    for pidx in iso_pairs:
        if cur and pidx - cur[-1] <= args.stride * 2:
            cur.append(pidx)
        else:
            if cur: scenes.append((cur[0], cur[-1], len(cur)))
            cur = [pidx]
    if cur: scenes.append((cur[0], cur[-1], len(cur)))
    print(f"\n=== isolated-person SCENES (pair ranges) — radar-cal candidates ===")
    for lo, hi, cnt in sorted(scenes, key=lambda s: -s[2]):
        print(f"  pairs {lo:5d} .. {hi:5d}   ({cnt} hits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
