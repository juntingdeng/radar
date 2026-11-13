"""One-time YOLO 2D-detection cache for the whole camera bag.

The camera ``.bag`` streams from Google Drive, and RANDOM frame access is slow (per-frame
chunk seeks). But a SINGLE SEQUENTIAL sweep of the color topic reads every frame without
seeking -- bandwidth-bound, not seek-bound -- and yolov8n on ~15k frames is only minutes of
compute. So we pay the camera+YOLO cost once here, caching each frame's 2D boxes to JSON, and
then ``yolo_to_lidar.py --yolo_cache`` reuses them with ZERO Drive I/O for all the
lidar/radar labeling, box padding, and calibration adjustments (which only touch local
lidar + radar).

Cache format (``<out>``):
    {"meta": {...}, "dets": {"<color_frame_idx>": [{"xyxy":[x0,y0,x1,y1],
                                                     "label":"person","score":0.83}, ...]}}
Keyed by the color frame index = the sync CSV's ``camera_color_idx`` (the sequential
enumeration order of the color topic, same index ``_extract_frames`` uses).

Modes:
  * default: sequential bag sweep (production; streams the 114 GB bag once). Resumable --
    re-running seeks past already-cached frames via camera_index.npz.
  * ``--from_local_cache``: run YOLO on the already-extracted ``color_*.npy`` frames in the
    frame cache dir (NO Drive) -- used to test the pipeline and to (re)cache frames you
    already have.

    # production one-time sweep:
    python code/sync/src/camera_projection/cache_yolo_detections.py -d 2026.05.08/17-34-27
    # test / cache-only-what's-local:
    python .../cache_yolo_detections.py -d 2026.05.08/17-34-27 --from_local_cache
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

_THIS_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _THIS_DIR.parent
_CODE_ROOT = _SYNC_DIR.parent
for _p in (_SYNC_DIR, _CODE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402
from lib.camera_io import DEFAULT_COLOR_TOPIC, iter_topic_messages, parse_ros1_image  # noqa: E402

# Same default classes as yolo_to_lidar; cache them ALL at a low conf so reuse can filter up.
DEFAULT_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arguments(p)
    p.add_argument("--camera_bag", default=None)
    p.add_argument("--camera_index", default=None)
    p.add_argument("--camera_sync_csv", default=None)
    p.add_argument("--sync_only", action="store_true",
                   help="Only YOLO the color frames referenced by sync pairs (camera_color_idx) -- "
                        "the complete labelable set. Non-pair frames are read past (fast) but not "
                        "inferred, and the sweep stops after the last pair frame. ~8x less YOLO work.")
    p.add_argument("--out", default=None, help="Cache JSON path (default res/<ds>/yolo_dets_cache.json).")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model.")
    p.add_argument("--conf", type=float, default=0.05,
                   help="Cache-time confidence floor (LOW so reuse can filter to any higher conf).")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    p.add_argument("--from_local_cache", action="store_true",
                   help="Run YOLO on already-extracted color_*.npy frames (no Drive) instead of "
                        "sweeping the bag. For testing / caching only locally-present frames.")
    p.add_argument("--frame_cache", default=None, help="Local color_*.npy dir (default the hashed cache).")
    p.add_argument("--flush_every", type=int, default=200, help="Write the JSON every N frames.")
    p.add_argument("--limit", type=int, default=0, help="Stop after N frames (0 = all). Testing.")
    p.add_argument("--overwrite", action="store_true", help="Ignore any existing cache and restart.")
    return p


def _out_path(args) -> Path:
    if args.out:
        return Path(args.out)
    return Path(f"res/{args.dataset}/yolo_dets_cache.json")


def _frame_cache_dir(args) -> Path:
    import hashlib
    if args.frame_cache:
        return Path(args.frame_cache)
    tag = hashlib.md5(str(Path(args.camera_bag).resolve()).encode()).hexdigest()[:10]
    return Path.home() / ".cache" / "ti_setup_frames" / tag


def _load_existing(path: Path, overwrite: bool):
    if overwrite or not path.is_file():
        return {}
    try:
        d = json.load(open(path))
        return {int(k): v for k, v in d.get("dets", {}).items()}
    except Exception:
        return {}


def _predict(model, rgb, keep_ids, conf) -> List[dict]:
    """rgb ndarray -> list of {xyxy,label,score}. ultralytics wants BGR -> flip."""
    res = model.predict(rgb[..., ::-1], conf=conf, classes=keep_ids, verbose=False)[0]
    out = []
    for b in res.boxes:
        cls = int(b.cls[0])
        # full precision (no rounding) so cache-reuse reproduces live YOLO exactly.
        out.append({"xyxy": [float(v) for v in b.xyxy[0].tolist()],
                    "label": model.names.get(cls, str(cls)), "score": float(b.conf[0])})
    return out


def _write(path: Path, meta: dict, dets: Dict[int, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    meta = {**meta, "n_frames": len(dets)}
    json.dump({"meta": meta, "dets": {str(k): dets[k] for k in sorted(dets)}}, open(tmp, "w"))
    os.replace(tmp, path)   # atomic -> a crash mid-write can't corrupt the cache


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())
    from ultralytics import YOLO

    out = _out_path(args)
    model = YOLO(args.model)
    name_to_id = {v: k for k, v in model.names.items()}
    keep_ids = [name_to_id[n] for n in args.classes if n in name_to_id]
    if not keep_ids:
        raise SystemExit("No valid --classes matched the model.")
    keep_names = {model.names[i] for i in keep_ids}
    print(f"model={args.model} conf={args.conf} classes={sorted(keep_names)}")

    # --sync_only: restrict YOLO to the color frames a sync pair references (the labelable set).
    pair_frames: Optional[set] = None
    max_pair_frame = None
    if args.sync_only:
        import csv
        with open(args.camera_sync_csv) as f:
            pair_frames = {int(r.get("camera_color_idx", r.get("camera_idx", -1)))
                           for r in csv.DictReader(f)}
        pair_frames.discard(-1)
        max_pair_frame = max(pair_frames)
        print(f"sync_only: {len(pair_frames)} distinct pair frames (max idx {max_pair_frame}).")

    dets = _load_existing(out, args.overwrite)
    if dets:
        print(f"resuming: {len(dets)} frame(s) already cached (max idx {max(dets)}).")
    meta = {"bag": args.camera_bag, "topic": DEFAULT_COLOR_TOPIC, "model": args.model,
            "conf": args.conf, "classes": sorted(keep_names), "source": None}

    n_new = 0
    if args.from_local_cache:
        meta["source"] = "local_cache"
        cdir = _frame_cache_dir(args)
        files = sorted(cdir.glob("color_*.npy"), key=lambda f: int(re.search(r"(\d+)", f.name).group(1)))
        print(f"from_local_cache: {len(files)} color_*.npy in {cdir}")
        for f in files:
            idx = int(re.search(r"(\d+)", f.name).group(1))
            if pair_frames is not None and idx not in pair_frames:
                continue
            if idx in dets and not args.overwrite:
                continue
            dets[idx] = _predict(model, np.load(f), keep_ids, args.conf)
            n_new += 1
            if args.limit and n_new >= args.limit:
                break
            if n_new % args.flush_every == 0:
                _write(out, meta, dets); print(f"  cached {n_new} new (idx {idx})...", flush=True)
    else:
        meta["source"] = "bag_sweep"
        # resume: seek past the last cached contiguous frame via camera_index timestamps.
        start_ns = None
        resume_from = 0
        if dets:
            resume_from = max(dets) + 1
            ipath = args.camera_index
            if ipath and Path(ipath).is_file():
                idx = np.load(ipath, allow_pickle=False)
                ts = idx["color_bag_ts_ns"] if "color_bag_ts_ns" in idx else None
                if ts is not None and resume_from < len(ts):
                    start_ns = int(ts[resume_from])
                    print(f"seeking bag to frame {resume_from} (start_ns={start_ns}).")
        print(f"sweeping color topic sequentially from frame {resume_from} ...", flush=True)
        for k, (_ts, raw) in enumerate(iter_topic_messages(args.camera_bag, DEFAULT_COLOR_TOPIC,
                                                           start_ns=start_ns)):
            idx = resume_from + k
            if max_pair_frame is not None and idx > max_pair_frame:
                print(f"  reached last pair frame ({max_pair_frame}); stopping.", flush=True)
                break
            if pair_frames is not None and idx not in pair_frames:
                continue   # non-pair frame: read past it, don't decode/YOLO
            if idx in dets and not args.overwrite:
                continue
            rgb = np.array(parse_ros1_image(raw).data, copy=True)
            dets[idx] = _predict(model, rgb, keep_ids, args.conf)
            n_new += 1
            if args.limit and n_new >= args.limit:
                break
            if n_new % args.flush_every == 0:
                _write(out, meta, dets); print(f"  cached {n_new} new (idx {idx})...", flush=True)

    _write(out, meta, dets)
    tot_boxes = sum(len(v) for v in dets.values())
    print(f"\nWrote {out}: {len(dets)} frames, {tot_boxes} boxes (+{n_new} new this run).")
    print("Reuse: yolo_to_lidar.py --yolo_cache " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
