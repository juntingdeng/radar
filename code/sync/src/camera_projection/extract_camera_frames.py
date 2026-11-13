"""Extract a RealSense camera .bag to image files + timestamps + intrinsics, so the sync / calibration /
detection pipeline can run WITHOUT re-reading the bag. Lidar (.pcap) and radar (.h5) are left as raw.

ONE sequential pass over the bag (Reader.messages), NOT per-frame seeks -- this is both the space win
(uncompressed RGB -> JPEG ~10x) and the fix for the random-seek stall that made per-frame bag reads crawl.

Color  rgb8   -> JPEG  (quality configurable; ~0.9MB raw -> ~60-90KB)   color/000000.jpg ...
Depth  16UC1  -> 16-bit PNG, LOSSLESS (preserves mm)                    depth/000000.png ...
camera_index.npz   color_ts/depth_ts (+ *_bag_ts_ns) + topics + bag_signature  (the pipeline's index)
camera_info.json   fx/fy/cx/cy/width/height/distortion per stream              (calibration / projection)
manifest.json      formats, counts, source bag signature, extractor version

Frame i in color/ aligns with index i of color_ts / color_bag_ts_ns (same per-topic order as build_camera_index),
so camera_color_idx addressing is identical to the bag path. Read back with lib.camera_io.read_extracted_image.

Usage:
  python3 sync/src/camera_projection/extract_camera_frames.py --bag <cam.bag> --out <dir> [--jpeg_q 95] [--limit N]
"""
import os, sys, json, time, argparse
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # sync/src  (for `lib`)
from lib.camera_io import (DEFAULT_COLOR_TOPIC, DEFAULT_DEPTH_TOPIC, parse_ros1_image,
                           parse_ros1_image_header, read_camera_infos, bag_file_signature, _require_rosbags)
from pathlib import Path

EXTRACT_VERSION = "extract-1"

def extract(bag, out, color_topic=DEFAULT_COLOR_TOPIC, depth_topic=DEFAULT_DEPTH_TOPIC,
            jpeg_q=95, limit=None, progress_every=2000, camera_info=None, resume=False):
    out = Path(out); (out/"color").mkdir(parents=True, exist_ok=True); (out/"depth").mkdir(parents=True, exist_ok=True)
    # resume: skip re-encoding frames already on disk (the bag is still streamed sequentially to that point, but the
    # expensive decode+encode+write is skipped). Redo the last existing frame of each stream in case it was partial.
    resume_c = resume_d = 0
    if resume:
        resume_c = max(0, len(list((out/"color").glob("*.jpg"))) - 1)
        resume_d = max(0, len(list((out/"depth").glob("*.png"))) - 1)
        print(f"resume: skipping color<{resume_c}, depth<{resume_d} (already on disk)", flush=True)
    # intrinsics (calib / projection). read_camera_infos uses AnyReader = a SECOND slow open; on a slow USB drive
    # that alone can take >5 min, so --camera_info lets you supply an identical-rig camera_info.json and skip it.
    if camera_info:
        infos = json.load(open(camera_info)); print(f"camera_info: reused from {camera_info} ({list(infos.keys())})", flush=True)
    else:
        try:
            infos = read_camera_infos(bag); print(f"camera_info: {list(infos.keys())}", flush=True)
        except Exception as e:
            infos = {}; print(f"warn: no camera_info ({e})", flush=True)
    json.dump(infos, open(out/"camera_info.json", "w"), indent=2)

    Reader = _require_rosbags()
    cts, cbt, dts, dbt = [], [], [], []
    ci = di = 0; t0 = time.time()
    with Reader(Path(bag)) as reader:
        conns = [c for c in reader.connections if c.topic in (color_topic, depth_topic)]
        if not conns:
            raise KeyError(f"neither color nor depth topic in bag: {color_topic}, {depth_topic}")
        for conn, ts, raw in reader.messages(connections=conns):
            if conn.topic == color_topic:
                if ci < resume_c:                                                       # already on disk -> ts only
                    hdr_t, *_ = parse_ros1_image_header(bytes(raw)); cts.append(hdr_t); cbt.append(int(ts)); ci += 1
                else:
                    img = parse_ros1_image(bytes(raw)); arr = img.data
                    if img.encoding == "rgb8": arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)   # cv2 writes BGR
                    cv2.imwrite(str(out/"color"/f"{ci:06d}.jpg"), arr, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])
                    cts.append(img.header_t); cbt.append(int(ts)); ci += 1
            elif conn.topic == depth_topic:
                if di < resume_d:
                    hdr_t, *_ = parse_ros1_image_header(bytes(raw)); dts.append(hdr_t); dbt.append(int(ts)); di += 1
                else:
                    img = parse_ros1_image(bytes(raw))
                    cv2.imwrite(str(out/"depth"/f"{di:06d}.png"), img.data)                # uint16 -> 16-bit PNG
                    dts.append(img.header_t); dbt.append(int(ts)); di += 1
            if progress_every and (ci+di) % progress_every == 0:
                el = time.time()-t0
                print(f"  {ci} color + {di} depth  ({el/max(1,ci+di)*1000:.1f} ms/frame, {el:.0f}s)", flush=True)
            if limit and ci >= limit and di >= limit: break

    np.savez_compressed(out/"camera_index.npz",
        color_ts=np.asarray(cts, np.float64), depth_ts=np.asarray(dts, np.float64),
        color_bag_ts_ns=np.asarray(cbt, np.int64), depth_bag_ts_ns=np.asarray(dbt, np.int64),
        color_topic=np.asarray(color_topic), depth_topic=np.asarray(depth_topic),
        bag_signature=json.dumps(bag_file_signature(bag)), index_version=np.asarray(EXTRACT_VERSION))
    json.dump(dict(n_color=ci, n_depth=di, color_format=f"jpeg_q{jpeg_q}", depth_format="png16_mm",
        color_topic=color_topic, depth_topic=depth_topic, camera_info_streams=list(infos.keys()),
        source=bag_file_signature(bag), extract_version=EXTRACT_VERSION),
        open(out/"manifest.json", "w"), indent=2)
    print(f"done: {ci} color + {di} depth frames -> {out}  ({time.time()-t0:.0f}s)", flush=True)
    return ci, di

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--color_topic", default=DEFAULT_COLOR_TOPIC)
    ap.add_argument("--depth_topic", default=DEFAULT_DEPTH_TOPIC)
    ap.add_argument("--jpeg_q", type=int, default=95)
    ap.add_argument("--limit", type=int, default=None, help="extract only first N of each stream (test)")
    ap.add_argument("--camera_info", default=None,
                    help="reuse an existing camera_info.json (same-rig D435) instead of AnyReader (avoids a slow 2nd open)")
    ap.add_argument("--resume", action="store_true", help="skip frames already written to <out> (restart-safe)")
    a = ap.parse_args()
    extract(a.bag, a.out, a.color_topic, a.depth_topic, a.jpeg_q, a.limit, camera_info=a.camera_info, resume=a.resume)
