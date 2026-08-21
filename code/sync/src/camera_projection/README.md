# Camera Synchronization + Projection

Standalone tools for attaching Intel D435 frames to existing radar/lidar sync pairs and
projecting camera detections into the existing `sync_annotations.py` top-down format.

The expected top-down frame matches the current sync visualizers:

- `forward` = sensor/lidar X in meters
- `lateral` = sensor/lidar Y in meters
- units = meters

---

## 1. Synchronize Camera Frames

Reads timestamps directly from the D435 ROS bag — no pre-extraction needed.
Mirrors `sync_radar_lidar.py`: estimates a clock offset (or full affine skew) before
nearest-neighbor matching against lidar timestamps.

```bash
python code/sync/src/sync_camera_pairs.py \
  -d 2026.05.10/18-05-08 \
  --fit_offset \
  --max_delta_ms 50
```

Or with explicit paths:

```bash
python code/sync/src/sync_camera_pairs.py \
  --sync_csv  code/sync/res/2026.05.10/18-05-08/sync_pairs.csv \
  --camera_bag /path/to/d435.bag \
  --fit_offset \
  --max_delta_ms 50 \
  --output_csv  code/sync/res/2026.05.10/18-05-08/camera_sync_pairs.csv \
  --output_json code/sync/res/2026.05.10/18-05-08/camera_sync_summary.json
```

Options:
- `--target_time lidar_t` (default) or `--radar_t` — which clock to match against
- `--fit_offset` — estimate constant camera↔lidar clock offset
- `--fit_skew` — estimate full affine map (scale + offset)
- `--rebuild_index` — rescan bag and rebuild the timestamp cache

The first run scans the bag to index header timestamps and caches them next to the CSV.
Subsequent runs use the cache and skip the bag scan.

---

## 2. Calibration JSON

Projection needs D435 intrinsics and a camera-to-sensor transform. The transform should
map D435 camera points into the same Ouster-style sensor frame used by the BEV visualizer:
X forward, Y lateral, Z up.

```json
{
  "intrinsics": {
    "fx": 615.0,
    "fy": 615.0,
    "cx": 424.0,
    "cy": 240.0
  },
  "depth_scale": 0.001,
  "camera_to_lidar": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

Replace the identity matrix with your measured extrinsic calibration before trusting
the annotations.

---

## 3. Generate Initial Proposals From Raw Depth

Reads depth frames directly from the ROS bag — no pre-extraction needed.
Scans the depth topic once, projects each needed frame, and writes initial
unlabeled boxes with `label: "object"`.

```bash
python code/sync/src/camera_projection/generate_annotations.py \
  --camera_sync_csv code/sync/res/2026.05.10/18-05-08/camera_sync_pairs.csv \
  --calibration_json /path/to/d435_to_lidar_calibration.json \
  --camera_bag /path/to/d435.bag \
  --cache_dir code/sync/res/2026.05.10/18-05-08/camera_projection_cache \
  --scene_lateral -15 15 \
  --scene_forward 0.5 30 \
  --output_annotations code/sync/res/2026.05.10/18-05-08/annotations.generated.json
```

Pre-extracted depth files are also accepted (legacy):

```bash
  --depth_dir /path/to/depth_frames \
  --depth_pattern 'depth_{camera_idx:06d}.npy'
```

These are object proposals, not semantic labels. They are meant as a fast first pass
for manual review or later RGB-based labeling.

---

## 4. Project External Detections

Detection JSON can be a list, COCO-like `annotations`, or `frames[].detections`. Each
detection needs a frame id (`camera_idx`, `frame_idx`, or `image_id`) and `bbox`.

```bash
python code/sync/src/camera_projection/project_detections.py \
  --camera_sync_csv code/sync/res/2026.05.10/18-05-08/camera_sync_pairs.csv \
  --detections_json /path/to/d435_detections.json \
  --calibration_json /path/to/d435_to_lidar_calibration.json \
  --depth_dir /path/to/depth_frames \
  --depth_pattern 'depth_{camera_idx:06d}.npy' \
  --cache_dir code/sync/res/2026.05.10/18-05-08/camera_projection_cache \
  --output_annotations code/sync/res/2026.05.10/18-05-08/annotations.camera.json
```

Depth can be `.npy`, `.npz`, or image files. If no depth map is provided, a detection
may include `depth_m`; the tool then creates a coarse box around the projected center.

---

## 5. Cache Behavior

`project_detections.py` and `generate_annotations.py` keep a disk cache by default:

- default cache path: `<output_annotations parent>/camera_projection_cache`
- cached item: projected point arrays (`forward`, `lateral`, `depth_m`) per frame
- cache keys for bag frames: bag path/size/mtime + depth topic + frame index + calibration hash
- cache keys for file frames: depth file path/size/mtime + calibration hash
- not cached as final boxes, so post-processing stays flexible

This means you can rerun with different values for:

- `--score_threshold`
- `--min_depth_m` / `--max_depth_m`
- `--min_points`
- `--box_percentiles`

without re-reading raw depth data for frames already in the cache.

Useful cache controls:

```bash
# Force recompute from raw depth and overwrite cache entries
--refresh_cache

# Disable disk cache for a one-off run
--no_cache
```

---

## Typical end-to-end workflow

```bash
# 1. Sync radar + lidar
python code/sync/src/sync_radar_lidar.py -d 2026.05.10/18-05-08 --fit_offset

# 2. Sync camera to lidar (reads directly from .bag)
python code/sync/src/sync_camera_pairs.py -d 2026.05.10/18-05-08 --fit_offset

# 3. Generate depth-cluster proposals (reads depth directly from .bag)
python code/sync/src/camera_projection/generate_annotations.py \
  --camera_sync_csv code/sync/res/2026.05.10/18-05-08/camera_sync_pairs.csv \
  --calibration_json /path/to/calibration.json \
  --camera_bag /path/to/d435.bag \
  --output_annotations code/sync/res/2026.05.10/18-05-08/annotations.generated.json
```
