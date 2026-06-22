# Camera Synchronization + Projection

Standalone tools for attaching Intel D435 frames to existing radar/lidar sync pairs and
projecting camera detections into the existing `sync_annotations.py` top-down format.

The expected top-down frame matches the current sync visualizers:

- `forward` = sensor/lidar X in meters
- `lateral` = sensor/lidar Y in meters
- units = meters

## 1. Synchronize Camera Frames

Use real frame timestamps when available:

```bash
python code/sync/camera_projection/sync_camera.py \
  --sync_csv code/sync/res/2026.05.10/18-05-08/sync_pairs.csv \
  --camera_timestamps /path/to/d435_timestamps.csv \
  --fit_offset \
  --max_delta_ms 50 \
  --output_csv code/sync/res/2026.05.10/18-05-08/camera_sync_pairs.csv \
  --output_json code/sync/res/2026.05.10/18-05-08/camera_sync_summary.json
```

If a video starts at a known Unix time:

```bash
python code/sync/camera_projection/sync_camera.py \
  --sync_csv code/sync/res/2026.05.10/18-05-08/sync_pairs.csv \
  --camera_start_time_s 1778462894.12 \
  --camera_fps 30 \
  --camera_frame_count 24000 \
  --fit_offset \
  --output_csv code/sync/res/2026.05.10/18-05-08/camera_sync_pairs.csv
```

Supported timestamp inputs: CSV, JSON, TXT, NPY, NPZ. CSV columns can be named
`timestamp`, `camera_t`, `time`, or `t`, with optional `frame_idx` / `camera_idx`.

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

## 3. Generate Initial Proposals From Raw Depth

Use this when the data has **not** been labeled yet. It reads synced D435 depth frames,
projects them to the shared top-down frame, clusters occupied cells, and writes initial
unlabeled boxes with `label: "object"`.

```bash
python code/sync/camera_projection/generate_annotations.py \
  --camera_sync_csv code/sync/res/2026.05.10/18-05-08/camera_sync_pairs.csv \
  --calibration_json /path/to/d435_to_lidar_calibration.json \
  --depth_dir /path/to/depth_frames \
  --depth_pattern 'depth_{camera_idx:06d}.npy' \
  --cache_dir code/sync/res/2026.05.10/18-05-08/camera_projection_cache \
  --scene_lateral -15 15 \
  --scene_forward 0.5 30 \
  --output_annotations code/sync/res/2026.05.10/18-05-08/annotations.generated.json
```

These are object proposals, not semantic labels. They are meant as a fast first pass
for manual review or later RGB-based labeling.

## 4. Project External Detections

Detection JSON can be a list, COCO-like `annotations`, or `frames[].detections`. Each
detection needs a frame id (`camera_idx`, `frame_idx`, or `image_id`) and `bbox`.

```bash
python code/sync/camera_projection/project_detections.py \
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

## 5. Cache Behavior

`project_detections.py` and `generate_annotations.py` keep a disk cache by default:

- default cache path: `<output_annotations parent>/camera_projection_cache`
- cached item for external detections: projected points for each detection bbox
- cached item for raw generation: projected points for each depth frame
- not cached as final boxes, so post-processing stays flexible
- cache hits do not load raw depth arrays again; the depth file is only stat-checked for
  automatic invalidation

This means you can rerun with different values for:

- `--score_threshold`
- `--min_depth_m` / `--max_depth_m`
- `--min_points`
- `--box_percentiles`

without re-reading raw depth maps for detections already in the cache.

Useful cache controls:

```bash
# Use an explicit cache location
--cache_dir code/sync/res/2026.05.10/18-05-08/camera_projection_cache

# Force recompute from raw depth and overwrite cache entries
--refresh_cache

# Disable disk cache for a one-off run
--no_cache
```

Cache keys include the calibration file hash, depth file path/size/mtime, bbox, bbox
format, camera index, pair index, and detection id. If any of those change, the cache
misses automatically and recomputes that detection.
