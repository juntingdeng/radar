# Radar/Lidar Time Sync

This folder contains scripts to synchronize:
- TI AWR2944 + DCA1000 radar capture (`.h5`, from `collect.py`)
- Ouster OS0 lidar capture (`.pcap`)

## Scripts

- `extract_timestamps.py`  
  Extract packet/frame timestamps and save to `.npz` for debugging.

- `sync_radar_lidar.py`  
  Build nearest-neighbor frame matches and save:
  - `sync_pairs.csv` (frame pairs)
  - `sync_summary.json` (offset and error stats)

- `visualize_sync.py`  
  Timing/sync quality plots (timelines, index mapping, delta_ms). Use `--cfg_file` so
  `packets_per_frame` matches the sync run.

- `visualize_sync_video.py`  
  **Data inspection video**: radar range-azimuth heatmap + lidar BEV point cloud per
  matched pair, streamed directly to MP4 (no per-frame PNG dump).

- `sync_utils.py`  
  Shared timestamp parsing and matching helpers.

- `camera_compat.py`  
  Intel RealSense R435 ROS bag reader (RGB + depth `sensor_msgs/Image`).

- `sync_camera_pairs.py`  
  Map camera frames onto existing `sync_pairs.csv` rows via `radar_t`.

- `camera_frame_testbench.ipynb`  
  Interactive RGB + depth viewer keyed by sync pair index.

## Basic Usage

```bash
python code/sync/sync_radar_lidar.py \
  --radar_h5 /path/to/radar.h5 \
  --lidar_pcap /path/to/lidar.pcap \
  --fit_offset \
  --output_csv ./code/sync/sync_pairs.csv \
  --output_json ./code/sync/sync_summary.json
```

## Visualize Sync Results

```bash
python code/sync/visualize_sync.py \
  --sync_csv ./code/sync/sync_pairs.csv \
  --radar_h5 /path/to/radar.h5 \
  --lidar_pcap /path/to/lidar.pcap \
  --estimated_offset_s 0.0 \
  --out_png ./code/sync/sync_visualization.png
```

Tip: if `sync_summary.json` reports a non-zero `estimated_lidar_to_radar_offset_s`,
pass that value to `--estimated_offset_s` for accurate aligned timeline display.

This command saves a **single summary PNG** that already includes **all matched pairs**.

To export one PNG per synchronized pair:

```bash
python code/sync/visualize_sync.py \
  --sync_csv ./code/sync/sync_pairs.csv \
  --radar_h5 /path/to/radar.h5 \
  --lidar_pcap /path/to/lidar.pcap \
  --estimated_offset_s 0.0 \
  --export_all_pairs_dir ./code/sync/pair_inspect \
  --export_max_pairs -1 \
  --no_show
```

To create an MP4 from all exported pair PNGs in the same run:

```bash
python code/sync/visualize_sync.py \
  --sync_csv ./code/sync/sync_pairs.csv \
  --radar_h5 /path/to/radar.h5 \
  --lidar_pcap /path/to/lidar.pcap \
  --estimated_offset_s 0.0 \
  --export_all_pairs_dir ./code/sync/pair_inspect \
  --export_max_pairs -1 \
  --out_video ./code/sync/pair_inspect.mp4 \
  --video_fps 12 \
  --no_show
```

Note: video export uses `ffmpeg` from your system `PATH`.

## Data video (range-azimuth + lidar BEV)

Requires Ouster SDK 0.16+ (`pip install ouster-sdk`). Uses `ouster.sdk.core` and
`ouster.sdk.pcap` (not the old `from ouster import client, pcap`). Metadata JSON is
optional if it sits beside the `.pcap` with a matching prefix.

```bash
python code/sync/visualize_sync_video.py \
  --sync_csv ./code/sync/res/sync_pairs.csv \
  --sync_summary ./code/sync/res/sync_summary.json \
  --radar_h5 /path/to/radar.h5 \
  --lidar_pcap /path/to/lidar.pcap \
  --lidar_metadata /path/to/ouster_metadata.json \
  --cfg_file ./code/mmWaveStudio/server.lua \
  --out_video ./code/sync/res/sync_data.mp4 \
  --video_fps 10 \
  --stride 2
```

**Important:** `radar_idx` in `sync_pairs.csv` is a **sync-time block index** (small
`packets_per_frame` used during sync), **not** the ADC frame index from `server.lua`.
The video script maps via `radar_t` timestamps by default (`--radar_load_mode timestamp`).
If you see many "Skip pair" messages, pass `--sync_summary` (or re-run sync to add
`radar_packets_per_frame` to the JSON).

- `--stride 2` uses every 2nd matched pair (faster preview).
- `--max_frames 500` caps video length.
- `--show_radar_bev` shows radar BEV instead of range-azimuth on the left panel.

## Optional: Ouster SDK Accurate Scan Timestamps

If you have Ouster metadata `.json` and SDK installed, pass:

```bash
--lidar_metadata /path/to/ouster_metadata.json
```

Then lidar frame timestamps use native scan timestamps instead of packet-block fallback.

## Notes

- By default, fallback lidar parsing filters UDP port `7502` (OS0 lidar data).
- If your port differs, set `--lidar_udp_port`.
- If your capture has no clean packet-per-frame ratio, tune:
  - `--radar_packets_per_frame`
  - `--lidar_packets_per_frame`
- `--fit_offset` is recommended when radar/lidar logs start at slightly different times.

## Camera (Intel RealSense R435)

Camera recordings are ROS bag files under `{data_root}/camera/*.bag` with topics:

- `/device_0/sensor_1/Color_0/image/data` (rgb8)
- `/device_0/sensor_0/Depth_0/image/data` (16UC1, mm)

Requires: `pip install rosbags`

After radar/lidar sync, attach camera frames (uses image header timestamps vs `radar_t`):

```bash
cd code
python3 ./sync/sync_camera_pairs.py -d 2026.05.10/18-05-08 --rebuild_index
```

This adds `camera_color_idx`, `camera_depth_idx`, `camera_t`, `camera_delta_ms` to
`sync_pairs.csv` and caches timestamps in `camera_index.npz`. First bag scan on USB
can take several minutes.

Interactive viewer: open `sync/camera_frame_testbench.ipynb`.
