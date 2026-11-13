# Archived one-off scripts

These were **one-time builders, plots, diagnostics, or superseded prototypes** — not part of the
active data → calibration → labeling → detection → fusion → memory → communication pipeline. They
are removed from the working tree to keep it clean, but remain **fully in git history** and are
retrievable at any time.

## Retrieve

Every file below exists in commit `ab82532`, also tagged **`archived-scripts-2026-11-13`**:

```bash
# one file back into place
git checkout archived-scripts-2026-11-13 -- code/sync/src/fusion/attn.py
# or just view it
git show archived-scripts-2026-11-13:code/sync/src/fusion/attn.py
# list everything that was in the tree at that point
git ls-tree -r archived-scripts-2026-11-13 --name-only | grep '\.py$'
```

## What was archived (removed in the commit after `ab82532`)

| path (relative to repo root) | what it was |
|---|---|
| `code/sync/dev/render.py` | preview/video renderers (`full`/`video`/`stride5`) — one-off visualization |
| `code/sync/dev/smoke_tools.py` | 10.24 smoke one-offs: hand-pick BEV + one-time benchmark add |
| `code/sync/src/fusion/attn.py` | BEV cross-attention fusion **prototype** (superseded by `late_fusion/`) |
| `code/sync/src/fusion/azimuth_validation.py` | radar azimuth-convention validation diagnostic |
| `code/sync/src/fusion/cfar_range_doppler_detector.py` | CFAR R-D detector prototype (superseded) |
| `code/sync/src/fusion/ego_speed_scan.py` | ego-speed estimation scan/diagnostic |
| `code/sync/src/fusion/lidar_dynamic_separation.py` | lidar static/dynamic separation experiment |
| `code/sync/src/fusion/lidar_odometry.py` | lidar odometry prototype (superseded by `window.py odometry`) |
| `code/sync/src/fusion/project_radar_to_camera_lidar.py` | radar→camera/lidar projection util (superseded) |
| `code/sync/src/fusion/radar_propose_pipeline.py` | early radar-propose pipeline (superseded by `radar_first/`) |
| `code/sync/src/fusion/radar_first/diag.py` | 11 merged radar-first diagnostic probes (plots) |
| `code/sync/src/camera_projection/calib_precision_check.py` | calibration-precision check/plot |
| `code/sync/src/camera_projection/scan_isolated_person.py` | one-off scan for isolated-person calib frames |
| `code/sync/src/camera_projection/cache_yolo_detections.py` | one-time YOLO 2D-detection cache builder |

The fusion-exploration record (why these prototypes were tried and dropped) lives in
`code/sync/src/fusion/FUSION_FINDINGS.md`, which is **kept**.
