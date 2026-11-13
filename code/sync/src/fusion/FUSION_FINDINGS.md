# Radar-first proposal — fusion R&D findings

**Goal.** Raise object proposals *from radar* when lidar and camera are weak (night, or
blinded by oncoming headlights — optical sensors saturate/see nothing, mmWave is
unaffected), then **cross-check** the radar proposals against the (weak but available)
lidar and camera. The hard part: distinguish real moving targets (car, person) from
static background and radar noise.

This is the mirror of the production pipeline (`yolo_to_lidar.py`: camera/lidar propose →
radar cross-check). All work was prototyped on one segment of the **2026.05.11 / 21.37.07**
night capture (pairs 13850–13900, a slow intersection with traffic). Figures: `../../res/2026.05.11/21.37.07/fusion/fusion_step*.png`.

## Status: feasibility mapped, NOT a working detector
The architecture is sound and the individual primitives work, but a *trustworthy* radar-first
detector is a real engineering build, not a notebook. Each prototype pass exposed the next
flaw (see "What failed"). **Do not treat `radar_propose_pipeline.py` output as correct.**

## What works (reusable primitives, in this folder)
- **`lidar_odometry.py`** — scan-to-scan point-to-plane ICP (Open3D) → ego poses → accumulated
  world cloud. Fitness ~0.96 on good segments. (step 1)
- **`ego_speed_scan.py`** — quick ego-speed probe across a window (found this window's ego
  crawls at ~0–0.8 m/s). (used to pick segments)
- **`lidar_dynamic_separation.py`** — world-frame voxel-temporal-persistence static/dynamic
  split. **Requires real odometry** — with identity poses it flags ~100% dynamic; with ICP
  poses it drops to ~4%. Ground removal + near-range + short window are essential. (steps 3, 12)
- **`cfar_range_doppler_detector.py`** — **the key insight.** Detect in the **range-Doppler**
  map (CFAR over range×velocity), NOT the range-azimuth BEV. A moving target is a *point* in
  range-velocity with a per-detection velocity; the BEV smears it into azimuth *arcs* that
  flood any threshold (59k "peaks" → useless). Gives discrete targets with range + azimuth +
  radial velocity. (step 7)
- **`project_radar_to_camera_lidar.py`** — radar (range,az) → radar XY → lidar frame
  (`radar_to_lidar_auto`) → camera (`inv(camera_to_lidar)` + color intrinsics; height borrowed
  from nearby lidar). Puts radar/lidar/camera in one coordinate system. (steps 8, 13)
- **`azimuth_validation.py`** — compares CFAR detections to the validated viz_labels radar
  peaks; this is how the azimuth-sign bug was caught.

## Bugs found (both real, both fixed in the saved primitives)
1. **`compute_doppler_resolution` is broken** in `utils/capon.py` (called by
   `utils/parse_config.py`): it expects bandwidth in **MHz** but is given **Hz**, so the center
   frequency blows up and `radar.doppler_resolution` computes to **0**. Invisible until now
   because nothing in the pipeline used velocity. Correct value here: ~0.305 m/s. **Fix in
   `parse_config.py` before anyone wires velocity in.**
2. **Azimuth sign flip.** `bev_render.compute_range_azimuth` does an `fliplr` on the angle axis,
   and `radar_to_lidar_auto` was solved against that convention. Raw per-detection beamforming
   (`angs[argmax(|sv·x|²)]`) does NOT flip → every radar detection was **mirrored laterally**
   (a target at Y=+8 projected to Y=−8). Caught via `azimuth_validation.py` (flipped matches the
   validated peaks at 0.0 m, as-is at 0.4 m). Fix: **negate the azimuth** (`np.radians(-az)`),
   applied in the saved `cfar_*` / `radar_propose_pipeline` / `project_*` primitives.

## Physics limits established
- **Velocity aliasing.** TDM MIMO (3 TX) over 16 Doppler loops at 390 µs spacing →
  `v_max ≈ ±2.4 m/s` (±7.3 with MIMO disambiguation). A 25 mph road is ~11 m/s → **every
  vehicle Doppler wraps**. Single-frame Doppler velocity is unusable on real roads.
  → **True speed must come from cross-frame range-rate** (Δrange/Δt is unaliased).
- **Radar angular blur.** Few antennas → coarse azimuth; radar owns range+velocity (precise),
  lidar owns bearing+shape. Fuse by **range+velocity**, not XY.

## What failed (and why — the unsolved core)
- **BEV percentile "peaks" as a detector** (steps 5–6): the range-azimuth BEV thresholds into
  azimuth arcs that flood the scene (tens of thousands of points); world-frame static/dynamic
  separation on them doesn't converge. → replaced by range-Doppler CFAR.
- **Single-frame Doppler velocity** (step 8): aliased garbage (read 1.6 m/s for an ~11 m/s car).
- **Greedy tracker + straightness heuristic** (steps 9–12): detection jitter chains into
  wandering "tracks" scored as slow movers; too crude.
- **Cross-check vs lidar *presence*** (step 10): over-confirms — a radar proposal near any static
  structure finds lidar and is wrongly confirmed. Must cross-check vs lidar *dynamic* content.
- **The residual failure (step 13, even after fixes):** static objects (a traffic light) get a
  **false residual world-velocity from odometry yaw drift** and pass the mover gate, while the
  real moving car is not cleanly tracked. ICP scan-to-scan drift + radar clutter + heuristic
  gating = confirmed artifacts, missed real movers.

## What a real build needs (scoped module, days — not a notebook)
1. **Yaw-accurate odometry / SLAM** (ICP scan-to-scan drifts; the dynamic separation and
   ego-compensation both depend on clean poses).
2. **Proper multi-target tracker** (Kalman/IMM + gating + M-of-N confirmation) so jitter never
   forms a track and static objects read zero world-velocity.
3. **Ego-Doppler consistency** gating (predicted static Doppler from ego velocity) + range-rate
   for unaliased speed; use the aliased per-frame Doppler only as an alias-constraint/association
   feature.
4. **Cross-check vs lidar *dynamic* clusters + camera features** (not lidar presence), with
   micro-Doppler for car/person and geometric plausibility for multipath-ghost rejection.
5. **Ground truth** to tune thresholds (the premise — lidar/camera weak — means no easy label).

## Figure index (`../../res/2026.05.11/21.37.07/fusion/`)
| step | file | result |
|---|---|---|
| 1 | fusion_step1_odometry | lidar odometry → world map ✅ |
| 3 | fusion_step3_dynamic(_v2) | static/dynamic separation (needs odometry) ✅/⚠️ |
| 5 | fusion_step5_radar_confirm(_vehicles) | BEV-peak cross-check — noise flood ❌ |
| 6 | fusion_step6_radar_first | world-frame separation on BEV peaks ❌ |
| 7 | fusion_step7_cfar_proto | **range-Doppler CFAR detector** ✅ |
| 8 | fusion_step8_project | CFAR → lidar+camera (aliased velocity) ⚠️ |
| 9 | fusion_step9_radar_only | radar-only track + range-rate (unaliased speed) ⚠️ |
| 10 | fusion_step10_radar_propose_crosscheck | vs lidar presence — over-confirms ❌ |
| 11–12 | fusion_step11/12_radar_redo(2) | +motion-plausibility +dynamic-lidar (odometry-backed) ⚠️ |
| 13 | fusion_step13_project_cam | final projection — still confirms artifacts ❌ |
| — | fusion_diag_azimuth | azimuth-sign bug diagnosis ✅ |

Legend: ✅ works · ⚠️ works with caveats · ❌ dead end / wrong result.
