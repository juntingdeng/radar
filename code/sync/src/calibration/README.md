# Sensor Extrinsic Calibration

Interactive tools to estimate the rigid transforms that place the D435 camera and
the TI radar into the **Ouster lidar sensor frame** (X forward, Y left, Z up) — the
same top-down frame used by the sync visualizers and `sync_annotations.py`.

Time sync (`sync_radar_lidar.py`, `sync_camera_pairs.py`) aligns the sensors in
*time*. These tools add the *spatial* alignment so camera/radar detections can be
projected into the shared BEV.

## Method

You hand-pick point correspondences between two sensors, and a rigid
[Umeyama/Kabsch](rigid_solve.py) least-squares fit recovers the transform
`target = R · source + t` (no scaling). Picking (GUI) is decoupled from solving
(headless), so correspondences are saved to JSON and can be re-solved anytime.

| Tool | Source → Target | DoF | Correspondence source |
|------|-----------------|-----|-----------------------|
| `calibrate_camera_lidar.py` | camera depth-optical → lidar | 3D rigid (6 DoF) | depth back-projection ↔ lidar XYZ |
| `calibrate_radar_lidar.py`  | radar top-down → lidar | planar (yaw + XY), Z supplied | radar BEV ↔ lidar BEV |

Radar has no reliable elevation, so its fit is planar; the vertical offset is
passed with `--z_offset_m` (measure once with a tape).

## Prerequisites

- A completed run of `sync_radar_lidar.py` **and** `sync_camera_pairs.py`
  (so `sync_pairs.csv` / `camera_sync_pairs.csv` exist).
- An interactive matplotlib backend (macOS/Qt/Tk) for picking; solving is headless.

The calibration JSON itself is **created automatically**: if it's missing, the
camera calibrator reads intrinsics + `depth_scale` from the bag `camera_info`
topics and writes the file (with an identity extrinsic) before you start picking.
No separate setup step.

## Two cases

- **First calibration of a rig** — run `calibrate_camera_lidar.py` /
  `calibrate_radar_lidar.py` to *solve* the extrinsic (interactive, below).
- **Same rig, already calibrated on another collection** — don't re-pick; copy
  the solved extrinsic with `--reuse_from` (intrinsics are re-read per collection
  unless you pass nothing, in which case they're copied too):

  ```bash
  python code/sync/src/calibration/calibrate_camera_lidar.py \
    -d 2026.05.08/17-34-27 \
    --reuse_from res/2026.05.10/18-05-08/d435_calibration.json --overwrite
  ```
  (same idea for `calibrate_radar_lidar.py`, which carries `radar_to_lidar`).

## Camera → lidar

Pick sync pairs containing sharp, static structure visible to both sensors
(wall corners, pole bases, curbs), spread across depth and lateral extent.

```bash
python code/sync/src/calibration/calibrate_camera_lidar.py \
  -d 2026.05.10/18-05-08 \
  --pairs 300 1200 2600 4000
```

Left panel = colorized depth, right panel = lidar BEV. Click the **same physical
point** in the depth panel, then the lidar panel. Repeat for ≥ 4 points, then
press `s` to solve and write `camera_to_lidar` into the calibration JSON.

Keys: `u` undo · `n` next pair · `s` solve+save · `q` quit.

Only the **colored** near region of the depth panel is pickable (D435 depth is
reliable to ~6 m; gray = too far). Clicks snap to the nearest valid depth pixel.

**Box mode (`--box`) — recommended when point picks give a high RMS.** Instead of
an exact point, **click two opposite corners** to box a compact object in each
panel (rubber-band preview); the
correspondence becomes the centroid of the enclosed points — far more robust to
cross-sensor mis-picks. Box **small, isolated** objects (poles, signs, boxes) so
the camera (front face) and lidar (footprint) centroids coincide.

```bash
python code/sync/src/calibration/calibrate_camera_lidar.py -d 2026.05.08/17-34-27 \
  --pairs 300 800 1500 --box
```

## Radar → lidar

Pick pairs where a compact strong reflector (parked car, pole, or a placed corner
reflector) appears in both BEVs.

```bash
python code/sync/src/calibration/calibrate_radar_lidar.py \
  -d 2026.05.10/18-05-08 \
  --pairs 500 1500 3000 \
  --z_offset_m 0.0
```

Click the reflector in the radar BEV, then the same object in the lidar BEV.
≥ 3 well-separated points recommended (2 is the minimum). Press `s` to solve.

**Box mode (`--box`) — recommended, since radar peaks are diffuse.** Click two
opposite corners to box the reflector in each BEV: radar uses the intensity-weighted centroid of
the peak (above background, `--box_radar_pct`), lidar uses the point centroid —
much steadier than clicking a fuzzy radar blob.

**`--show_camera`** adds a leftmost camera RGB panel — `[camera | radar | lidar]`
in a row — so you can see what a radar blob actually is when choosing candidates
(context only; scans the camera bag once).

## Derive camera↔radar + validate

Both solves target the **lidar** frame, so once both are done the camera↔radar
transform is derivable (`inv(radar_to_lidar) @ camera_to_lidar`) — no joint solve
needed. This also cross-checks the calibration:

```bash
python code/sync/src/calibration/derive_extrinsics.py -d 2026.05.08/17-34-27
```

It writes `camera_to_radar` / `radar_to_camera` into the calibration JSON and
reports, per pairwise solve, the fit RMS recomputed from the stored matrix + its
saved correspondences, plus rotation/translation sanity checks. Warnings fire on
unsolved placeholders, a high RMS (bad picks / possible mirror), or an
implausibly large translation.

## Re-solve without re-picking (headless)

Correspondences are saved next to the calibration JSON
(`camera_lidar_correspondences.json`, `radar_lidar_correspondences.json`).
Re-run the fit (e.g. after pruning a bad point by hand) without the GUI:

```bash
python code/sync/src/calibration/calibrate_camera_lidar.py \
  --solve_only .../camera_lidar_correspondences.json \
  --calibration_json .../d435_calibration.json
```

## Output

The solved 4×4 is merged into the calibration JSON under `camera_to_lidar` /
`radar_to_lidar`, with a `<key>_meta` block recording the method, point count,
and RMS/max residual (meters) so a placeholder is never silently trusted again.
Every other field (intrinsics, depth_scale) is preserved.

**Check the RMS** printed on solve: a few cm is good for camera↔lidar; large
values mean mis-picked points or too little spread.

## Files

- `rigid_solve.py` — Umeyama 3D/2D solvers. `python rigid_solve.py --self_test`.
- `correspondences.py` — correspondence JSON load/save.
- `interactive.py` — shared dual-panel click picker.
- `extrinsic_io.py` — merge a solved 4×4 into the calibration JSON.
- `calibrate_camera_lidar.py`, `calibrate_radar_lidar.py` — the two solve entry points.
- `derive_extrinsics.py` — derive camera↔radar from the two solves and validate.
