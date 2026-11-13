# Late Fusion, RL Selection, Temporal Memory, Belief Codec

Stages 4–7 of the pipeline (see `../../../README.md` for the overview). Everything here runs on
the per-capture outputs of sync + calibration + benchmark labeling.

## Three frozen detectors (stage 4)

| detector | model | train | checkpoint |
|---|---|---|---|
| camera | frozen DETR-ResNet-50 + small trained heads (cls + gated range + extent); analytic bearing | `python camera_detector.py mixed` | `camera_detr.pt` (production) |
| lidar | voxel center-head (in_ch=4) | `python train_detectors.py lidar radar packextra` | `detectors_l2.pt` |
| radar | voxel center-head (in_ch=3) on CFAR-style points | (same command) | `detectors_l2.pt` |

- `packextra` = condition-balanced data-scale recipe (L2 train ×3 + pack day stride-20 + pack night
  stride-30); day-only scaling drowns night — balance is mandatory.
- Stage funnel per modality/condition: `python evaluate.py bev lidar day_clean 3`
  (signal → head-fires → proposal ceiling → operating recall; BEV viz to `eval_bev_viz/`).
- Pure-2D camera eval (two-world GT, forward bins, overlays): `python evaluate.py cam2d` → `eval2d_viz/`.

## System ladder + RL selection (stage 5)

```bash
LF_CAM_BACKEND=mono python precompute_proposals.py   # detectors -> proposals_l2.pkl (cache)
python evaluate.py l2 5                              # single / equal / learned fusion + BEV viz
python rl_fusion.py                                  # cost-aware sensor-subset policy
```

Scene split lives in `split.json`; results in `results_l2.json`, `results_rl.json`.
The broader RL study (contextual bandit over probes) is in `../rl_modality/` (own README).

## Temporal memory on moving windows (stage 6)

Window data comes from `dev/extract.py multiview_cam` + per-window sensor extraction
(windows live under `~/CMU/Research/mobile_data_local/multiview_v1/0508/win_*`).

```bash
python window.py odometry win_8825-9175 viz   # point-to-line ICP + deskew + keyframes; V1-V7 validation
python window.py detect   win_8825-9175       # three frozen detectors per frame
python window.py memory   win_8825-9175 viz   # world-frame belief bank (radar-certified cam writes)
python window.py compare  win_8825-9175       # 6 sensor suites through the identical bank
```

**Locked metrics** (every table, fixed denominators): `LIVE` single-frame recall in current FOV ·
`BLIND` memory recall outside the cam wedge · `RETRIEVAL` n2/n1 on blind GT the suite ever knew ·
`WORLD` live+memory vs all GT ≤ 30 m · ghosts/frame. Unit of evaluation = one **memory cycle**
(accumulation window: warm-up W write-only, query span Q); sequences aggregate across tiled cycles.

Key per-window artifacts: `odometry.npz`, `proposals_window.pkl`, `odometry_viz/`
(V1–V7 odometry validation, M7/M8 memory flipbooks + stills).

Hard-won notes: pcap dropouts are invisible in the scan index (use real Δt everywhere); camera
places well only ≤ 12 m (far writes need the radar range cross-check); point-to-point ICP damps
translation on foliage (use point-to-line + planarity filter); negative-evidence decay must be
gentler than the detector's per-object miss streaks.

## Belief codec + channel studies (stage 7)

```bash
python window.py codec win_8825-9175        # compression: explicit fp32/int8/2B vs VAE latents
python window.py codec win_8825-9175 jscc   # digital bit-flip channel (BSC) ablation
python window.py codec win_8825-9175 qam    # fixed 16-QAM modulation, coding varies (clean ablation)
python window.py codec win_8825-9175 awgn   # analog DeepJSCC vs QAM at matched power/uses
```

Headlines: a transmissible belief costs **16 bits** (2-byte explicit x,y matches fp32); learned
codes earn their place as **channel codes, not compressors** — noise-trained latents stay flat
where explicit+Hamming cliffs (BSC p=0.15: 0.77 vs 0.63; QAM −10 dB: 0.69 vs 0.15), and the
attribution is noise-in-the-training-loop, not learning per se.
