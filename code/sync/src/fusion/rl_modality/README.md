# rl_modality — RL prototype for useful-modality selection

A contextual-bandit agent that looks at **cheap probes** of the scene and chooses which sensor subset
(camera / lidar / radar) to spend compute on, trained and evaluated **entirely on measured data** —
the 495 real frames of the 10.15 smoke/fog windows (clean → dense smoke → fog).

## Is there a pretrained model for this?
No. There is no off-the-shelf pretrained modality selector. Closest things: supervised *gating* layers in
robust-fusion nets (BEVFusion-robust variants, HRFuser, CRN) — trained per-dataset, no reusable checkpoint,
none match this radar; RL sensor-scheduling literature — research code only. Reusable pretrained *components*:
YOLOv8 for the camera branch (used here), PointPillars-class lidar detectors (large domain gap), nothing for
the AWR2944 BEV format. The selector itself must be trained on our own degradation data — which is what the
495-layer extraction dataset and the cross-scene compositor are for (pretrain on synthetic composites over a
labeled corpus, fine-tune on real).

## Environment (all measured, `build_env.py` → `env_table.npz`)
- **Reward ingredients** (full measurements, offline): per-frame visibility in 0..1 per modality —
  camera = far-band Laplacian-sharpness ratio vs clean background (smoke blurs + washes; the discriminator
  lesson); lidar = far-field point survival vs clean (stored counts); radar = strong-cell recall vs the median
  clean reference (stored BEV crops). A modality with no valid same-moment measurement contributes 0
  (real sensor dropout, e.g. smoke1's first 4.5 s lidar gap).
- **State** (cheap probes only — the full measurements are *not* visible to the agent): thumbnail
  brightness/contrast, near-band sharpness, lidar near-count, dropout flag. 6-quantile-bin one-hot features.
- **Actions**: the 7 non-empty subsets of {C, L, R}. **Cost**: C 0.3, L 0.4, R 1.0 (radar BEV beamforming
  dominates this stack), λ=0.15.
- **Reward** = max visibility over the chosen subset − λ·Σcost.

## Agent + results (`bandit.py` → `results.json`, `rl_modality_policy.png`)
ε-greedy linear Q on binned features (numpy SGD, partial feedback — only the chosen action's reward is
observed). Contiguous-block 70/30 split. Held-out test:

| policy | reward | coverage | cost |
|---|---|---|---|
| **learned** | **0.852** | 0.944 | 0.62 |
| oracle (per-frame best) | 0.899 | 0.966 | 0.44 |
| always camera | 0.819 | 0.864 | 0.30 |
| always lidar | 0.741 | 0.801 | 0.40 |
| always radar | 0.774 | 0.924 | 1.00 |
| always all three | 0.722 | 0.977 | 1.70 |
| random subset | 0.767 | — | — |

The learned policy: clean/fog frames → camera (cheap, sufficient); smoke frames → radar(+) when the probes
(near-field lidar flood, washed thumbnail) indicate camera/lidar collapse. It beats every fixed policy and
closes ~58 % of the fixed-camera→oracle gap.

## Reward design (v1 — kept for first testing)
One scalar per (frame, chosen-subset):

> **reward(action) = recall(chosen subset) − λ · Σ cost(chosen sensors)**

- **action** = one of the 7 non-empty subsets of {camera, lidar, radar} (C, L, R, CL, CR, LR, CLR) — which sensors
  to turn on this frame.
- **recall (benefit)** = fraction of the frame's GT objects detected by **at least one chosen sensor** (per-object
  *union*, so the env stores per-object hit lists `hits_cam/hits_lidar/hits_radar`, not just aggregate recall). A
  hit is: camera = YOLOv8 box IoU ≥ 0.3 vs GT; lidar = points in the GT BEV box ≥ max(8, 0.3·clean); radar = ≥ 3
  strong cells (> clean p99) in the footprint. So recall(CR) on a smoke-occluded car = 1 (radar sees it) even when
  camera-alone = 0.
- **cost (penalty)** = sum of per-sensor compute costs: **camera 0.3, lidar 0.4, radar 1.0** (radar BEV beamforming
  dominates this stack). CLR = 1.7, C = 0.3.
- **λ = 0.15** — trade-off: radar (cost 1.0) is worth turning on only if it adds > 0.15 recall. Clean frame →
  camera alone (recall≈1, cost 0.3 → 0.955) beats CLR (0.745); smoke frame → CR (recall≈1, cost 1.3 → 0.805) beats
  camera-only (recall 0 → −0.045).
- **State the agent sees** = cheap probes only (thumbnail brightness/contrast, near-field Laplacian, lidar
  near-count) — NOT the recalls. It must learn which regime it's in.
- **Baselines** (held-out, split by base scene so a scene's conditions can't leak train→test): per-frame **oracle**
  (ceiling), fixed always-C/L/R/CLR, random.

Two caveats kept on record: (1) λ and the cost vector are hand-set — they should match the real compute/latency
budget; absolute rewards are meaningful only *relative to the baselines under the same λ*. (2) recall scores
detection *presence*, not localization quality — a radar "hit" (≥3 cells) is coarser than a camera box but counts
equally, which slightly flatters the cheaper/coarser sensors. This v1 reward is kept unchanged for first testing.

## 50-scene / 16-layer test (`build_env50.py` panels + `models.py` learned detectors → `bandit50.py`)
The scaled env: **50 labeled night base scenes × {clean + 4 of the 16 real layers (rotating so all 16 covered)}
= 250 samples**, PHYSICS composites (no FLUX). Per-scene placement = flip the 10.15 dense core onto the scene's car
side; all 3 modalities composited from the real layers (camera matte+airlight, lidar backscatter+occlusion, radar
dB-delta blend). `build_env50.py` generates the composites + a `cam|lidar|radar` inspection panel per sample in
`rl_env50_frames/` (annotated with RAW stats only — it does **not** score detection). **Scoring is done by the
learned detectors in `models.py`** — the earlier YOLO-labeler + occupancy-threshold scoring was CIRCULAR (see the
next section) and has been removed. The result is the fair learned-detector one below. Still in-domain (physics
composites); a real-smoke transfer check (on the 6 car-track frames) is a TODO on the learned path.

## Fair evaluation — learned per-modality detectors (`models.py` → `detectors.pt`, `rl_modality_env50_fair.png`)
The threshold recall above is **biased**: the GT boxes were made by the camera YOLO (`yolo_to_lidar` labeler), so
scoring camera with YOLO is circular (clean camera = 1.00 by construction), lidar is GT-circular via raising, and
the lidar/radar "detectors" were just occupancy thresholds (points-in-box, ≥3 cells) — not comparable to a learned
camera detector. `models.py` fixes this with a **learned detector per modality**, each an object-vs-background
classifier trained on CLEAN frames only (so it learns the modality's object appearance, independent of the labeler),
then evaluated on degraded frames:
- **CameraNet** — conv stem → **attention head** over the image-crop tokens.
- **LidarNet** — **points** branch (per-point MLP + masked max-pool) + **BEV** branch (CNN on a rasterized height
  patch), fused.
- **RadarNet** — **points** branch (top-k BEV peak cells as (Δfwd,Δlat,power)) + **BEV** branch (CNN on the power
  patch), fused.

**CORRECTED LIBRARY (sync rebuild).** The original 16-layer library was built from the *old* span_map nearest-pairing,
which inherited a known 10.15 lidar dropout (el 1766–1770.5): its "light" layers paired thick real camera-smoke with a
*borrowed clear* lidar scan → a fake camera-blind/lidar-fine regime. Rebuilt `smoke_library_corrected.npz` from
`smoke1_layers.npz` valid frames only (`lidar_valid & radar_valid`, same-instant triplets, el 1770.4–1776). All numbers
below are the corrected ones. (The pre-rebuild table had cam 0.28 / lidar 0.91 in "light" — that was the artifact.)

Fairly scored on the corrected library:

| condition | camera | lidar | radar |
|---|---|---|---|
| clean | 1.00 | 0.95 | 0.74 |
| light smoke | 0.77 | 0.42 | 0.79 |
| dense smoke | 0.83 | 0.24 | 0.73 |

With correct sync: camera never fully crashes in the valid frames (the α≈0.5 camera-blind moments coincide with the
lidar dropout and are excluded — this event has **no** valid camera-blind + real-lidar triplet); lidar is degraded
wherever smoke is validly present (0.42/0.24, not the artifact 0.91); radar penetrates but detects mediocre (~0.73–0.79).
RL on the corrected learned recalls (per-scene split): **learned(weighted) 0.844 = always-camera 0.844** (ties),
discrete-binary 0.794, fixed-radar 0.494, fixed-lidar 0.269; oracle 0.927. **Takeaway: the earlier selector "win" was
largely the sync-mispairing artifact.** With correct data, always-camera is ~optimal on this event (camera best and
cheapest almost everywhere valid, lidar flooded with the smoke, radar mediocre); the oracle's small headroom comes from
a few moderately-thick-camera frames where radar would help, which the simple policy misses. `models.py` is
CPU-trainable (~1 min); detectors are prototypes (~150 clean positives) — scale the clean set to sharpen.

## Weighted (continuous) policy — `policy_weighted.py` → `rl_modality_weighted.png`
The action is a **weight per modality**, not a binary on/off subset. Reliability-gated late fusion: the policy
pi(probes) → **w on the simplex** (softmax, Σw=1); per object the learned detector LOGITS are fused
`fused = Σ wₘ·logitₘ` (detected iff fused>0) and cost is `Σ wₘ·cₘ`. It generalizes the binary subset (w∈{0,1}³)
and is the interpretable twin of the attention-fusion reliability gate.

Why simplex, not independent [0,1]³: with independent weights the decision depends only on the *sign* of Σwₘlogitₘ,
so the policy can shrink weights →0 and still detect on a confident modality, faking cost<0.3 (it did: 0.903 at
cost 0.20). Softmax forces a reliability *distribution*, cost ∈ [0.3,1.0], no loophole.

Held-out on the CORRECTED library (hard recall − 0.15·cost, per-scene split): **learned(weighted) 0.844 = fixed-camera
0.844** (ties), discrete-binary 0.794, fixed-equal 0.791, fixed-radar 0.494, fixed-lidar 0.269; simplex oracle 0.927.
With correctly-synced data the policy learns to put ~all weight on camera (it's best and cheapest in nearly every valid
frame), so it ties always-camera — the earlier "weighted beats everything" (0.886 vs 0.768) was on the stale,
sync-mispaired library. The method is sound (simplex weights, differentiable soft reward, no cost loophole); this event
just doesn't contain a regime where soft re-weighting helps once the data is correct. Trained by direct gradient on a
soft reward (soft_recall = mean sigmoid(fused)), reported with hard recall for comparability.

## Honest limitations / next steps
1. **Sharpen the learned detectors** — they're prototypes (small nets, ~150 clean positives, CPU ~1 min); scale the
   clean training set and add harder negatives so the fair recalls are tighter.
2. **Real-smoke transfer check** (TODO) — the earlier version (`eval_realsmoke.py`, since removed) was tied to the
   deleted object-env policy and scored camera with YOLO; rebuild it on the learned path: save the `bandit50`
   policy, and score the 6 real car-track frames consistently (learned detectors or the production detector, stated
   explicitly). The earlier run already showed a real synthetic→real gap, so expect one.
3. Bandit, not sequential RL: no temporal state yet (smoke persists — a last-outcome state would let the agent
   re-check expensive sensors sparsely).
4. λ and the cost vector are design choices; sweep for the cost-coverage frontier.
5. The probes are hand-picked; the attention-fusion gate ([[attn-fusion-prototype]]) is the learned-feature
   version of the same decision — this prototype is its interpretable, budgeted counterpart.

## Earlier: scene-level reward (v0, `build_env.py` + `bandit.py`, kept)
First pass used scene-**visibility** (not object-detection) rewards on 495 frames — learned 0.852 > always-camera
0.819, oracle 0.899. No detector circularity (visibility, not detection), kept as a documented earlier approach. The
static parked cars of 10.15 were too marginal for per-object GT there (the only mid-range candidate was the *moving*
SUV, dropped by the two-reference persistence check in `label_objects.py`) — which motivated the composited env.

Removed (superseded / circular): the 12-frame object-level env (`build_env_obj.py`/`bandit_obj.py` + outputs) and its
`eval_realsmoke.py` — both used YOLO-as-labeler + occupancy thresholds; replaced by the 50-scene env + learned
detectors. `viz_placement.py`/`rl_env_sample.png` (the object-env sample figure) and its placement-artifact notes
are retained for reference.

## Files (current)
- **Fair pipeline:** `build_env50.py` (composites + inspection panels, no scoring) · `models.py` (learned
  per-modality detectors → `detectors.pt`, `env50_learned_rows.json`) · `bandit50.py` (RL test on the learned recalls
  → `results50.json`, `rl_modality_env50.png`) · `rl_modality_env50_fair.png` (threshold-vs-learned comparison).
- **Scene-level v0 (kept):** `build_env.py` · `bandit.py` · `env_table.npz` · `results.json` · `rl_modality_policy.png`.
- **Reference:** `label_objects.py` (static-labeling dead-end + two-reference persistence technique) ·
  `viz_placement.py` / `rl_env_sample.png`.
