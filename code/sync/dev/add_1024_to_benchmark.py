"""Add the 10.24 green-pants SMOKE window to benchmark_v1 as seq 2026.10.24 (condition 'smoke', split 'test_smoke').

Keeps the reliable yolo_lidar_frustum PERSON boxes (drops the lidar_radar_proposal 'car' clutter -- trees/poles/rig,
not real cars in this pedestrian scene). Writes:
  benchmark_v1/labels/test_smoke.json   sample_id -> person boxes (benchmark box format, already produced by the pipeline)
  benchmark_v1/calib/2026.10.24.json    copy of the 15.02.30 calibration
  benchmark_v1/samples.jsonl            APPENDS the 2026.10.24 smoke samples (does not touch 0508/0511)
"""
import os, csv, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
BM = os.path.join(CODE, "sync", "res", "benchmark_v1")
RES = os.path.join(CODE, "sync", "res", "2026.10.24", "15.02.30")
AUTO = os.path.join(RES, "annotations", "smoke_auto.json")

# per-pair modality indices from the camera sync csv
sync = {}
for r in csv.DictReader(open(os.path.join(RES, "camera_sync_pairs.csv"))):
    try:
        sync[int(r["pair_idx"])] = dict(camera_color_idx=int(float(r["camera_color_idx"])),
            camera_depth_idx=int(float(r["camera_depth_idx"])) if r.get("camera_depth_idx") else int(float(r["camera_color_idx"])),
            lidar_idx=int(r["lidar_idx"]), radar_idx=int(r["radar_idx"]))
    except (ValueError, KeyError):
        pass

obj = json.load(open(AUTO))["objects"]
labels = {}; samples = []; npers = 0
for pair_s, boxes in obj.items():
    pair = int(pair_s)
    persons = [b for b in boxes if b.get("label") == "person"]      # drop the car-proposal clutter
    if not persons or pair not in sync: continue
    sid = f"1024_{pair:06d}"
    labels[sid] = persons; npers += len(persons)
    rel = any(b.get("box3d", {}).get("reliable") or b.get("reliable") for b in persons)
    samples.append(dict(sample_id=sid, seq="2026.10.24", pair_idx=pair, modality=sync[pair],
                        condition="smoke", reliable=bool(rel), split="test_smoke", n_boxes=len(persons)))

# write labels + calib
os.makedirs(os.path.join(BM, "labels"), exist_ok=True); os.makedirs(os.path.join(BM, "calib"), exist_ok=True)
json.dump(labels, open(os.path.join(BM, "labels", "test_smoke.json"), "w"), indent=1)
shutil.copy(os.path.join(RES, "calibration.json"), os.path.join(BM, "calib", "2026.10.24.json"))

# APPEND samples (guard against double-add: strip any existing 2026.10.24 lines first)
sp = os.path.join(BM, "samples.jsonl")
existing = [l for l in open(sp) if l.strip() and json.loads(l)["seq"] != "2026.10.24"]
with open(sp, "w") as f:
    f.writelines(existing)
    for s in samples: f.write(json.dumps(s) + "\n")

print(f"added seq 2026.10.24 'smoke': {len(samples)} samples, {npers} person boxes -> test_smoke")
print(f"  labels: {os.path.join(BM,'labels','test_smoke.json')}")
print(f"  calib:  {os.path.join(BM,'calib','2026.10.24.json')}")
print(f"  samples.jsonl now: {len(existing)} prev + {len(samples)} smoke = {len(existing)+len(samples)}")
