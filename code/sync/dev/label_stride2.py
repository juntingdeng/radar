import subprocess, json, csv, os, sys

ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT)
CSV="sync/res/2026.05.11/21.37.07/camera_sync_pairs.csv"
OUTDIR="sync/res/2026.05.11/21.37.07/annotations"
CACHEJSON="sync/res/2026.05.11/21.37.07/yolo_dets_cache.json"
NOIDX="/tmp/force_seq_s2.npz"
BATCH=4000   # ~3 GB depth in RAM per subprocess (safe on 16 GB)

ids=[r['pair_idx'] for r in csv.DictReader(open(CSV))]
s2=ids[::2]  # stride 2
batches=[s2[i:i+BATCH] for i in range(0,len(s2),BATCH)]
print(f"{len(s2)} stride-2 pairs in {len(batches)} batches of <= {BATCH}", flush=True)

for k,batch in enumerate(batches):
    out=f"{OUTDIR}/yolo_s2_batch{k}.json"
    if os.path.exists(out):
        print(f"BATCH {k}: already done, skip", flush=True); continue
    if os.path.exists(NOIDX): os.remove(NOIDX)
    print(f"\n===== BATCH {k}/{len(batches)-1}: {len(batch)} pairs [{batch[0]}..{batch[-1]}] =====", flush=True)
    cmd=["python3","-u","sync/src/camera_projection/yolo_to_lidar.py",
         "--datasets","sync/datasets.0511.json","-d","2026.05.11/21.37.07",
         "--lidar_propose","--yolo_cache",CACHEJSON,"--camera_index",NOIDX,
         "--output_annotations",out,"--pairs",*batch]
    r=subprocess.run(cmd)
    print(f"BATCH {k} exit={r.returncode}", flush=True)
    if r.returncode!=0:
        print("ABORT (batch failed)", flush=True); sys.exit(1)

# merge all batches -> yolo.json
merged=None
for k in range(len(batches)):
    d=json.load(open(f"{OUTDIR}/yolo_s2_batch{k}.json"))
    if merged is None:
        merged={kk:vv for kk,vv in d.items() if kk!="objects"}; merged["objects"]={}
    merged["objects"].update(d.get("objects",{}))
json.dump(merged, open(f"{OUTDIR}/yolo.json","w"), indent=1)
tot=sum(len(v) for v in merged["objects"].values())
npairs=len([k for k,v in merged["objects"].items() if v])
print(f"\nMERGED -> {OUTDIR}/yolo.json : {tot} boxes across {npairs} pairs", flush=True)
