import subprocess, json, csv, os, glob, shutil, sys

ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT)
CSV="sync/res/2026.05.11/21.37.07/camera_sync_pairs.csv"
CACHE=os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
OUTDIR="sync/res/2026.05.11/21.37.07/annotations"
CACHEJSON="sync/res/2026.05.11/21.37.07/yolo_dets_cache.json"
NOIDX="/tmp/force_seq_chunk.npz"
N=5

ids=[r['pair_idx'] for r in csv.DictReader(open(CSV))]
chunks=[ids[k*len(ids)//N:(k+1)*len(ids)//N] for k in range(N)]

def free_depth():
    for f in glob.glob(CACHE+"/depth_*.npy"):
        try: os.remove(f)
        except OSError: pass

# free space: old 05.08 cache + current depth npy
old=os.path.expanduser("~/.cache/ti_setup_frames/f7099361c5")
if os.path.isdir(old):
    shutil.rmtree(old, ignore_errors=True); print("removed old 05.08 frame cache", flush=True)
free_depth()

for k,chunk in enumerate(chunks):
    if os.path.exists(NOIDX): os.remove(NOIDX)
    out=f"{OUTDIR}/yolo_chunk{k}.json"
    print(f"\n===== CHUNK {k}/{N-1}: {len(chunk)} pairs [{chunk[0]}..{chunk[-1]}] =====", flush=True)
    cmd=["python3","-u","sync/src/camera_projection/yolo_to_lidar.py",
         "--datasets","sync/datasets.0511.json","-d","2026.05.11/21.37.07",
         "--lidar_propose","--yolo_cache",CACHEJSON,"--camera_index",NOIDX,
         "--output_annotations",out,"--pairs",*chunk]
    r=subprocess.run(cmd)
    print(f"CHUNK {k} exit={r.returncode}", flush=True)
    if r.returncode!=0:
        print("ABORTING (chunk failed)", flush=True); sys.exit(1)
    free_depth()  # bound disk for next chunk

# merge chunk annotations -> yolo.json
merged=None
for k in range(N):
    d=json.load(open(f"{OUTDIR}/yolo_chunk{k}.json"))
    if merged is None:
        merged={kk:vv for kk,vv in d.items() if kk!="objects"}; merged["objects"]={}
    merged["objects"].update(d.get("objects",{}))
json.dump(merged, open(f"{OUTDIR}/yolo.json","w"), indent=1)
tot=sum(len(v) for v in merged["objects"].values())
npairs=len([k for k,v in merged["objects"].items() if v])
print(f"\nMERGED -> {OUTDIR}/yolo.json : {tot} boxes across {npairs} pairs", flush=True)
