import subprocess, json, os, sys

ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT)
OUTDIR="sync/res/2026.05.11/21.37.07/annotations"
CACHEJSON="sync/res/2026.05.11/21.37.07/yolo_dets_cache.json"
NOIDX="/tmp/force_seq_win.npz"
LO,HI,STRIDE=13000,30000,10
BATCH=500

pairs=list(range(LO,HI+1,STRIDE))
batches=[pairs[i:i+BATCH] for i in range(0,len(pairs),BATCH)]
print(f"window {LO}-{HI} stride {STRIDE}: {len(pairs)} pairs in {len(batches)} batches", flush=True)

for k,batch in enumerate(batches):
    out=f"{OUTDIR}/yolo_win_batch{k}.json"
    if os.path.exists(out):
        print(f"BATCH {k}: done, skip", flush=True); continue
    if os.path.exists(NOIDX): os.remove(NOIDX)
    print(f"\n===== BATCH {k}/{len(batches)-1}: {len(batch)} pairs [{batch[0]}..{batch[-1]}] =====", flush=True)
    cmd=["python3","-u","sync/src/camera_projection/yolo_to_lidar.py",
         "--datasets","sync/datasets.0511.json","-d","2026.05.11/21.37.07",
         "--lidar_propose","--yolo_cache",CACHEJSON,"--camera_index",NOIDX,
         "--output_annotations",out,"--pairs",*map(str,batch)]
    r=subprocess.run(cmd)
    print(f"BATCH {k} exit={r.returncode}", flush=True)
    if r.returncode!=0:
        print("ABORT", flush=True); sys.exit(1)

# merge -> windowed annotations
merged=None
for k in range(len(batches)):
    d=json.load(open(f"{OUTDIR}/yolo_win_batch{k}.json"))
    if merged is None:
        merged={kk:vv for kk,vv in d.items() if kk!="objects"}; merged["objects"]={}
    merged["objects"].update(d.get("objects",{}))
outp=f"{OUTDIR}/yolo_window_13k30k.json"
json.dump(merged, open(outp,"w"), indent=1)
tot=sum(len(v) for v in merged["objects"].values())
npairs=len([k for k,v in merged["objects"].items() if v])
print(f"\nMERGED -> {outp} : {tot} boxes across {npairs} labeled pairs", flush=True)
