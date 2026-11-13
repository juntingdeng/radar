import subprocess, json, os, sys, csv

# Label the FULL sequence at STRIDE 1, built as 10 INDEPENDENT stride-10 passes
# (offsets 0..9). Each offset is further split into resumable batches. Durable
# output accumulates per batch file, so a failure in any offset/batch never loses
# the others -- rerun this script and it skips every batch already on disk.
#
# DISK SAFETY: labeling pulls only DEPTH frames from Drive (color is fully YOLO-
# cached). Depth frames are ~0.75 MB each; 48k of them would be ~36 GB and blow the
# near-full disk. So after each batch we delete exactly that batch's depth npys
# (each depth frame is used once, never needed again) -> disk growth bounded to one
# batch. Depth-fallback boxes (~7% of boxes) are preserved; quality matches the
# proven window pipeline.
ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT)
RES="sync/res/2026.05.11/21.37.07"
OUTDIR=f"{RES}/annotations"
CSV=f"{RES}/camera_sync_pairs.csv"
CACHEJSON=f"{RES}/yolo_dets_cache.json"
CACHE=os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
NOIDX="/tmp/force_seq_full.npz"
NPAIRS=47970            # full sequence: pair_idx 0..47969
STRIDE=10              # 10 offsets 0..9 -> combined stride 1
BATCH=500
os.makedirs(OUTDIR, exist_ok=True)

# pair_idx -> camera_depth_idx, so we can free exactly the depth frames each batch used.
depth_of={int(r['pair_idx']):int(r['camera_depth_idx']) for r in csv.DictReader(open(CSV))}

def free_depth_for(batch):
    for p in batch:
        d=depth_of.get(p)
        if d is None: continue
        fp=f"{CACHE}/depth_{d:06d}.npy"
        try: os.remove(fp)
        except OSError: pass

def run_batch(out, batch):
    if os.path.exists(out):
        print(f"  {os.path.basename(out)}: done, skip", flush=True); return True
    if os.path.exists(NOIDX): os.remove(NOIDX)
    cmd=["python3","-u","sync/src/camera_projection/yolo_to_lidar.py",
         "--datasets","sync/datasets.0511.json","-d","2026.05.11/21.37.07",
         "--lidar_propose","--yolo_cache",CACHEJSON,"--camera_index",NOIDX,
         "--output_annotations",out,"--pairs",*map(str,batch)]
    r=subprocess.run(cmd)
    print(f"  {os.path.basename(out)} exit={r.returncode}", flush=True)
    free_depth_for(batch)   # bound disk whether it succeeded or not
    return r.returncode==0

# ---- label each offset independently ----
for off in range(STRIDE):
    pairs=list(range(off, NPAIRS, STRIDE))
    batches=[pairs[i:i+BATCH] for i in range(0,len(pairs),BATCH)]
    print(f"\n########## OFFSET {off}: {len(pairs)} pairs [{pairs[0]}..{pairs[-1]}] in {len(batches)} batches ##########", flush=True)
    ok=True
    for k,batch in enumerate(batches):
        out=f"{OUTDIR}/yolo_full_off{off}_b{k}.json"
        if not run_batch(out, batch):
            print(f"OFFSET {off} batch {k} FAILED -- leaving other offsets/batches intact; rerun to resume", flush=True)
            ok=False; break
    if not ok:
        continue  # keep going to next offset; failed one can be resumed later
    # merge this offset's batches -> per-offset annotations
    merged=None
    for k in range(len(batches)):
        fp=f"{OUTDIR}/yolo_full_off{off}_b{k}.json"
        if not os.path.exists(fp): merged=None; break
        d=json.load(open(fp))
        if merged is None: merged={kk:vv for kk,vv in d.items() if kk!="objects"}; merged["objects"]={}
        merged["objects"].update(d.get("objects",{}))
    if merged is not None:
        json.dump(merged, open(f"{OUTDIR}/yolo_full_off{off}.json","w"), indent=1)
        tot=sum(len(v) for v in merged["objects"].values())
        print(f"OFFSET {off} merged: {tot} boxes", flush=True)

# ---- combine all completed offsets -> stride-1 annotations ----
merged=None; done_offs=[]
for off in range(STRIDE):
    fp=f"{OUTDIR}/yolo_full_off{off}.json"
    if not os.path.exists(fp):
        print(f"offset {off} not complete yet -- stride-1 merge partial", flush=True); continue
    d=json.load(open(fp)); done_offs.append(off)
    if merged is None: merged={kk:vv for kk,vv in d.items() if kk!="objects"}; merged["objects"]={}
    merged["objects"].update(d.get("objects",{}))
if merged is not None:
    outp=f"{OUTDIR}/yolo_full_stride1.json"
    json.dump(merged, open(outp,"w"), indent=1)
    tot=sum(len(v) for v in merged["objects"].values())
    npairs=len([k for k,v in merged["objects"].items() if v])
    print(f"\nSTRIDE-1 MERGE ({len(done_offs)}/10 offsets {done_offs}) -> {outp}: {tot} boxes across {npairs} labeled pairs", flush=True)
