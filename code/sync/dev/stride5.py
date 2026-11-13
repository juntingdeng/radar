import subprocess, json, os, glob, sys

ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT)
OUTDIR="sync/res/2026.05.11/21.37.07/annotations"
CACHEJSON="sync/res/2026.05.11/21.37.07/yolo_dets_cache.json"
VIZDIR="sync/res/2026.05.11/21.37.07/label_video_frames"   # already holds offset-0 frames
NOIDX="/tmp/force_seq_s5.npz"
BATCH=500

off5=list(range(13005,30000,10))   # 13005,13015,...,29995 (interleaves the offset-0 set)
print(f"offset-5 set: {len(off5)} pairs [{off5[0]}..{off5[-1]}]", flush=True)

# ---- 1) LABEL offset-5 pairs (batched, resumable) ----
lbatches=[off5[i:i+BATCH] for i in range(0,len(off5),BATCH)]
for k,batch in enumerate(lbatches):
    out=f"{OUTDIR}/yolo_off5_batch{k}.json"
    if os.path.exists(out): print(f"LABEL batch {k}: done, skip", flush=True); continue
    if os.path.exists(NOIDX): os.remove(NOIDX)
    print(f"\n=== LABEL BATCH {k}/{len(lbatches)-1}: {len(batch)} pairs ===", flush=True)
    cmd=["python3","-u","sync/src/camera_projection/yolo_to_lidar.py",
         "--datasets","sync/datasets.0511.json","-d","2026.05.11/21.37.07",
         "--lidar_propose","--yolo_cache",CACHEJSON,"--camera_index",NOIDX,
         "--output_annotations",out,"--pairs",*map(str,batch)]
    if subprocess.run(cmd).returncode!=0: print("ABORT label",flush=True); sys.exit(1)

# ---- 2) merge offset-0 + offset-5 -> stride-5 annotations ----
merged=None
files=[f"{OUTDIR}/yolo_window_13k30k.json"]+[f"{OUTDIR}/yolo_off5_batch{k}.json" for k in range(len(lbatches))]
for fp in files:
    d=json.load(open(fp))
    if merged is None: merged={kk:vv for kk,vv in d.items() if kk!="objects"}; merged["objects"]={}
    merged["objects"].update(d.get("objects",{}))
ann5=f"{OUTDIR}/yolo_window_stride5.json"
json.dump(merged, open(ann5,"w"), indent=1)
print(f"merged stride-5 annotations: {sum(len(v) for v in merged['objects'].values())} boxes", flush=True)

# ---- 3) RENDER offset-5 frames into VIZDIR (resumable) ----
png=lambda p: f"{VIZDIR}/label_pair{p}.png"
todo=[p for p in off5 if not os.path.exists(png(p))]
print(f"render: {len(todo)} new frames (offset-0 already rendered)", flush=True)
rbatches=[todo[i:i+450] for i in range(0,len(todo),450)]
for k,batch in enumerate(rbatches):
    if os.path.exists(NOIDX): os.remove(NOIDX)
    print(f"\n=== RENDER BATCH {k}/{len(rbatches)-1}: {len(batch)} frames ===", flush=True)
    cmd=["python3","-u","sync/src/camera_projection/viz_labels.py",
         "--datasets","sync/datasets.0511.json","-d","2026.05.11/21.37.07",
         "--radar","--radar_points","--fixed_bev","--bev_fwd_max","30","--bev_lat_half","16",
         "--camera_index",NOIDX,"--annotations",ann5,"--out_dir",VIZDIR,"--pairs",*map(str,batch)]
    if subprocess.run(cmd).returncode!=0: print("ABORT render",flush=True); sys.exit(1)

# ---- 4) STITCH all frames (offset-0 + offset-5) sorted by pair -> stride-5 mp4 ----
frames=sorted(glob.glob(VIZDIR+"/label_pair*.png"),
              key=lambda f:int(f.split('label_pair')[1].split('.')[0]))
listfile="/tmp/s5frames.txt"
with open(listfile,"w") as fh:
    for f in frames: fh.write(f"file '{os.path.abspath(f)}'\n")
out5="sync/res/2026.05.11/21.37.07/label_video_13k30k_stride5.mp4"
r=subprocess.run(["ffmpeg","-y","-loglevel","error","-r","12","-f","concat","-safe","0",
                  "-i",listfile,"-c:v","libx264","-pix_fmt","yuv420p",
                  "-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2",out5])
print(f"\nffmpeg exit={r.returncode}; wrote {out5} ({len(frames)} frames @12fps = {len(frames)/12:.0f}s)", flush=True)
