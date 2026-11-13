import subprocess, os, glob, sys

ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT)
ANN="sync/res/2026.05.11/21.37.07/annotations/yolo_window_13k30k.json"
VIZDIR="sync/res/2026.05.11/21.37.07/label_video_frames"
OUTMP4="sync/res/2026.05.11/21.37.07/label_video_13k30k_prop.mp4"
NOIDX="/tmp/force_seq_vid.npz"
LO,HI,STRIDE=13000,30000,10
BATCH=450
os.makedirs(VIZDIR, exist_ok=True)

allpairs=list(range(LO,HI+1,STRIDE))
png=lambda p: f"{VIZDIR}/label_pair{p}.png"
todo=[p for p in allpairs if not os.path.exists(png(p))]
print(f"{len(allpairs)} frames total, {len(todo)} to render", flush=True)
batches=[todo[i:i+BATCH] for i in range(0,len(todo),BATCH)]

for k,batch in enumerate(batches):
    if os.path.exists(NOIDX): os.remove(NOIDX)
    print(f"\n===== RENDER BATCH {k}/{len(batches)-1}: {len(batch)} frames [{batch[0]}..{batch[-1]}] =====", flush=True)
    cmd=["python3","-u","sync/src/camera_projection/viz_labels.py",
         "--datasets","sync/datasets.0511.json","-d","2026.05.11/21.37.07",
         "--radar","--radar_points","--fixed_bev","--bev_fwd_max","30","--bev_lat_half","16",
         "--camera_index",NOIDX,"--annotations",ANN,"--out_dir",VIZDIR,"--pairs",*map(str,batch)]
    r=subprocess.run(cmd)
    print(f"RENDER BATCH {k} exit={r.returncode}", flush=True)
    if r.returncode!=0:
        print("ABORT", flush=True); sys.exit(1)

# stitch sorted-by-pair via ffmpeg concat
frames=sorted(glob.glob(VIZDIR+"/label_pair*.png"),
              key=lambda f:int(f.split('label_pair')[1].split('.')[0]))
print(f"stitching {len(frames)} frames -> {OUTMP4}", flush=True)
listfile="/tmp/vidframes.txt"
with open(listfile,"w") as fh:
    for f in frames: fh.write(f"file '{os.path.abspath(f)}'\n")
r=subprocess.run(["ffmpeg","-y","-loglevel","error","-r","12","-f","concat","-safe","0",
                  "-i",listfile,"-c:v","libx264","-pix_fmt","yuv420p",
                  "-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2",OUTMP4])
print(f"ffmpeg exit={r.returncode}; wrote {OUTMP4} ({len(frames)} frames @12fps = {len(frames)/12:.0f}s)", flush=True)
