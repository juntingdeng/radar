import subprocess, json, os, glob, csv, sys, threading

# Render the DENSE stride-1 preview (pairs 13000..29999) -- PARALLEL (4 workers).
# The machine is 10-core/16GB and all data is LOCAL (146GB bag on SSD); the cost is
# per-frame CPU (matplotlib 4-panel + lidar/radar DSP) plus local bag seeks. So we
# run 4 concurrent viz_labels workers on DISJOINT contiguous pair blocks, each using
# the real camera_index.npz for DIRECT seeks (4 sequential scans would thrash the
# bag). Segment model (2000-frame, pair-aligned sub-mp4s) is kept for disk bounding:
# each worker renders -> stitches -> frees per segment. Fully resumable: a finished
# segment's seg{NN}.mp4 is skipped.
ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT)
RES="sync/res/2026.05.11/21.37.07"
ANN=f"{RES}/annotations/yolo_full_stride1.json"
CSV=f"{RES}/camera_sync_pairs.csv"
CIDX=f"{RES}/camera_index.npz"                       # REAL seek index (direct frame access)
LCACHE=os.path.expanduser("~/.cache/ti_setup_frames/lidar_scan_cache")  # pre-extracted scans
CACHE=os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
SEGDIR=f"{RES}/stride1_segments"
VIZDIR="/tmp/s1_frames"
OUTMP4=f"{RES}/label_video_full_13k30k_stride1.mp4"
LO,HI,STRIDE=13000,30000,1
SEG=2000  # keep 2000 so the already-built seg00.mp4 (13000-14999) stays valid/resumable
NWORK=1   # SINGLE process: parallel readers thrash the shared 70/146/33GB files. One
          # sequential reader is what worked originally. Lidar cache stays (free: the one
          # reader skips the 70GB pcap), but no concurrency -> no contention, no stall.
os.makedirs(SEGDIR, exist_ok=True); os.makedirs(VIZDIR, exist_ok=True)

if not os.path.exists(ANN): print(f"annotations not ready: {ANN}", flush=True); sys.exit(2)
if not os.path.isfile(CIDX): print(f"camera index missing: {CIDX}", flush=True); sys.exit(2)

_rows={int(r['pair_idx']):r for r in csv.DictReader(open(CSV))}
depth_of={p:int(r['camera_depth_idx']) for p,r in _rows.items()}
color_of={p:int(r['camera_color_idx']) for p,r in _rows.items()}
lidar_of={p:int(r['lidar_idx']) for p,r in _rows.items()}
pairs=list(range(LO,HI,STRIDE))
segs=[pairs[i:i+SEG] for i in range(0,len(pairs),SEG)]
print(f"stride-1 {LO}..{HI}: {len(pairs)} frames, {len(segs)} segments, {NWORK} parallel workers", flush=True)
png=lambda p: f"{VIZDIR}/label_pair{p}.png"
plock=threading.Lock()

def free_seg(seg):
    for p in seg:
        for idx,kind in ((color_of.get(p),'color'),(depth_of.get(p),'depth')):
            if idx is None: continue
            try: os.remove(f"{CACHE}/{kind}_{idx:06d}.npy")
            except OSError: pass
        li=lidar_of.get(p)                      # free the pre-extracted scan too (used once)
        if li is not None:
            try: os.remove(f"{LCACHE}/l_{li}.npy")
            except OSError: pass
        try: os.remove(png(p))
        except OSError: pass

def render_seg(s, seg):
    submp4=f"{SEGDIR}/seg{s:02d}.mp4"
    if os.path.exists(submp4):
        with plock: print(f"SEG {s}: exists, skip", flush=True); return
    todo=[p for p in seg if not os.path.exists(png(p))]
    with plock: print(f"SEG {s}: render {len(todo)} frames [{seg[0]}..{seg[-1]}]", flush=True)
    # 450-frame batches with SEQUENTIAL camera scan (NOIDX) -- the original working path.
    # Per-frame camera_index SEEK (900 random seeks into the 146GB bag) was far slower for
    # dense contiguous frames than one forward pass; that was the stall. Lidar stays cached.
    NOIDX=f"/tmp/noidx_seg{s}.npz"  # non-existent -> viz_labels falls back to sequential scan
    for i in range(0,len(todo),450):
        batch=todo[i:i+450]
        cmd=["python3","-u","sync/src/camera_projection/viz_labels.py",
             "--datasets","sync/datasets.0511.json","-d","2026.05.11/21.37.07",
             "--radar","--radar_points","--fixed_bev","--bev_fwd_max","30","--bev_lat_half","16",
             "--camera_index",NOIDX,"--lidar_cache",LCACHE,
             "--annotations",ANN,"--out_dir",VIZDIR,"--pairs",*map(str,batch)]
        if subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode!=0:
            with plock: print(f"SEG {s} FAILED at sub-batch {i//450}", flush=True)
            raise RuntimeError(f"seg {s} render failed")
    frames=[png(p) for p in seg if os.path.exists(png(p))]
    lst=f"/tmp/s1seg{s}.txt"
    with open(lst,"w") as fh:
        for f in frames: fh.write(f"file '{os.path.abspath(f)}'\n")
    r=subprocess.run(["ffmpeg","-y","-loglevel","error","-r","12","-f","concat","-safe","0",
                      "-i",lst,"-c:v","libx264","-pix_fmt","yuv420p",
                      "-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2",submp4])
    if r.returncode!=0: raise RuntimeError(f"seg {s} ffmpeg failed")
    free_seg(seg)
    with plock: print(f"SEG {s}: wrote {submp4} ({len(frames)} frames), freed", flush=True)

# assign segments to workers in CONTIGUOUS blocks (keeps each worker's seeks local)
todo_segs=[s for s in range(len(segs)) if not os.path.exists(f"{SEGDIR}/seg{s:02d}.mp4")]
# contiguous split:
def chunk(lst,n):
    k,m=divmod(len(lst),n); return [lst[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]
worker_segs=chunk(todo_segs, NWORK)
print(f"remaining segments {todo_segs} -> workers {worker_segs}", flush=True)

def worker(wid, seglist):
    for s in seglist:
        render_seg(s, segs[s])

threads=[threading.Thread(target=worker,args=(w,sl)) for w,sl in enumerate(worker_segs) if sl]
for t in threads: t.start()
for t in threads: t.join()

# ---- final concat of all segment sub-mp4s in pair order ----
subs=sorted(glob.glob(f"{SEGDIR}/seg*.mp4"))
lst="/tmp/s1_concat.txt"
with open(lst,"w") as fh:
    for f in subs: fh.write(f"file '{os.path.abspath(f)}'\n")
r=subprocess.run(["ffmpeg","-y","-loglevel","error","-f","concat","-safe","0","-i",lst,"-c","copy",OUTMP4])
nf=len(pairs)
print(f"\nFINAL concat exit={r.returncode}: {OUTMP4} ({nf} frames @12fps = {nf/12:.0f}s, {len(subs)} segments)", flush=True)
