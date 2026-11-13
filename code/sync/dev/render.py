"""Preview/video renderers (merged). Subcommands: full | video | stride5"""
import sys

# ======== dev/render_full_stride1.py ========
def full_main():
    import subprocess, json, os, glob, csv, sys, threading
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
    os.makedirs(SEGDIR, exist_ok=True); os.makedirs(VIZDIR, exist_ok=True)
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
    todo_segs=[s for s in range(len(segs)) if not os.path.exists(f"{SEGDIR}/seg{s:02d}.mp4")]
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
    subs=sorted(glob.glob(f"{SEGDIR}/seg*.mp4"))
    lst="/tmp/s1_concat.txt"
    with open(lst,"w") as fh:
        for f in subs: fh.write(f"file '{os.path.abspath(f)}'\n")
    r=subprocess.run(["ffmpeg","-y","-loglevel","error","-f","concat","-safe","0","-i",lst,"-c","copy",OUTMP4])
    nf=len(pairs)
    print(f"\nFINAL concat exit={r.returncode}: {OUTMP4} ({nf} frames @12fps = {nf/12:.0f}s, {len(subs)} segments)", flush=True)

# ======== dev/render_video.py ========
def video_main():
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

# ======== dev/stride5.py ========
def stride5_main():
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
    merged=None
    files=[f"{OUTDIR}/yolo_window_13k30k.json"]+[f"{OUTDIR}/yolo_off5_batch{k}.json" for k in range(len(lbatches))]
    for fp in files:
        d=json.load(open(fp))
        if merged is None: merged={kk:vv for kk,vv in d.items() if kk!="objects"}; merged["objects"]={}
        merged["objects"].update(d.get("objects",{}))
    ann5=f"{OUTDIR}/yolo_window_stride5.json"
    json.dump(merged, open(ann5,"w"), indent=1)
    print(f"merged stride-5 annotations: {sum(len(v) for v in merged['objects'].values())} boxes", flush=True)
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

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'full': full_main, 'video': video_main, 'stride5': stride5_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
