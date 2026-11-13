"""Labeling campaign runners (merged; each was a completed one-off batch job -- kept for
provenance/re-runs). Subcommands: smoke1024 | chunked | stride2 | window | full
  smoke1024  10.24 green-pants smoke window: DETR + lidar frustum -> auto labels for hand-verify
  chunked    0511 night labeling in resumable chunks
  stride2    0511 stride-2 pass        window    0511 single-window pass
  full       0511 FULL sequence at stride 1 as 10 resumable offset passes
Run: python label_runs.py <subcommand>"""
import os, sys, json, numpy as np
import subprocess, json, csv, os, glob, shutil, sys
import subprocess, json, csv, os, sys
import subprocess, json, os, sys
import subprocess, json, os, sys, csv

# ======== label_1024_smoke.py ========
# Label the 10.24 green-pants SMOKE window: DETR (person/car) + lidar FRUSTUM -> 3D boxes (forward/lateral).
# 
# For each window frame: DETR 2D boxes -> project the frame's lidar into the camera (inv camera_to_lidar + colour
# intrinsics) -> the box's frustum lidar points -> median (fwd,lat) = object position + a size prior (person/car).
# Outputs benchmark-format label entries (forward, lateral, bbox_xyxy, label, source, score, median_depth, n_lidar)
# and a verification preview montage. Smoke makes this unreliable -> the output is AUTO labels to be hand-verified.
def smoke1024_main():
    HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
    sys.path.insert(0, os.path.join(CODE, "sync", "src")); sys.path.insert(0, os.path.join(CODE, "sync", "src", "fusion", "late_fusion"))
    import camera_proposals as CP
    RES = os.path.join(CODE, "sync", "res", "2026.10.24", "15.02.30")
    CALIB = json.load(open(os.path.join(RES, "calibration.json")))
    NPZ = os.path.expanduser("~/.cache/ti_setup_frames/smoke_window.npz")
    LIDC = os.path.expanduser("~/.cache/ti_setup_frames/smoke1024_lidar")
    OUT = os.path.join(RES, "smoke_labels_auto.json")
    T = np.array(CALIB["camera_to_lidar"], float); Tinv = np.linalg.inv(T)
    ci = CALIB["color_intrinsics"]; FX, FY, CX, CY = ci["fx"], ci["fy"], ci["cx"], ci["cy"]
    SIZE = {1: (0.5, 0.5), 2: (1.0, 0.6), 3: (2.1, 0.9), 4: (1.0, 0.6), 6: (6.0, 1.4), 8: (6.0, 1.4)}  # half-extent fwd,lat by COCO class

    def frustum(bbox2d, lid):
        """DETR 2D box + lidar (N,>=3 fwd,lat,z) -> (fwd,lat, median_depth, n_pts) or None."""
        P = lid[:, :3].astype(float); Pc = (Tinv @ np.c_[P, np.ones(len(P))].T).T[:, :3]   # lidar -> camera frame
        Z = Pc[:, 2]; front = Z > 0.5
        u = FX * Pc[:, 0] / np.where(front, Z, 1) + CX; v = FY * Pc[:, 1] / np.where(front, Z, 1) + CY
        x0, y0, x1, y1 = bbox2d
        inb = front & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
        if inb.sum() < 3: return None
        sel = P[inb]; rng = np.linalg.norm(sel[:, :2], axis=1)
        rmin = np.percentile(rng, 10)                                                       # robust nearest range (ignore a stray near point)
        near = sel[rng <= rmin + 2.5]                                                        # FOREGROUND cluster only (person, not the building behind)
        if len(near) < 3: near = sel[rng <= np.percentile(rng, 30)]
        if len(near) < 3: near = sel
        return float(np.median(near[:, 0])), float(np.median(near[:, 1])), float(np.median(np.linalg.norm(near[:, :2], axis=1))), int(len(near))

    def main():
        S = np.load(NPZ, allow_pickle=True); cams = S["cams"]; meta = json.loads(str(S["meta"]))
        detr = CP._detector("detr"); labels = {}; nlab = 0
        for k, mt in enumerate(meta):
            cam = cams[k]
            if cam.max() == 0: continue
            lid = np.load(f"{LIDC}/l_{mt['lidar_idx']}.npy").astype(float)
            sid = f"1024_{mt['pair']:06d}"; boxes = []
            for bbox2d, score, label in detr(cam):
                fr = frustum(bbox2d, lid)
                if fr is None: continue
                fwd, lat, md, npts = fr; hf, hl = SIZE.get(label, (0.6, 0.6))
                boxes.append(dict(forward=[fwd - hf, fwd + hf], lateral=[lat - hl, lat + hl], bbox_xyxy=[round(x, 1) for x in bbox2d],
                                  label="person" if label == 1 else "car", source="detr_lidar_frustum", score=round(score, 3),
                                  median_depth_m=round(md, 2), n_depth_points=npts, reliable=False))
            if boxes: labels[sid] = boxes; nlab += len(boxes)
            if k % 40 == 0: print(f"  frame {k+1}/{len(meta)}: {len(boxes)} boxes", flush=True)
        json.dump(labels, open(OUT, "w"), indent=1)
        import collections; lc = collections.Counter(b["label"] for v in labels.values() for b in v)
        print(f"\nsaved {OUT}: {len(labels)} frames labeled, {nlab} boxes {dict(lc)}")

    if __name__ == "__main__":
        main()


# ======== label_chunked.py ========
def chunked_main():

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


# ======== label_stride2.py ========
def stride2_main():

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


# ======== label_window.py ========
def window_main():

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


# ======== label_full_stride1.py ========
def full_main():

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


if __name__ == '__main__':
    import sys as _s
    _c = _s.argv[1] if len(_s.argv) > 1 else ''
    _m = {'smoke1024': smoke1024_main, 'chunked': chunked_main, 'stride2': stride2_main, 'window': window_main, 'full': full_main}
    if _c not in _m:
        print(__doc__ or 'subcommands: ' + ', '.join(_m)); _s.exit(1)
    _m[_c]()
