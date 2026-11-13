"""Multi-view WINDOW SELECTION from the full sequential recordings (camera-only pass, no lidar/radar extraction).

The L2 benchmark frames were sampled for scene DIVERSITY (RL env), so they are nearly useless for multi-view.
This selects continuous windows where the ego is actually MOVING and GT labels are dense, using only data that is
already local: the extracted camera streams (pid -> colour frame via camera_sync_pairs.csv) and benchmark_v1 labels.

  motion metric   mean |diff| of consecutive downsampled grey frames, sampled every STRIDE pids (~1 s)
  window score    median motion over the window  x  log(1 + mean labelled objects/frame)
  output          top-K windows per sequence -> multiview_windows.json + an annotated check VIDEO per window
                  (res/<capture>/multiview_windows/win<i>_pid<a>-<b>.mp4, 5 Hz, pid/motion/object-count overlay)

After visual approval, the lidar/radar extraction job targets exactly these pid ranges.
Run: python sync/dev/multiview_windows.py
"""
import os, csv, json, glob, collections, tempfile, shutil
import numpy as np, cv2
MCACHE = os.path.expanduser("~/.cache/ti_setup_frames")      # motion-scan cache (local, survives re-runs)
MIN_BRIGHT = 20.0                                            # window mean grey floor: rejects near-black night windows

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # dev -> sync -> code
SEQS = {
    "0511": dict(res=os.path.join(CODE, "sync", "res", "2026.05.11", "21.37.07"),
                 cam=os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.11/camera_extracted")),
    "0508": dict(res=os.path.join(CODE, "sync", "res", "2026.05.08", "17-34-27"),
                 cam=os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.08/camera_extracted")),
}
LABELS = os.path.join(CODE, "sync", "res", "benchmark_v1", "labels")
STRIDE = 10            # motion sampled every 10 pids (~1 s)
WIN = 250              # window length in pids (~25 s)
TOPK = 2               # windows to render per sequence
FPS_OUT = 5            # check-video rate (every 2nd pid)

def load_pairs(res):
    m = {}
    with open(os.path.join(res, "camera_sync_pairs.csv")) as fh:
        for r in csv.DictReader(fh):
            m[int(r["pair_idx"])] = int(r["camera_color_idx"])
    return m

def load_obj_counts():
    cnt = collections.Counter()
    for f in glob.glob(os.path.join(LABELS, "*.json")):
        for sid, boxes in json.load(open(f)).items():
            cnt[sid] = max(cnt[sid], sum(1 for b in boxes if "forward" in b))
    return cnt

def cam_path(camdir, idx):
    for ext in ("jpg", "png", "jpeg"):
        p = os.path.join(camdir, "color", f"{idx:06d}.{ext}")
        if os.path.exists(p): return p
    return None

def gray_small(camdir, idx):
    p = cam_path(camdir, idx)
    if p is None: return None
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (80, 60)).astype(np.float32) if im is not None else None

def main():
    objc = load_obj_counts()
    chosen = {}
    for seq, S in SEQS.items():
        pairs = load_pairs(S["res"])
        pids = sorted(pairs)
        grid = pids[::STRIDE]
        # ---- motion + brightness scan over the extracted camera stream (cached) ----
        cachep = os.path.join(MCACHE, f"mv_motion_{seq}.npz")
        if os.path.exists(cachep):
            Z = np.load(cachep)
            mot = dict(zip(Z["pids"].tolist(), Z["mot"].tolist()))
            bri = dict(zip(Z["pids"].tolist(), Z["bri"].tolist()))
            print(f"[{seq}] motion scan loaded from cache ({len(mot)} samples)", flush=True)
        else:
            mot = {}; bri = {}; prev = None
            for i, p in enumerate(grid):
                g = gray_small(S["cam"], pairs[p])
                if g is not None and prev is not None:
                    mot[p] = float(np.abs(g - prev).mean() / (prev.mean() + 1.0))
                    bri[p] = float(g.mean())
                if g is not None: prev = g
                if i % 500 == 0: print(f"[{seq}] motion scan {i}/{len(grid)}", flush=True)
            ks = sorted(mot)
            np.savez(cachep, pids=np.array(ks), mot=np.array([mot[k] for k in ks]),
                     bri=np.array([bri[k] for k in ks]))
        # ---- score sliding windows ----
        cands = []
        gp = sorted(mot)
        for a in range(0, len(gp) - WIN // STRIDE, max(1, WIN // STRIDE // 4)):
            w = gp[a:a + WIN // STRIDE]
            if w[-1] - w[0] > WIN * 2: continue                       # gap in pairs/frames
            m = float(np.median([mot[p] for p in w]))
            b = float(np.mean([bri[p] for p in w]))
            if b < MIN_BRIGHT: continue                               # near-black night stretch: motion = noise
            oc = [objc.get(f"{seq}_{p:06d}", 0) for p in range(w[0], w[-1])]
            lab = float(np.mean(oc)); labfrac = float(np.mean([c > 0 for c in oc]))
            if labfrac < 0.5: continue                                # need dense labels
            cands.append(dict(pid0=int(w[0]), pid1=int(w[-1]), motion=round(m, 4), bright=round(b, 1),
                              mean_objs=round(lab, 1), score=round(m * np.log1p(lab), 4)))
        cands.sort(key=lambda c: -c["score"])
        sel = []; ranked = []
        for c in cands:                                               # non-overlapping ranking
            if all(c["pid0"] > s["pid1"] + WIN or c["pid1"] < s["pid0"] - WIN for s in ranked):
                ranked.append(c)
            if len(ranked) == 8: break
        sel = ranked[:TOPK]
        chosen[seq] = dict(selected=sel, candidates=ranked)
        print(f"[{seq}] {len(cands)} candidate windows; selected: {sel}", flush=True)
        # ---- render check videos via ffmpeg (this cv2 build's GStreamer VideoWriter cannot encode mp4 at all);
        # frames staged locally, encoded locally, result copied to the Drive res dir ----
        import subprocess
        outdir = os.path.join(S["res"], "multiview_windows"); os.makedirs(outdir, exist_ok=True)
        for wi, c in enumerate(sel):
            name = f"win{wi+1}_pid{c['pid0']}-{c['pid1']}.mp4"
            stage = tempfile.mkdtemp(prefix=f"mv_{seq}_")
            k = 0
            for p in range(c["pid0"], c["pid1"] + 1, 2):              # 5 Hz
                if p not in pairs: continue
                fp = cam_path(S["cam"], pairs[p])
                if fp is None: continue
                im = cv2.imread(fp)
                if im is None: continue
                n = objc.get(f"{seq}_{p:06d}", 0)
                bar = np.full((34, im.shape[1], 3), 250, np.uint8)
                cv2.putText(bar, f"{seq} pid {p}  cam {pairs[p]}  objs {n}  win motion {c['motion']:.3f}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 40, 50), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(stage, f"{k:04d}.jpg"), np.vstack([bar, im])); k += 1
            tmp = os.path.join(stage, "out.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS_OUT),
                            "-i", os.path.join(stage, "%04d.jpg"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", tmp], check=True)
            out = os.path.join(outdir, name); shutil.copy2(tmp, out); shutil.rmtree(stage)
            print(f"[{seq}] wrote {out} ({os.path.getsize(out)//1024} KB, {k} frames)", flush=True)
    jp = os.path.join(CODE, "sync", "src", "fusion", "late_fusion", "analysis", "multiview_windows.json")
    json.dump(chosen, open(jp, "w"), indent=1)
    print("saved", jp)

if __name__ == "__main__":
    main()
