"""ASPECT-CHANGE window selection: find video windows where the camera sees MORE THAN ONE SIDE of vehicles --
close passes (object bearing sweeps widely as ego passes) and turns/loops (ego heading changes, revisiting the
scene from another direction). Drive-alongside windows (one aspect per car) score ZERO here.

Camera-only signals, nothing extracted:
  yaw rate     phase-correlation x-shift between consecutive sampled frames, UPPER image half (far field is
               rotation-dominated); yaw_deg ~ shift_px / fx_scaled * 57.3. Cumulative |net yaw| over the window
               finds turns and loops.
  close pass   greedy nearest-neighbour tracks over the per-frame GT labels (fwd, lat); a track whose bearing
               sweeps > SWEEP_MIN while its range dips < NEAR_MAX is an object the ego actually passed around.
  score        net|yaw|/90 + passes (windows need at least one of the two)

Outputs top-K windows/seq -> aspect check videos (res/<capture>/multiview_windows/aspect_win*.mp4) and
appends an 'aspect' section to late_fusion/multiview_windows.json.
Run: python sync/dev/multiview_aspect_windows.py
"""
import os, csv, json, glob, collections, tempfile, shutil, subprocess
import numpy as np, cv2

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEQS = {
    "0511": dict(res=os.path.join(CODE, "sync", "res", "2026.05.11", "21.37.07"),
                 cam=os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.11/camera_extracted")),
    "0508": dict(res=os.path.join(CODE, "sync", "res", "2026.05.08", "17-34-27"),
                 cam=os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.08/camera_extracted")),
}
LABELS = os.path.join(CODE, "sync", "res", "benchmark_v1", "labels")
MCACHE = os.path.expanduser("~/.cache/ti_setup_frames")
STRIDE = 5                  # yaw sampled every 5 pids (~0.5 s)
WIN = 350                   # window length in pids (~35 s): long enough to span a turn
TOPK = 2
FPS_OUT = 5
SWEEP_MIN = 70.0            # deg of bearing sweep for a close pass
NEAR_MAX = 6.0              # the track must come at least this close (m)
FX_FULL = 604.0             # colour fx at 640 px

def load_pairs(res):
    m = {}
    for r in csv.DictReader(open(os.path.join(res, "camera_sync_pairs.csv"))):
        m[int(r["pair_idx"])] = int(r["camera_color_idx"])
    return m

def load_boxes():
    per = collections.defaultdict(list)                     # sid -> [(fwd_c, lat_c)]
    for f in glob.glob(os.path.join(LABELS, "*.json")):
        for sid, boxes in json.load(open(f)).items():
            for b in boxes:
                if "forward" in b and "lateral" in b:
                    per[sid].append(((b["forward"][0] + b["forward"][1]) / 2,
                                     (b["lateral"][0] + b["lateral"][1]) / 2))
    return per

def cam_path(camdir, idx):
    for ext in ("jpg", "png", "jpeg"):
        p = os.path.join(camdir, "color", f"{idx:06d}.{ext}")
        if os.path.exists(p): return p
    return None

def gray_top(camdir, idx, w=160):
    p = cam_path(camdir, idx)
    if p is None: return None
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    im = cv2.resize(im, (w, int(w * im.shape[0] / im.shape[1])))
    return im[: im.shape[0] // 2].astype(np.float32)        # far field: rotation-dominated flow

def yaw_scan(seq, S, pairs):
    cachep = os.path.join(MCACHE, f"mv_yaw_{seq}.npz")
    if os.path.exists(cachep):
        Z = np.load(cachep); return dict(zip(Z["pids"].tolist(), Z["yaw"].tolist()))
    fx_s = FX_FULL * 160 / 640
    grid = sorted(pairs)[::STRIDE]
    yaw = {}; prev = None; prev_p = None
    for i, p in enumerate(grid):
        g = gray_top(S["cam"], pairs[p])
        if g is not None and prev is not None and g.shape == prev.shape:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g)
            yaw[p] = float(np.degrees(dx / fx_s))           # deg of heading change since prev sample
        if g is not None: prev, prev_p = g, p
        if i % 1000 == 0: print(f"[{seq}] yaw scan {i}/{len(grid)}", flush=True)
    ks = sorted(yaw)
    np.savez(cachep, pids=np.array(ks), yaw=np.array([yaw[k] for k in ks]))
    return yaw

def track_passes(seq, p0, p1, per):
    """greedy NN tracks over labels in [p0,p1] -> number of tracks with bearing sweep>SWEEP_MIN & range<NEAR_MAX."""
    tracks = []                                             # each: dict(last=(f,l), pid, bear=[..], rng=[..])
    for p in range(p0, p1 + 1):
        obs = per.get(f"{seq}_{p:06d}", [])
        used = set()
        for t in tracks:
            if p - t["pid"] > 10: continue
            best = None
            for i, (f, l) in enumerate(obs):
                if i in used: continue
                d = np.hypot(f - t["last"][0], l - t["last"][1])
                if d < 1.2 and (best is None or d < best[0]): best = (d, i)
            if best:
                i = best[1]; used.add(i); f, l = obs[i]
                t["last"] = (f, l); t["pid"] = p
                t["bear"].append(np.degrees(np.arctan2(l, f))); t["rng"].append(np.hypot(f, l))
        for i, (f, l) in enumerate(obs):
            if i not in used:
                tracks.append(dict(last=(f, l), pid=p, bear=[np.degrees(np.arctan2(l, f))], rng=[np.hypot(f, l)]))
    n = 0
    for t in tracks:
        if len(t["bear"]) < 8: continue
        if (max(t["bear"]) - min(t["bear"])) >= SWEEP_MIN and min(t["rng"]) <= NEAR_MAX: n += 1
    return n

def main():
    per = load_boxes()
    out_all = json.load(open(os.path.join(CODE, "sync", "src", "fusion", "late_fusion", "analysis", "multiview_windows.json")))
    for seq, S in SEQS.items():
        pairs = load_pairs(S["res"])
        yaw = yaw_scan(seq, S, pairs)
        gp = sorted(yaw)
        cands = []
        step = WIN // STRIDE // 4
        for a in range(0, len(gp) - WIN // STRIDE, step):
            w = gp[a:a + WIN // STRIDE]
            if w[-1] - w[0] > WIN * 2: continue
            net = abs(sum(yaw[p] for p in w))
            if net < 45: continue                           # pre-filter: needs real heading change
            labs = np.mean([len(per.get(f"{seq}_{p:06d}", [])) > 0 for p in range(w[0], w[-1], 5)])
            if labs < 0.5: continue
            passes = track_passes(seq, w[0], w[-1], per)
            score = net / 90.0 + passes
            cands.append(dict(pid0=int(w[0]), pid1=int(w[-1]), net_yaw_deg=round(net, 0),
                              close_passes=passes, score=round(score, 2)))
        cands.sort(key=lambda c: -c["score"])
        ranked = []
        for c in cands:
            if all(c["pid0"] > s["pid1"] or c["pid1"] < s["pid0"] for s in ranked): ranked.append(c)
            if len(ranked) == 8: break
        sel = ranked[:TOPK]
        out_all.setdefault(seq, {})
        if isinstance(out_all[seq], list): out_all[seq] = dict(selected=out_all[seq])
        out_all[seq]["aspect"] = ranked
        print(f"[{seq}] aspect candidates {len(cands)}; selected {sel}", flush=True)
        # render
        outdir = os.path.join(S["res"], "multiview_windows"); os.makedirs(outdir, exist_ok=True)
        for wi, c in enumerate(sel):
            stage = tempfile.mkdtemp(prefix=f"mva_{seq}_"); k = 0
            for p in range(c["pid0"], c["pid1"] + 1, 2):
                if p not in pairs: continue
                fp = cam_path(S["cam"], pairs[p])
                if fp is None: continue
                im = cv2.imread(fp)
                if im is None: continue
                n = len(per.get(f"{seq}_{p:06d}", []))
                bar = np.full((34, im.shape[1], 3), 250, np.uint8)
                cv2.putText(bar, f"{seq} pid {p}  objs {n}  net yaw {c['net_yaw_deg']:.0f} deg  passes {c['close_passes']}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 40, 50), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(stage, f"{k:04d}.jpg"), np.vstack([bar, im])); k += 1
            tmp = os.path.join(stage, "out.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS_OUT),
                            "-i", os.path.join(stage, "%04d.jpg"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", tmp], check=True)
            out = os.path.join(outdir, f"aspect_win{wi+1}_pid{c['pid0']}-{c['pid1']}.mp4")
            shutil.copy2(tmp, out); shutil.rmtree(stage)
            print(f"[{seq}] wrote {out} ({os.path.getsize(out)//1024} KB, {k} frames)", flush=True)
    jp = os.path.join(CODE, "sync", "src", "fusion", "late_fusion", "analysis", "multiview_windows.json")
    json.dump(out_all, open(jp, "w"), indent=1)
    print("saved", jp)

if __name__ == "__main__":
    main()
