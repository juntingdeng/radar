"""Multiview window tooling (merged). Subcommands: windows | aspect | previews
  windows    select + extract candidate multi-view windows from a sequence
  aspect     census of ego-turn / vehicle-turn aspect-diversity events
  previews   render preview.mp4 into every multiview_v1 window dir
Run: python multiview.py <subcommand>"""
import os, csv, json, glob, collections, tempfile, shutil
import numpy as np, cv2
import os, csv, json, glob, collections, tempfile, shutil, subprocess
import os, csv, json, glob, subprocess, tempfile, shutil, collections

# ======== multiview_windows.py ========
# Multi-view WINDOW SELECTION from the full sequential recordings (camera-only pass, no lidar/radar extraction).
# 
# The L2 benchmark frames were sampled for scene DIVERSITY (RL env), so they are nearly useless for multi-view.
# This selects continuous windows where the ego is actually MOVING and GT labels are dense, using only data that is
# already local: the extracted camera streams (pid -> colour frame via camera_sync_pairs.csv) and benchmark_v1 labels.
# 
#   motion metric   mean |diff| of consecutive downsampled grey frames, sampled every STRIDE pids (~1 s)
#   window score    median motion over the window  x  log(1 + mean labelled objects/frame)
#   output          top-K windows per sequence -> multiview_windows.json + an annotated check VIDEO per window
#                   (res/<capture>/multiview_windows/win<i>_pid<a>-<b>.mp4, 5 Hz, pid/motion/object-count overlay)
# 
# After visual approval, the lidar/radar extraction job targets exactly these pid ranges.
# Run: python sync/dev/multiview_windows.py
def windows_main():
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


# ======== multiview_aspect_windows.py ========
# ASPECT-CHANGE window selection: find video windows where the camera sees MORE THAN ONE SIDE of vehicles --
# close passes (object bearing sweeps widely as ego passes) and turns/loops (ego heading changes, revisiting the
# scene from another direction). Drive-alongside windows (one aspect per car) score ZERO here.
# 
# Camera-only signals, nothing extracted:
#   yaw rate     phase-correlation x-shift between consecutive sampled frames, UPPER image half (far field is
#                rotation-dominated); yaw_deg ~ shift_px / fx_scaled * 57.3. Cumulative |net yaw| over the window
#                finds turns and loops.
#   close pass   greedy nearest-neighbour tracks over the per-frame GT labels (fwd, lat); a track whose bearing
#                sweeps > SWEEP_MIN while its range dips < NEAR_MAX is an object the ego actually passed around.
#   score        net|yaw|/90 + passes (windows need at least one of the two)
# 
# Outputs top-K windows/seq -> aspect check videos (res/<capture>/multiview_windows/aspect_win*.mp4) and
# appends an 'aspect' section to late_fusion/multiview_windows.json.
# Run: python sync/dev/multiview_aspect_windows.py
def aspect_main():

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


# ======== multiview_previews.py ========
# Render a preview.mp4 inside every multiview_v1 window dir (real-time 10 Hz, annotated with pid + labelled
# object count + the window's selection reason). Dataset lives on local disk, so ffmpeg writes directly.
# Run: python sync/dev/multiview_previews.py
def previews_main():

    CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OUT = os.path.expanduser("~/CMU/Research/mobile_data_local/multiview_v1/0508")
    LABELS = os.path.join(CODE, "sync", "res", "benchmark_v1", "labels")
    FPS = 10

    objc = collections.Counter()
    for f in glob.glob(os.path.join(LABELS, "*.json")):
        for sid, boxes in json.load(open(f)).items():
            if sid.startswith("0508"):
                objc[int(sid.split("_")[1])] = max(objc[int(sid.split("_")[1])],
                                                   sum(1 for b in boxes if "forward" in b))
    man = json.load(open(os.path.join(OUT, "manifest.json")))
    for w in man["windows"]:
        wdir = w["dir"]; name = os.path.basename(wdir)
        stage = tempfile.mkdtemp(prefix="mvprev_"); k = 0
        for fp in sorted(glob.glob(os.path.join(wdir, "color", "p*.*"))):
            pid = int(os.path.basename(fp)[1:7])
            im = cv2.imread(fp)
            if im is None: continue
            bar = np.full((34, im.shape[1], 3), 250, np.uint8)
            why = w["why"][:58]
            cv2.putText(bar, f"{name}  pid {pid}  objs {objc.get(pid, 0)}  | {why}",
                        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 40, 50), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(stage, f"{k:05d}.jpg"), np.vstack([bar, im])); k += 1
        out = os.path.join(wdir, "preview.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", os.path.join(stage, "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out], check=True)
        shutil.rmtree(stage)
        print(f"{name}: preview.mp4 {os.path.getsize(out)//1024} KB ({k} frames)", flush=True)
    print("done")


if __name__ == '__main__':
    import sys as _s
    _c = _s.argv[1] if len(_s.argv) > 1 else ''
    _m = {'windows': windows_main, 'aspect': aspect_main, 'previews': previews_main}
    if _c not in _m:
        print(__doc__ or 'subcommands: ' + ', '.join(_m)); _s.exit(1)
    _m[_c]()
