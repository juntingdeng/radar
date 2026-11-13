"""10.24 smoke dataset tools (merged). Subcommands: pick_bev | add_1024"""
import sys

# ======== dev/pick_smoke_bev.py ========
# Hand-pick the BEV position for the 3 closest-pass 10.24 smoke frames the auto pipeline lost.
# 
# Pairs 2041/2044/2045: the green-pants subject is large & clearly visible in the camera, but the
# lidar frustum assigned a bad ~0.8 m depth (no good alternative box), so they were dropped from the
# gold set. Camera AZIMUTH is well constrained; only the RANGE is wrong -> pick the true location.
# 
# Two panels per frame:
#   [ camera + YOLO 2D box ]          [ lidar BEV: lateral (X) vs forward (Y) ]
# The BEV shows the lidar points (above-ground highlighted), the camera-ray guide (person lies along
# it), and the current bad box (red). CLICK in the BEV panel where the person is -> green box.
# 
# Keys:  click = set/update pick   r = reset   n/enter = confirm & next   s = skip frame   q = quit & save
# Saves corrected boxes (source 'hand_bev_pick', reliable=True) into labels/test_smoke_verified.json.
def pick_bev_main():
    import os, json, numpy as np
    import matplotlib                                   # same backend selection as the calibration pickers
    _cur = matplotlib.get_backend().lower()             # (ensure_interactive_backend in sync/src/calibration/interactive.py)
    if not any(k in _cur for k in ("macosx", "qt", "tk", "gtk", "wx")):
        for _cand in ("macosx", "qtagg", "qt5agg", "tkagg"):
            try:
                matplotlib.use(_cand, force=True); print(f"Using interactive matplotlib backend: {_cand}"); break
            except Exception:
                continue
    import matplotlib.pyplot as plt
    HERE = os.path.dirname(os.path.abspath(__file__))
    CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
    C = os.path.expanduser("~/.cache/ti_setup_frames")
    BM = os.path.join(CODE, "sync", "res", "benchmark_v1")
    TARGET_PAIRS = [2041, 2044, 2045]
    PERSON_HALF = 0.35   # half-extent (m) of the person footprint written as forward/lateral span
    S = np.load(os.path.join(C, "smoke_window.npz"), allow_pickle=True)
    cams = S["cams"]; meta = json.loads(str(S["meta"])); p2k = {m["pair"]: k for k, m in enumerate(meta)}
    cams = S["cams"]; meta = json.loads(str(S["meta"])); p2k = {m["pair"]: k for k, m in enumerate(meta)}
    cams = S["cams"]; meta = json.loads(str(S["meta"])); p2k = {m["pair"]: k for k, m in enumerate(meta)}
    full = json.load(open(os.path.join(BM, "labels", "test_smoke.json")))
    calib = json.load(open(os.path.join(BM, "calib", "2026.10.24.json")))
    def _K(ci):
        if isinstance(ci, dict): return np.array([[ci["fx"],0,ci["cx"]],[0,ci["fy"],ci["cy"]],[0,0,1]], float)
        a = np.asarray(ci, float); return a.reshape(3, 3) if a.size == 9 else \
            np.array([[a[0],0,a[2]],[0,a[1],a[3]],[0,0,1]], float)
    K = _K(calib["color_intrinsics"]); Kinv = np.linalg.inv(K)
    K = _K(calib["color_intrinsics"]); Kinv = np.linalg.inv(K)
    T = np.asarray(calib["camera_to_lidar_auto"], float).reshape(4, 4)   # cam-frame pt -> lidar frame
    R, t = T[:3, :3], T[:3, 3]
    def cam_ray(u, v):
        """Pixel (u,v) -> (origin xy, dir xy) of the viewing ray in lidar BEV (x=forward, y=lateral)."""
        d_cam = Kinv @ np.array([u, v, 1.0]); d_lid = R @ d_cam
        return t[:2], d_lid[:2] / (np.linalg.norm(d_lid[:2]) + 1e-9)
    def lidar(pair): return np.load(os.path.join(C, "smoke1024_lidar", f"l_{pair}.npy"))
    picks = {}   # pair -> (forward, lateral)
    class Picker:
        def __init__(self):
            self.i = 0; self.pick = None; self._pick_art = []
            self.fig, (self.axc, self.axb) = plt.subplots(1, 2, figsize=(15, 7))
            self.fig.canvas.mpl_connect("button_press_event", self.onclick)
            self.fig.canvas.mpl_connect("key_press_event", self.onkey)
            self.draw_frame()
        def cur(self): return TARGET_PAIRS[self.i]
        def box(self):  # the dropped auto box for this pair
            return full[f"1024_{self.cur():06d}"][0]
        def draw_frame(self):
            """STATIC scene for the current frame -- drawn once (not inside the click callback)."""
            self.axc.clear(); self.axb.clear(); self._pick_art = []; pair = self.cur(); b = self.box()
            self.axc.imshow(cams[p2k[pair]]); x0, y0, x1, y1 = b["bbox_xyxy"]
            self.axc.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec="lime", lw=2))
            self.axc.set_title(f"camera  pair {pair}  (YOLO score {b.get('score',0):.2f})"); self.axc.axis("off")
            L = lidar(pair); m = (L[:,0] > 0) & (L[:,0] < 32) & (np.abs(L[:,1]) < 16)
            Lv = L[m]; above = Lv[:,2] > -1.0
            self.axb.scatter(Lv[~above,1], Lv[~above,0], s=1, c="#cccccc", alpha=.4)      # ground-ish
            self.axb.scatter(Lv[above,1],  Lv[above,0],  s=3, c="#1f77b4", alpha=.6)      # above-ground
            o, d = cam_ray((x0+x1)/2, y1)
            ss = np.array([0, 30]); ray = o[None,:] + ss[:,None]*d[None,:]
            self.axb.plot(ray[:,1], ray[:,0], "g--", lw=1, alpha=.7, label="camera ray")
            fc = np.mean(b["forward"]); lc = np.mean(b["lateral"])
            self.axb.plot(lc, fc, "rx", ms=12, mew=3, label=f"auto {fc:.1f}m (bad)")
            self.axb.set_xlim(16, -16); self.axb.set_ylim(0, 32); self.axb.set_aspect("equal")
            self.axb.set_xlabel("lateral Y (m)  [left +<-]"); self.axb.set_ylabel("forward X (m)")
            self.axb.set_title(f"[{self.i+1}/{len(TARGET_PAIRS)}] CLICK person  |  r reset  n next  s skip  q quit")
            self.axb.legend(loc="upper right", fontsize=8); self.axb.grid(alpha=.2)
            self.fig.canvas.draw_idle()
        def _clear_pick(self):
            for a in self._pick_art:
                try: a.remove()
                except Exception: pass
            self._pick_art = []
        def onclick(self, e):
            inb = e.inaxes is self.axb
            print(f"click: in_bev={inb} xy=({e.xdata},{e.ydata})", flush=True)
            if inb and e.xdata is not None:
                self.pick = (float(e.ydata), float(e.xdata))                 # (forward, lateral)
                self._clear_pick(); pf, pl = self.pick                       # incremental artist add + draw_idle (macosx-safe)
                r = plt.Rectangle((pl-PERSON_HALF, pf-PERSON_HALF), 2*PERSON_HALF, 2*PERSON_HALF, fill=False, ec="lime", lw=2.5)
                self.axb.add_patch(r); self._pick_art.append(r)
                m, = self.axb.plot(pl, pf, "g+", ms=16, mew=2.5); self._pick_art.append(m)
                self.fig.canvas.draw_idle()
        def onkey(self, e):
            if e.key in ("n", "enter"):
                if self.pick is not None: picks[self.cur()] = self.pick; print(f"pair {self.cur()} -> fwd {self.pick[0]:.2f} lat {self.pick[1]:.2f}", flush=True)
                self.advance()
            elif e.key == "s": print(f"pair {self.cur()} skipped", flush=True); self.advance()
            elif e.key == "r": self.pick = None; self._clear_pick(); self.fig.canvas.draw_idle()
            elif e.key == "q": plt.close(self.fig)
        def advance(self):
            self.i += 1; self.pick = None
            if self.i >= len(TARGET_PAIRS): plt.close(self.fig)
            else: self.draw_frame()
    print(f"backend: {matplotlib.get_backend()}  (need a GUI backend; a window should open)")
    _picker = Picker()      # MUST keep a strong reference: mpl's callback registry holds the handler WEAKLY,
    plt.show()              # so an inline Picker() would be GC'd and the click callbacks would silently die.
    if picks:
        ver = json.load(open(os.path.join(BM, "labels", "test_smoke_verified.json")))
        for pair, (pf, pl) in picks.items():
            sid = f"1024_{pair:06d}"; b = dict(full[sid][0])
            b["forward"] = [pf-PERSON_HALF, pf+PERSON_HALF]; b["lateral"] = [pl-PERSON_HALF, pl+PERSON_HALF]
            b["source"] = "hand_bev_pick"; b["reliable"] = True
            b.setdefault("box3d", {})["reliable"] = True
            ver.setdefault(sid, []).append(b)
        json.dump(ver, open(os.path.join(BM, "labels", "test_smoke_verified.json"), "w"), indent=1)
        nb = sum(len(v) for v in ver.values())
        print(f"\nsaved {len(picks)} hand-picked boxes -> test_smoke_verified.json  (now {len(ver)} frames / {nb} boxes)")
    else:
        print("no picks made; nothing written")

# ======== dev/add_1024_to_benchmark.py ========
# Add the 10.24 green-pants SMOKE window to benchmark_v1 as seq 2026.10.24 (condition 'smoke', split 'test_smoke').
# 
# Keeps the reliable yolo_lidar_frustum PERSON boxes (drops the lidar_radar_proposal 'car' clutter -- trees/poles/rig,
# not real cars in this pedestrian scene). Writes:
#   benchmark_v1/labels/test_smoke.json   sample_id -> person boxes (benchmark box format, already produced by the pipeline)
#   benchmark_v1/calib/2026.10.24.json    copy of the 15.02.30 calibration
#   benchmark_v1/samples.jsonl            APPENDS the 2026.10.24 smoke samples (does not touch 0508/0511)
def add_1024_main():
    import os, csv, json, shutil
    HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
    HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
    BM = os.path.join(CODE, "sync", "res", "benchmark_v1")
    RES = os.path.join(CODE, "sync", "res", "2026.10.24", "15.02.30")
    AUTO = os.path.join(RES, "annotations", "smoke_auto.json")
    sync = {}
    for r in csv.DictReader(open(os.path.join(RES, "camera_sync_pairs.csv"))):
        try:
            sync[int(r["pair_idx"])] = dict(camera_color_idx=int(float(r["camera_color_idx"])),
                camera_depth_idx=int(float(r["camera_depth_idx"])) if r.get("camera_depth_idx") else int(float(r["camera_color_idx"])),
                lidar_idx=int(r["lidar_idx"]), radar_idx=int(r["radar_idx"]))
        except (ValueError, KeyError):
            pass
    obj = json.load(open(AUTO))["objects"]
    labels = {}; samples = []; npers = 0
    labels = {}; samples = []; npers = 0
    labels = {}; samples = []; npers = 0
    for pair_s, boxes in obj.items():
        pair = int(pair_s)
        persons = [b for b in boxes if b.get("label") == "person"]      # drop the car-proposal clutter
        if not persons or pair not in sync: continue
        sid = f"1024_{pair:06d}"
        labels[sid] = persons; npers += len(persons)
        rel = any(b.get("box3d", {}).get("reliable") or b.get("reliable") for b in persons)
        samples.append(dict(sample_id=sid, seq="2026.10.24", pair_idx=pair, modality=sync[pair],
                            condition="smoke", reliable=bool(rel), split="test_smoke", n_boxes=len(persons)))
    os.makedirs(os.path.join(BM, "labels"), exist_ok=True); os.makedirs(os.path.join(BM, "calib"), exist_ok=True)
    os.makedirs(os.path.join(BM, "labels"), exist_ok=True); os.makedirs(os.path.join(BM, "calib"), exist_ok=True)
    json.dump(labels, open(os.path.join(BM, "labels", "test_smoke.json"), "w"), indent=1)
    shutil.copy(os.path.join(RES, "calibration.json"), os.path.join(BM, "calib", "2026.10.24.json"))
    sp = os.path.join(BM, "samples.jsonl")
    existing = [l for l in open(sp) if l.strip() and json.loads(l)["seq"] != "2026.10.24"]
    with open(sp, "w") as f:
        f.writelines(existing)
        for s in samples: f.write(json.dumps(s) + "\n")
    print(f"added seq 2026.10.24 'smoke': {len(samples)} samples, {npers} person boxes -> test_smoke")
    print(f"  labels: {os.path.join(BM,'labels','test_smoke.json')}")
    print(f"  calib:  {os.path.join(BM,'calib','2026.10.24.json')}")
    print(f"  samples.jsonl now: {len(existing)} prev + {len(samples)} smoke = {len(existing)+len(samples)}")

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'pick_bev': pick_bev_main, 'add_1024': add_1024_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
