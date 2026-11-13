"""Benchmark pack builders (merged). Subcommands: gold | split"""
import sys

# ======== dev/pack_gold.py ========
# Pack the GOLD (hand/auto-verified) benchmark subset into a standalone folder on /Volumes/1.5T.
# 
# 351 frames = 147 glare_verified (2026.05.11 night) + 204 smoke_verified (2026.10.24 smoke).
# Self-contained: no repo needed to read it back (loader.py is dependency-free numpy/cv2).
# 
# Format: one CONSOLIDATED compressed npz per condition (the 1.5T is exFAT w/ 512 KB clusters -- thousands
# of tiny files would waste ~half the space), holding per frame:
#     color_jpg  : JPEG bytes (object)            (camera RGB, reused from extracted imgs / smoke_window npz -- no bag decode)
#     lidar      : (N,3) float16 (object)         (ego-frame XYZ, metres; from scan caches or 05.11 pcap)
#     radar_ra   : (511,181) float16              (DENSE range-azimuth power map = the smoke-penetration field; beamformed)
#     radar_dets : structured range/az/vel/power (object)  (CFAR detections WITH Doppler velocity)
#     modality   : per-frame source indices (camera/lidar/radar idx into the raw captures)
# plus labels/*.json (verified GT), calib/*.json, README.md, loader.py.
# 
# Run:  python sync/dev/pack_gold.py   [--conds glare smoke]
def gold_main():
    import os, sys, json, io, time, argparse
    import numpy as np, cv2
    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    BM = os.path.join(REPO, "sync", "res", "benchmark_v1")
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames")
    OUT = "/Volumes/1.5T/benchmark_v1_gold"
    EXT511 = os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.11/camera_extracted/color")
    H5_1024 = "/Volumes/1.5T/2026.10.24/radar/2026.10.24-15.02.30.h5"
    CONDS = {
        "glare": dict(seq="2026.05.11", label="test_glare_verified.json",
                      pcap=json.load(open(f"{BM}/sources.json"))["2026.05.11"]["lidar_pcap"],
                      meta=json.load(open(f"{BM}/sources.json"))["2026.05.11"]["lidar_meta"],
                      h5=json.load(open(f"{BM}/sources.json"))["2026.05.11"]["radar_h5"]),
        "smoke": dict(seq="2026.10.24", label="test_smoke_verified.json", pcap=None, meta=None, h5=H5_1024),
    }
    def _radar(seq_cfg):
        """Fresh RadarDetector pointed at this seq's h5 (mirrors dataset.py _set_globals)."""
        sys.path.insert(0, os.path.join(REPO, "sync/src/fusion/radar_first"))
        import common
        if seq_cfg["pcap"]: common.PCAP = seq_cfg["pcap"]; common.META = seq_cfg["meta"]
        common.H5 = seq_cfg["h5"]
        from detector import RadarDetector
        return RadarDetector(processing="legacy")
    def _ra_map(R, radar_idx):
        rd = R.range_doppler(int(radar_idx))[:R.maxR]            # (nR,nA,nD) complex
        B = np.einsum("qa,rad->rqd", R.sv, rd)                   # beamform -> (nR,nAng,nD)
        power = np.sum(np.abs(B) ** 2, axis=2)                   # (511,181) range x azimuth power
        return (10 * np.log10(power + 1e-9)).astype(np.float16)  # dB -> fits f16 (linear power overflowed f16)
    def _lidar_source(seq_cfg):
        sys.path.insert(0, os.path.join(REPO, "sync/src/fusion/radar_first"))
        import common
        common.PCAP = seq_cfg["pcap"]; common.META = seq_cfg["meta"]
        return common.LidarSource()
    def pack_condition(cond, samples):
        c = CONDS[cond]; seq = c["seq"]
        labels = json.load(open(f"{BM}/labels/{c['label']}"))
        ids = sorted(labels)
        outp = f"{OUT}/{cond}_verified.npz"
        if os.path.exists(outp):
            print(f"[{cond}] already packed -> {outp} (delete to rebuild)"); return
        print(f"[{cond}] {len(ids)} frames  seq={seq}")

        # ---- camera (no bag) ----
        color = []
        if cond == "smoke":
            SW = np.load(f"{CACHE}/smoke_window.npz", allow_pickle=True)
            p2k = {m["pair"]: k for k, m in enumerate(json.loads(str(SW["meta"])))}
            cams = SW["cams"]
        for sid in ids:
            if cond == "glare":
                cci = samples[sid]["modality"]["camera_color_idx"]
                color.append(np.frombuffer(open(f"{EXT511}/{cci:06d}.jpg", "rb").read(), np.uint8))  # reuse jpg bytes
            else:
                pair = int(sid.split("_")[1]); rgb = cams[p2k[pair]]
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                color.append(buf.reshape(-1))
        print(f"[{cond}]   camera done ({len(color)})")

        # ---- lidar ----
        lidar = [None] * len(ids); need_pcap = []
        for i, sid in enumerate(ids):
            li = samples[sid]["modality"]["lidar_idx"]
            for cand in (f"{CACHE}/smoke1024_lidar/l_{li}.npy", f"{CACHE}/lidar_scan_cache/l_{li}.npy"):
                if os.path.exists(cand): lidar[i] = np.load(cand).astype(np.float16); break
            if lidar[i] is None: need_pcap.append(i)
        if need_pcap:
            print(f"[{cond}]   {len(need_pcap)} lidar scans from pcap (indexing {os.path.basename(c['pcap'])}) ...")
            t0 = time.time(); L = _lidar_source(c); print(f"[{cond}]   lidar index built {time.time()-t0:.0f}s")
            for n, i in enumerate(need_pcap):
                lidar[i] = L.xyz(int(samples[ids[i]]["modality"]["lidar_idx"]))[:, :3].astype(np.float16)
                if n % 25 == 0: print(f"[{cond}]     lidar {n}/{len(need_pcap)}", flush=True)
        print(f"[{cond}]   lidar done")

        # ---- radar (RA map + dets) ----
        import contextlib
        R = _radar(c); ra = np.zeros((len(ids), 511, 181), np.float16); dets = []
        t0 = time.time()
        for i, sid in enumerate(ids):
            ri = samples[sid]["modality"]["radar_idx"]
            with contextlib.redirect_stdout(io.StringIO()):           # mute per-frame CFAR chatter
                ra[i] = _ra_map(R, ri); d, _ = R.detect(int(ri))
            dets.append(d)
            if i % 40 == 0: print(f"[{cond}]     radar {i}/{len(ids)}  ({(time.time()-t0):.0f}s)", flush=True)
        print(f"[{cond}]   radar done ({time.time()-t0:.0f}s)")

        mods = [samples[sid]["modality"] for sid in ids]
        os.makedirs(OUT, exist_ok=True)
        np.savez_compressed(outp, ids=np.array(ids), color_jpg=np.array(color, dtype=object),
                            lidar=np.array(lidar, dtype=object), radar_ra=ra,
                            radar_dets=np.array(dets, dtype=object), modality=np.array(mods, dtype=object),
                            seq=seq, condition=cond, rres=R.rres, angs=np.linspace(-90, 90, 181).astype(np.float32),
                            ra_range_m=np.arange(511) * R.rres)
        print(f"[{cond}] SAVED {outp}  ({os.path.getsize(outp)/1e6:.0f} MB)")
    LOADER = '''"""Standalone loader for benchmark_v1_gold -- no repo required, just numpy + opencv."""
    import os, json, numpy as np, cv2
    class GoldSet:
        """ds = GoldSet(<this dir>, "glare"|"smoke");  ds[i] -> dict(color,lidar,radar_ra,radar_dets,boxes,calib)."""
        LAB = {"glare": "test_glare_verified.json", "smoke": "test_smoke_verified.json"}
        def __init__(self, root, condition):
            z = np.load(os.path.join(root, f"{condition}_verified.npz"), allow_pickle=True)
            self.root = root; self.ids = [str(x) for x in z["ids"]]
            self.color = z["color_jpg"]; self.lidar = z["lidar"]; self.ra = z["radar_ra"]
            self.dets = z["radar_dets"]; self.modality = z["modality"]
            self.rres = float(z["rres"]); self.angs = z["angs"]; self.ra_range_m = z["ra_range_m"]; self.seq = str(z["seq"])
            self.labels = json.load(open(os.path.join(root, self.LAB[condition])))
            self.calib = json.load(open(os.path.join(root, f"{self.seq}.json")))
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            sid = self.ids[i]
            return dict(id=sid, seq=self.seq,
                color=cv2.imdecode(self.color[i], cv2.IMREAD_COLOR)[:, :, ::-1],  # HxWx3 RGB
                lidar=self.lidar[i].astype(np.float32),        # (N,3) ego XYZ metres
                radar_ra=self.ra[i].astype(np.float32),        # (511,181) range x azimuth power in dB (linear=10**(x/10))
                radar_dets=self.dets[i],                       # struct: range,az,vel,power (Doppler!)
                boxes=self.labels.get(sid, []), modality=dict(self.modality[i]), calib=self.calib)
        def radar_bev_xy(self, i):
            """Dense RA map -> Cartesian (forward, lateral, power) for BEV plotting."""
            ra = self.ra[i].astype(np.float32); R, A = np.meshgrid(self.ra_range_m, np.deg2rad(self.angs), indexing="ij")
            return (R*np.cos(A)).ravel(), (R*np.sin(A)).ravel(), ra.ravel()
    if True:
        import sys; ds = GoldSet(os.path.dirname(os.path.abspath(__file__)), sys.argv[1] if len(sys.argv) > 1 else "smoke")
        print(len(ds), "frames"); s = ds[0]
        print(s["id"], "color", s["color"].shape, "lidar", s["lidar"].shape, "radar_ra", s["radar_ra"].shape,
              "dets", len(s["radar_dets"]), "boxes", len(s["boxes"]))
    '''
    README = """# benchmark_v1_gold -- hand/auto-verified multimodal detection subset

    **351 frames**, person/vehicle BEV detection, two adverse conditions, fully self-contained
    (camera + lidar + radar + verified GT + calibration). No external captures needed.

    | condition | frames | boxes | sequence | scene |
    |---|---|---|---|---|
    | `glare` | 147 | 234 | 2026.05.11 (night) | headlight glare, hand-verified subset |
    | `smoke` | 204 | 255 | 2026.10.24 (day)   | person walking through smoke, hand+auto-verified |

    ## Files
    - `glare_verified.npz`, `smoke_verified.npz` -- the packed sensor data (one row per frame)
    - `test_glare_verified.json`, `test_smoke_verified.json` -- GT boxes (keyed by sample_id)
    - `2026.05.11.json`, `2026.10.24.json` -- camera/lidar/radar calibration (intrinsics + extrinsics)
    - `loader.py` -- dependency-free reader (numpy + opencv)

    ## Per frame
    - **color**: camera RGB (480x640), JPEG. Reused from the synced capture -- no re-decode.
    - **lidar**: (N,3) float16 ego-frame XYZ (metres), range-filtered.
    - **radar_ra**: (511,181) float16 DENSE range-azimuth power map, **in dB** (range step `rres`=0.0489 m,
      az -90..90 deg). This is the smoke-penetration field -- the dense returns, not just detections.
      Linear power = `10**(radar_ra/10)`.
    - **radar_dets**: CFAR detections with **Doppler velocity** (range, az, vel, power).
    - **boxes**: verified GT. Common fields BOTH conditions: `forward`/`lateral` (BEV metres, [lo,hi]) + `label`.
      `smoke` additionally has camera-space `bbox_xyxy`, `box3d`, `source`, `score`, `reliable` (its labels came from
      the YOLO->lidar-frustum pipeline); `glare` is BEV-only (night camera is near-blind, so it was labeled in lidar
      space). For BEV detection use `forward`/`lateral` uniformly.

    ## Load
    ```python
    from loader import GoldSet
    ds = GoldSet(".", "smoke")          # or "glare"
    s = ds[0]                           # dict: color, lidar, radar_ra, radar_dets, boxes, calib
    fx, ly, pw = ds.radar_bev_xy(0)     # dense radar -> Cartesian BEV scatter
    ```

    ## Provenance / caveats
    - `smoke` GT: 252 auto (YOLO->lidar-frustum, BEV-position-gated) + 3 hand-picked closest-pass frames
      (`source: hand_bev_pick`). See the main repo's `VERIFY_smoke.md`.
    - `glare` GT: hand-verified subset of the night-glare sequence.
    - Radar processed with `rng40_res0004_tx3_rx4` config (same rig both sequences). `radar_ra` is amplitude
      (sqrt of beamformed power) stored float16; `vel` in `radar_dets` is single-frame aliased (legacy mode).
    """
    def write_sidecars():
        os.makedirs(OUT, exist_ok=True)
        open(f"{OUT}/loader.py", "w").write(LOADER)
        open(f"{OUT}/README.md", "w").write(README)
    def main():
        ap = argparse.ArgumentParser(); ap.add_argument("--conds", nargs="+", default=["glare", "smoke"]); a = ap.parse_args()
        samples = {json.loads(l)["sample_id"]: json.loads(l) for l in open(f"{BM}/samples.jsonl")}
        os.makedirs(OUT, exist_ok=True)
        import shutil
        for cond in a.conds:
            shutil.copy(f"{BM}/labels/{CONDS[cond]['label']}", f"{OUT}/")
            shutil.copy(f"{BM}/calib/{CONDS[cond]['seq']}.json", f"{OUT}/")
        write_sidecars()
        for cond in a.conds:
            pack_condition(cond, samples)
        print("\nALL DONE ->", OUT)
    if True:
        main()

# ======== dev/pack_split.py ========
# Pack the benchmark_v1 TEST splits into a standalone folder (3-seq generalisation of pack_gold.py).
# 
# Splits: test_smoke (212, 10.24) + test_glare (3456, 05.11) + test_clear (5325, 05.08+05.11) = 8993 frames.
# Output: /Volumes/1.5T/benchmark_v1_test/ -- same self-contained format as the gold pack (camera JPEG,
# lidar (N,3) f16, radar range-azimuth power in dB, CFAR dets w/ Doppler, verified-or-auto GT, calib).
# 
# Sharded <split>__<seq>__<chunk>.npz (<=CHUNK frames each) so: (a) RAM stays bounded, (b) RESUMABLE
# (an existing shard is skipped), (c) each shard is ONE seq -> one radar h5 context (the detector binds
# H5 at import, so we set detector.H5 per seq and never interleave sequences). loader.py concatenates shards.
# 
# NO camera bag decode (reuses extracted jpgs / smoke_window.npz). Run: python sync/dev/pack_split.py [--splits ...]
def split_main():
    import os, sys, json, io, time, argparse, contextlib, glob
    import numpy as np, cv2
    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    BM = os.path.join(REPO, "sync", "res", "benchmark_v1")
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames")
    MML = os.path.expanduser("~/CMU/Research/mobile_data_local")
    OUTBASE = "/Volumes/1.5T"
    def OUTDIR(sp): return f"{OUTBASE}/benchmark_v1_{sp}"   # each split -> its OWN standalone folder
    CHUNK = 1500
    SRC = json.load(open(f"{BM}/sources.json"))
    L508 = f"{MML}/2026.05.08"
    SEQ = {
        "2026.05.08": dict(cam=f"{MML}/2026.05.08/camera_extracted/color",
                           pcap=f"{L508}/lidar/2026058_1734_47_OS-0-128_122519000479-001.pcap",
                           meta=f"{L508}/lidar/2026058_1734_47_OS-0-128_122519000479.json",
                           h5=f"{L508}/radar/2026.05.08-17.34.27-002.h5"),
        "2026.05.11": dict(cam=f"{MML}/2026.05.11/camera_extracted/color", pcap=SRC["2026.05.11"]["lidar_pcap"],
                           meta=SRC["2026.05.11"]["lidar_meta"], h5=SRC["2026.05.11"]["radar_h5"]),
        "2026.10.24": dict(cam_npz=f"{CACHE}/smoke_window.npz", pcap=None, meta=None,
                           h5="/Volumes/1.5T/2026.10.24/radar/2026.10.24-15.02.30.h5", lidar_cache=f"{CACHE}/smoke1024_lidar"),
    }
    LCACHE = [f"{CACHE}/lidar_scan_cache", f"{CACHE}/day08_lidar", f"{CACHE}/smoke1024_lidar"]
    SEQ_ORDER = ["2026.05.08", "2026.05.11", "2026.10.24"]
    def _lidar_cache(li):
        for d in LCACHE:
            p = f"{d}/l_{li}.npy"
            if os.path.exists(p): return np.load(p).astype(np.float16)
        return None
    def _ra_map(R, radar_idx):
        rd = R.range_doppler(int(radar_idx))[:R.maxR]
        B = np.einsum("qa,rad->rqd", R.sv, rd)
        power = np.sum(np.abs(B) ** 2, axis=2)
        return (10 * np.log10(power + 1e-9)).astype(np.float16)             # dB (linear power overflows f16)
    def seq_sources(seq):
        """Set per-seq globals and build (LidarSource|None, RadarDetector). detector.H5 set explicitly (import-bound)."""
        sys.path.insert(0, os.path.join(REPO, "sync/src/fusion/radar_first"))
        import common
        c = SEQ[seq]
        if c.get("pcap"): common.PCAP = c["pcap"]; common.META = c["meta"]
        common.H5 = c["h5"]
        import detector as D
        D.H5 = c["h5"]                                                      # <- the import-bound name the detector actually reads
        R = D.RadarDetector(processing="legacy")
        L = common.LidarSource() if c.get("pcap") else None
        return L, R
    def build_shard(seq, sids, samples, L, R, cams_npz):
        color, lidar, dets = [], [], []
        ra = np.zeros((len(sids), 511, 181), np.float16)
        for n, sid in enumerate(sids):
            m = samples[sid]["modality"]
            # camera
            if "cam" in SEQ[seq]:
                color.append(np.frombuffer(open(f"{SEQ[seq]['cam']}/{m['camera_color_idx']:06d}.jpg", "rb").read(), np.uint8))
            else:
                rgb = cams_npz[0][cams_npz[1][int(sid.split('_')[1])]]
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                color.append(buf.reshape(-1))
            # lidar
            lp = _lidar_cache(m["lidar_idx"])
            if lp is None and L is not None: lp = L.xyz(int(m["lidar_idx"]))[:, :3].astype(np.float16)
            lidar.append(lp if lp is not None else np.zeros((0, 3), np.float16))
            # radar
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ra[n] = _ra_map(R, m["radar_idx"]); d, _ = R.detect(int(m["radar_idx"]))
                dets.append(d)
            except Exception as e:
                dets.append(np.zeros(0, dtype=[("range", "f8"), ("az", "f8"), ("vel", "f8"), ("power", "f8")]))
            if n % 200 == 0: print(f"      {n}/{len(sids)}", flush=True)
        mods = [samples[s]["modality"] for s in sids]
        rel = [bool(samples[s].get("reliable", False)) for s in sids]
        return dict(ids=np.array(sids), seqs=np.array([seq] * len(sids)), color_jpg=np.array(color, dtype=object),
                    lidar=np.array(lidar, dtype=object), radar_ra=ra, radar_dets=np.array(dets, dtype=object),
                    modality=np.array(mods, dtype=object), reliable=np.array(rel),
                    rres=R.rres, angs=np.linspace(-90, 90, 181).astype(np.float32), ra_range_m=np.arange(511) * R.rres)
    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--splits", nargs="+", default=["test_smoke", "test_glare", "test_clear"])
        a = ap.parse_args()
        samples = {json.loads(l)["sample_id"]: json.loads(l) for l in open(f"{BM}/samples.jsonl")}
        labels = {sp: json.load(open(f"{BM}/labels/{sp}.json")) for sp in a.splits}
        split_seqs = {sp: sorted(set(samples[s]["seq"] for s in labels[sp])) for sp in a.splits}
        import shutil
        for sp in a.splits:                                                  # each split -> its OWN self-contained folder
            od = OUTDIR(sp); os.makedirs(od, exist_ok=True)
            shutil.copy(f"{BM}/labels/{sp}.json", f"{od}/")
            for seq in split_seqs[sp]: shutil.copy(f"{BM}/calib/{seq}.json", f"{od}/")
            open(f"{od}/loader.py", "w").write(LOADER)
            open(f"{od}/README.md", "w").write(readme_for(sp, split_seqs[sp], len(labels[sp])))

        # frames grouped (split, seq); process seq-by-seq so each seq's pcap/h5 context is built ONCE and reused across splits
        todo = {seq: {sp: sorted([s for s in labels[sp] if samples[s]["seq"] == seq]) for sp in a.splits} for seq in SEQ_ORDER}
        for seq in SEQ_ORDER:
            work = {sp: ids for sp, ids in todo[seq].items() if ids}
            if not work: continue
            pending = any(not os.path.exists(f"{OUTDIR(sp)}/{sp}__{seq}__{ci:02d}.npz")
                          for sp in work for ci in range((len(work[sp]) + CHUNK - 1) // CHUNK))
            if not pending: print(f"[{seq}] all shards present, skip"); continue
            print(f"[{seq}] building sources ...", flush=True); t0 = time.time()
            L, R = seq_sources(seq)
            cams_npz = None
            if "cam_npz" in SEQ[seq]:
                Z = np.load(SEQ[seq]["cam_npz"], allow_pickle=True)
                cams_npz = (Z["cams"], {mt["pair"]: k for k, mt in enumerate(json.loads(str(Z["meta"])))})
            print(f"[{seq}] sources ready {time.time()-t0:.0f}s", flush=True)
            for sp, ids in work.items():
                for ci in range(0, len(ids), CHUNK):
                    shard = f"{OUTDIR(sp)}/{sp}__{seq}__{ci//CHUNK:02d}.npz"
                    if os.path.exists(shard): print(f"  [{sp}][{seq}] chunk {ci//CHUNK} exists, skip"); continue
                    print(f"  [{sp}][{seq}] chunk {ci//CHUNK} ({len(ids[ci:ci+CHUNK])} frames) ...", flush=True); tc = time.time()
                    d = build_shard(seq, ids[ci:ci+CHUNK], samples, L, R, cams_npz)
                    np.savez_compressed(shard, **d)
                    print(f"  [{sp}][{seq}] SAVED {os.path.basename(shard)} ({os.path.getsize(shard)/1e6:.0f}MB, {time.time()-tc:.0f}s)", flush=True)
        print("\nALL DONE ->", ", ".join(OUTDIR(sp) for sp in a.splits))
    LOADER = '''"""Standalone loader for a benchmark_v1 split -- LAZY per-shard (bounded memory). numpy + opencv only.
    Loads one shard (~1 GB) at a time, so an 18k-frame split streams through a DataLoader without a 12 GB blow-up.
    Sequential iteration touches each shard once; random access reloads the owning shard (fine for shuffled epochs
    if your sampler is shard-blocked, slower if fully random)."""
    import os, re, json, glob, numpy as np, cv2
    class Bench:
        """ds = Bench(<dir>, "<split>"); len(ds); ds[i] -> dict(color,lidar,radar_ra,radar_dets,boxes,seq,calib,reliable)."""
        def __init__(self, root, split):
            self.root = root; self.split = split
            self.shards = sorted(glob.glob(os.path.join(root, f"{split}__*.npz")))
            assert self.shards, f"no shards for {split} in {root}"
            self.index = []; seqs = set()
            for si, sh in enumerate(self.shards):
                with np.load(sh, allow_pickle=True) as z:                 # read only the small per-shard index
                    n = len(z["ids"])
                    if si == 0: self.rres = float(z["rres"]); self.angs = z["angs"]; self.ra_range_m = z["ra_range_m"]
                self.index += [(si, li) for li in range(n)]
                m = re.search(r"__(\\d{4}\\.\\d{2}\\.\\d{2})__", os.path.basename(sh))
                if m: seqs.add(m.group(1))
            self.labels = json.load(open(os.path.join(root, f"{split}.json")))
            self.calib = {s: json.load(open(os.path.join(root, f"{s}.json"))) for s in seqs}
            self._ci = -1; self._cache = None
        def __len__(self): return len(self.index)
        def _shard(self, si):
            if self._ci != si:
                z = np.load(self.shards[si], allow_pickle=True)
                self._cache = {k: z[k] for k in ("ids","seqs","color_jpg","lidar","radar_ra","radar_dets","modality","reliable")}
                z.close(); self._ci = si
            return self._cache
        def __getitem__(self, i):
            si, li = self.index[i]; a = self._shard(si); sid = str(a["ids"][li]); seq = str(a["seqs"][li])
            return dict(id=sid, seq=seq, reliable=bool(a["reliable"][li]),
                color=cv2.imdecode(a["color_jpg"][li], cv2.IMREAD_COLOR)[:, :, ::-1],
                lidar=a["lidar"][li].astype(np.float32), radar_ra=a["radar_ra"][li].astype(np.float32),  # radar_ra in dB
                radar_dets=a["radar_dets"][li], boxes=self.labels.get(sid, []),
                modality=dict(a["modality"][li]), calib=self.calib[seq])
        def radar_bev_xy(self, i):
            si, li = self.index[i]; ra = self._shard(si)["radar_ra"][li].astype(np.float32)
            R, A = np.meshgrid(self.ra_range_m, np.deg2rad(self.angs), indexing="ij")
            return (R*np.cos(A)).ravel(), (R*np.sin(A)).ravel(), ra.ravel()
    if True:
        import sys; ds = Bench(os.path.dirname(os.path.abspath(__file__)), sys.argv[1] if len(sys.argv) > 1 else "train")
        print(len(ds), "frames"); s = ds[0]
        print(s["id"], s["seq"], "color", s["color"].shape, "lidar", s["lidar"].shape, "ra", s["radar_ra"].shape, "boxes", len(s["boxes"]))
    '''
    def readme_for(split, seqs, nframes):
        return f"""# benchmark_v1_{split} -- multimodal detection `{split}` split (standalone)

    {nframes} frames, self-contained (camera+lidar+radar+GT+calib). No repo / raw captures needed.
    Sequences: {', '.join(seqs)}.

    Per-frame format (identical to the other benchmark_v1 packs):
    - **color**: camera RGB (480x640) JPEG
    - **lidar**: (N,3) float16 ego XYZ metres
    - **radar_ra**: (511,181) float16 range-azimuth power **in dB** (linear = 10**(x/10)), range step 0.0489 m, az -90..90 deg
    - **radar_dets**: CFAR detections (range, az, vel, power) -- has Doppler velocity
    - **boxes**: GT; common fields `forward`/`lateral` (BEV [lo,hi] m) + `label`. Clear/smoke seqs also have camera
      `bbox_xyxy` + provenance; night (05.11) is BEV-only. Each frame carries `seq` + a `reliable` flag.

    Data is sharded `{split}__<seq>__<chunk>.npz`; the loader concatenates them.

    ```python
    from loader import Bench
    ds = Bench(".", "{split}")
    s = ds[0]            # color, lidar, radar_ra, radar_dets, boxes, seq, calib, reliable
    ```

    Sibling packs on the same drive: `benchmark_v1_gold` (351 hand-verified eval frames),
    `benchmark_v1_test` (test_clear/glare/smoke). NOTE: 05.08 capture starts INDOORS -- some early
    clear-split frames are indoor scans (filter if undesired).
    """
    if True:
        main()

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'gold': gold_main, 'split': split_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
