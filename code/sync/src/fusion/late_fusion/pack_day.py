"""Data-scale training frames from the benchmark_v1_train/val PACKS (local copies) -- 0508 day only.

The packs store radar as POLAR range-azimuth (511x181 dB, 0.0489 m step, az -90..90); the pipeline consumes the
Cartesian BEV grid (fa 0..30 x la -30..30, 0.1 m). Conversion = inverse-mapped bilinear sampling (cv2.remap).
CONVENTIONS ARE NOT TRUSTED: calibrate_conversion() converts overlapping pids all four ways (az sign x range
flip) and locks the combo that correlates with the stored day.npz BEVs (the azimuth-mirror trap bit this project
before). The chosen combo is cached in pack_day_convention.json.

pack_day_frames(stride) -> list of frame dicts (pid, lidar_pts (N,4), radar_pts via data_l2.radar_points on the
converted BEV, union_gt) with the L2 day TEST pids +/- GUARD excluded and L2 train-frame pids deduped.
Run standalone to (re)calibrate + report:  python pack_day.py
"""
import os, json, glob
import numpy as np
import data_l2
HERE = os.path.dirname(os.path.abspath(__file__))
PACKS = [os.path.expanduser("~/CMU/Research/mobile_data_local/benchmark_v1_train"),
         os.path.expanduser("~/CMU/Research/mobile_data_local/benchmark_v1_val")]
RRES = 0.0489; NR = 511; NA = 181                       # pack polar geometry (README)
GUARD = 50
CONV = os.path.join(HERE, "pack_day_convention.json")

def _maps(az_sign, r_flip):
    """cv2.remap maps from Cartesian (fa 0..30, la -30..30, 0.1 m) into the polar RA image."""
    fa = np.arange(0, 30.0001, 0.1); la = np.arange(-30, 30.0001, 0.1)
    FA, LA = np.meshgrid(fa, la, indexing="ij")
    r = np.hypot(FA, LA) / RRES                          # range bin (float)
    a = (np.degrees(np.arctan2(az_sign * LA, FA)) + 90.0)  # az bin (1 deg)
    if r_flip: r = (NR - 1) - r
    return a.astype(np.float32), r.astype(np.float32)   # map_x = az col, map_y = range row

def ra_to_bev(ra, az_sign=1, r_flip=False):
    import cv2
    mx, my = _maps(az_sign, r_flip)
    return cv2.remap(np.asarray(ra, np.float32), mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

REFNPZ = {"2026.05.08": "day.npz", "2026.05.11": "scenes50.npz"}   # env npz with stored BEVs for validation

def _shards(seq="2026.05.08"):
    out = []
    for p in PACKS:
        out += sorted(glob.glob(os.path.join(p, f"*__{seq}__*.npz")))
    return out

def calibrate_conversion(seq="2026.05.08", max_check=8):
    """Pick (az_sign, r_flip) by correlation against the env npz BEVs on overlapping pids. Caches per seq."""
    S = np.load(os.path.join(data_l2.RL_ENV, REFNPZ[seq]), allow_pickle=True)
    meta = json.loads(str(S["meta"])); ref = {m["pid"]: i for i, m in enumerate(meta)}
    scores = {}
    checked = 0
    for sh in _shards(seq):
        z = np.load(sh, allow_pickle=True)
        ids = [str(x) for x in z["ids"]]
        for li, sid in enumerate(ids):
            pid = int(sid.split("_")[1])
            if pid not in ref: continue
            ra = z["radar_ra"][li].astype(np.float32)
            tgt = np.nan_to_num(S["bevs"][ref[pid]].astype(np.float32))
            m = tgt > 1e-6
            for az in (1, -1):
                for rf in (False, True):
                    bev = ra_to_bev(ra, az, rf)
                    c = np.corrcoef(bev[m], tgt[m])[0, 1]
                    scores.setdefault((az, rf), []).append(float(c))
            checked += 1
            if checked >= max_check: break
        z.close()
        if checked >= max_check: break
    if not scores: raise RuntimeError("no overlapping pids between packs and day.npz")
    table = {f"az{az:+d}_rflip{rf}": round(float(np.mean(v)), 3) for (az, rf), v in scores.items()}
    best = max(scores, key=lambda k: np.mean(scores[k]))
    print("convention correlations:", table, "-> chosen", best)
    assert float(np.mean(scores[best])) > 0.8, f"best convention only r={np.mean(scores[best]):.2f} -- do not trust"
    allc = json.load(open(CONV)) if os.path.exists(CONV) else {}
    allc[seq] = dict(az_sign=best[0], r_flip=bool(best[1]), corr=table)
    json.dump(allc, open(CONV, "w"), indent=1)
    return best

def _convention(seq="2026.05.08"):
    if os.path.exists(CONV):
        c = json.load(open(CONV))
        if seq in c: return (c[seq]["az_sign"], c[seq]["r_flip"])
        if "az_sign" in c and seq == "2026.05.08":            # migrate the old single-seq cache format
            return (c["az_sign"], c["r_flip"])
    return calibrate_conversion(seq)

def pack_frames(seq="2026.05.08", stride=10):
    """-> training frame dicts from the packs for `seq`, guards applied, L2-benchmark pids deduped."""
    az, rf = _convention(seq)
    trp = data_l2.load_split()
    import pickle
    P = pickle.load(open(os.path.join(HERE, "proposals_l2.pkl"), "rb"))
    pref = seq[5:7] + seq[8:10]
    l2_pids = {f["pid"] for f in P["frames"] if f["seq"] == seq}
    test_pids = sorted(p for p in l2_pids if p not in trp)
    labels = {}
    for p in PACKS:
        for fn in glob.glob(os.path.join(p, "*.json")):
            if os.path.basename(fn) in ("train.json", "val.json"):
                for k, v in json.load(open(fn)).items():
                    if k.startswith(pref): labels.setdefault(k, []).extend(v)
    fa = np.arange(0, 30.0001, 0.1); la = np.arange(-30, 30.0001, 0.1)
    frames = []; skipped = 0
    for sh in _shards(seq):
        z = np.load(sh, allow_pickle=True)
        ids = [str(x) for x in z["ids"]]
        for li in range(0, len(ids), stride):
            sid = ids[li]; pid = int(sid.split("_")[1])
            if pid in l2_pids or any(abs(pid - t) <= GUARD for t in test_pids):
                skipped += 1; continue
            gt = data_l2._gt_for(sid, labels)
            if not gt: continue
            lid = z["lidar"][li].astype(np.float32)
            if lid.shape[1] == 3:
                lid = np.concatenate([lid, np.ones((len(lid), 1), np.float32)], 1)
            bev = ra_to_bev(z["radar_ra"][li].astype(np.float32), az, rf)
            frames.append(dict(pid=pid, fid=f"extra:{seq}:{pid}", cls=f"extra_{pref}", seq=seq,
                               lidar_pts=lid, radar_pts=data_l2.radar_points(bev, la, fa),
                               union_gt=data_l2._union(gt)))
        z.close()
        print(f"  {os.path.basename(sh)}: {len(frames)} frames so far", flush=True)
    print(f"pack_frames[{seq}]: {len(frames)} frames (stride {stride}; {skipped} skipped by guard/dedup)")
    return frames

def pack_day_frames(stride=10):                           # back-compat alias
    return pack_frames("2026.05.08", stride)

if __name__ == "__main__":
    import sys
    seq = sys.argv[1] if len(sys.argv) > 1 else "2026.05.08"
    calibrate_conversion(seq)
    fr = pack_frames(seq, stride=40)                     # quick standalone sanity at coarse stride
    f = fr[0]
    print(f"sample: pid {f['pid']} lidar {f['lidar_pts'].shape} radar_pts {f['radar_pts'].shape} "
          f"gt {len(f['union_gt'])}")
