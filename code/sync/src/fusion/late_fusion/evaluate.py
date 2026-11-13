"""Evaluation suite -- one file, subcommands (merged from eval_bev_single / eval2d_cam / eval_l2).

  python evaluate.py bev [lidar|radar] [condition] [viz_n]   single voxel detector: stage funnel
      (signal -> head-fires -> ceiling -> operating) + per-object debug + BEV viz
  python evaluate.py cam2d [stride] [viz_n]                  camera-only pure-2D eval: two-world GT
      (native bbox vs reprojection), forward bins, image overlays
  python evaluate.py l2 [viz_n]                              full system ladder: single sensors /
      equal / learned fusion, per-condition miss, mAP sweep, BEV scene viz

All three share the canonical split (split.json) and the benchmark label conventions.
"""
import os, sys

# ---- subcommand dispatch happens FIRST: merged sections parse sys.argv at import time ----
_CMD = None
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("bev", "cam2d", "l2"):
        print(__doc__); sys.exit(1)
    _CMD = sys.argv[1]
    sys.argv = [f"evaluate-{_CMD}"] + sys.argv[2:]

# ===================== BEV single-detector (was eval_bev_single.py) =====================
import numpy as np, torch
import data_l2, boxes as BX
from voxel_detector import VoxelDetector, decode
HERE = os.path.dirname(os.path.abspath(__file__))
if _CMD == "bev":                                   # section parses argv only when selected
    MOD = sys.argv[1] if len(sys.argv) > 1 else "lidar"
    COND = sys.argv[2] if len(sys.argv) > 2 else "day_clean"
    VIZN = int(sys.argv[3]) if len(sys.argv) > 3 else 8
RADIUS = {"car": 2., "truck": 2., "bus": 2., "person": 1., "bicycle": 1., "motorcycle": 1.}
IN_CH = {"lidar": 4, "radar": 3}

def pts_of(f):
    return f["lidar_pts"] if MOD == "lidar" else f["radar_pts"]

def signal_in(f, box):
    if MOD == "lidar":
        P = f["lidar_pts"]
        m = (P[:, 0] >= box[0]) & (P[:, 0] <= box[2]) & (P[:, 1] >= box[1]) & (P[:, 1] <= box[3]) & (P[:, 2] > 0.1)
        return int(m.sum()), int(m.sum()) >= 5
    P = f["radar_pts"]
    if len(P) == 0: return 0, False
    m = (P[:, 0] >= box[0]) & (P[:, 0] <= box[2]) & (P[:, 1] >= box[1]) & (P[:, 1] <= box[3])
    return int(m.sum()), int(m.sum()) >= 1

def bev_main():
    ck = torch.load(f"{HERE}/detectors_l2.pt", map_location="cpu")
    model = VoxelDetector(in_ch=IN_CH[MOD]); model.load_state_dict(ck[MOD]); model.eval()
    frames = data_l2.load_frames((COND,))
    trp = data_l2.load_split()
    splits = {"TEST": [f for f in frames if f["pid"] not in trp],
              "train": [f for f in frames if f["pid"] in trp]}
    print(f"{MOD} on {COND}: {len(splits['train'])} train / {len(splits['TEST'])} test frames")
    bins = [(0, 10), (10, 20), (20, 30), (30, 50)]
    viz_jobs = []
    for spname, fl in splits.items():
        funnel = dict(total=0, signal=0, scoring=0, proposal=0, operating=0)
        binrec = {b: [0, 0] for b in bins}
        rows = []
        for f in fl:
            pts = torch.tensor(pts_of(f)).float()
            if len(pts) == 0: continue
            with torch.no_grad(): out = model(pts)
            obj = torch.sigmoid(out[0])
            pb, ps = decode(model, out, score_thr=0.05, topk=100, nms_thr=0.5)     # permissive
            ob, os_ = decode(model, out, score_thr=0.3, topk=20, nms_thr=0.2)      # operating (pipeline)
            if spname == "TEST": viz_jobs.append((f, pb, ob))
            for u in f["union_gt"]:
                g = u["box"]; R = RADIUS.get(u.get("label", "car"), 2.)
                fc, lc = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
                nsig, has_sig = signal_in(f, g)
                px = int(np.clip((fc - model.fmin) / model.cell, 1, model.H - 2))
                py = int(np.clip((lc - model.lmin) / model.cell, 1, model.W - 2))
                mo = float(obj[px - 1:px + 2, py - 1:py + 2].max())
                dp = float(BX.center_dist_matrix(np.array([g]), pb).min()) if len(pb) else 999.
                do = float(BX.center_dist_matrix(np.array([g]), ob).min()) if len(ob) else 999.
                funnel["total"] += 1; funnel["signal"] += has_sig
                funnel["scoring"] += (has_sig and mo >= 0.3)
                funnel["proposal"] += (dp <= R); funnel["operating"] += (do <= R)
                for lo, hi in bins:
                    if lo <= fc < hi: binrec[(lo, hi)][0] += (do <= R); binrec[(lo, hi)][1] += 1
                if spname == "TEST":
                    stage = ("OK" if do <= R else "loc/NMS" if dp <= R else
                             "decode" if mo >= 0.3 else "scoring" if has_sig else "NO-SIGNAL")
                    rows.append(f"  pid {f['pid']} {u.get('label','?'):7s} fwd {fc:5.1f} | "
                                f"sig {nsig:4d} obj@GT {mo:.2f} | perm-dist {min(dp,99):5.2f} "
                                f"op-dist {min(do,99):5.2f} | {stage}")
        t = funnel["total"]
        print(f"\n[{spname}] funnel over {t} GT objects:")
        print(f"  sensor SIGNAL in box : {funnel['signal']/max(1,t):.2f}")
        print(f"  head fires (obj>=.3) : {funnel['scoring']/max(1,t):.2f}")
        print(f"  PROPOSAL ceiling     : {funnel['proposal']/max(1,t):.2f}   (permissive decode, within radius)")
        print(f"  OPERATING recall     : {funnel['operating']/max(1,t):.2f}   (pipeline decode -- eval_l2's view)")
        print(f"  recall by forward: " + "  ".join(f"{lo}-{hi}m {h}/{n}" for (lo, hi), (h, n) in binrec.items() if n))
        if spname == "TEST":
            print("per-object debug (TEST):")
            print("\n".join(rows))
    # ---- viz: labelling-style BEV panels for the test frames ----
    if VIZN:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        VIZDIR = f"{HERE}/eval_bev_viz"; os.makedirs(VIZDIR, exist_ok=True)
        def rect(ax, b, **kw): ax.add_patch(Rectangle((b[1], b[0]), b[3]-b[1], b[2]-b[0], fill=False, **kw))
        for f, pb, ob in viz_jobs[:VIZN]:
            fig, ax = plt.subplots(figsize=(8, 9), facecolor="white")
            L = f["lidar_pts"]; m = (L[:, 0] > -2) & (L[:, 0] < 34) & (np.abs(L[:, 1]) < 18) & (L[:, 2] > 0.05)
            ax.scatter(L[m, 1], L[m, 0], s=2, c="0.6", linewidths=0)
            if MOD == "radar" and len(f["radar_pts"]):
                P = f["radar_pts"]; ax.scatter(P[:, 1], P[:, 0], s=6, c="#3A5CA8", alpha=0.7, linewidths=0)
            for b in pb: rect(ax, b, ec="#9BB8A5" if MOD == "lidar" else "#A9B8D8", lw=0.9, alpha=0.9)
            for b in ob: rect(ax, b, ec="#2E6E45" if MOD == "lidar" else "#3A5CA8", lw=2.4)
            G, _ = np.array([u["box"] for u in f["union_gt"]]) if f["union_gt"] else np.zeros((0, 4)), None
            for u in f["union_gt"]:
                rect(ax, u["box"], ec="#C0392B", lw=1.8, ls="--")
                ax.text(u["box"][3], u["box"][0] - 0.35, u.get("label", "?"), color="#C0392B", fontsize=8, ha="right")
            ax.set_xlim(17, -17); ax.set_ylim(-1, 33); ax.set_aspect("equal")
            ax.set_xlabel("lateral (m)  [+left]"); ax.set_ylabel("forward (m)")
            ax.set_title(f"{MOD} {f['fid']}   GT red-dashed | permissive thin | operating thick"
                         + ("  | radar cells blue dots" if MOD == "radar" else ""), fontsize=9.5)
            plt.tight_layout(); plt.savefig(f"{VIZDIR}/{MOD}_{f['fid'].replace(':', '_')}.png", dpi=120)
            plt.close(fig)
        print(f"\nviz: {HERE}/eval_bev_viz/ ({min(VIZN, len(viz_jobs))} test frames)")


# ===================== CAM 2D (was eval2d_cam.py) =====================
import csv, json
import numpy as np, cv2, torch
import camera_detector as CD, camera_proposals as CP, boxes as BX
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
RES = os.path.join(CODE, "sync", "res", "2026.05.08", "17-34-27")
CAM = os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.08/camera_extracted")
SEQ = "2026.05.08"; IOU2D = 0.5
if _CMD == "cam2d":                                 # section parses argv only when selected
    STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    VIZN = int(sys.argv[2]) if len(sys.argv) > 2 else 24
VIZDIR = os.path.join(HERE, "eval2d_viz")

def frame_gt(sid, lab, cb, img_hw):
    """-> list of dict(box2d, native, rng): one entry per union OBJECT visible to a 2D eval."""
    raw = [b for b in lab.get(sid, []) if "forward" in b and "lateral" in b
           and 0.0 <= (b["forward"][0] + b["forward"][1]) / 2 <= 50.0]
    if not raw: return []
    B = np.stack([BX.to_bev(b["forward"], b["lateral"]) for b in raw])
    used = np.zeros(len(raw), bool); out = []
    order = np.argsort([-(bb[2] - bb[0]) * (bb[3] - bb[1]) for bb in B])
    for i in order:
        if used[i]: continue
        ious = BX.iou_matrix(B[i:i + 1], B)[0]; mem = np.where((ious >= 0.3) & (~used))[0]; used[mem] = True
        hull = [B[mem][:, 0].min(), B[mem][:, 1].min(), B[mem][:, 2].max(), B[mem][:, 3].max()]
        rng = float((hull[0] + hull[2]) / 2)          # bin by FORWARD (matches the 0-50 m GT filter)
        nat = [raw[m]["bbox_xyxy"] for m in mem if raw[m].get("bbox_xyxy")]
        if nat:
            b2 = max(nat, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])); native = True
        else:
            b2 = CD.project_gt_box(dict(box=hull), cb, img_hw); native = False
            if b2 is None: continue
        out.append(dict(box2d=[float(x) for x in b2], native=native, rng=rng))
    return out

def cam2d_main():
    LABD = os.path.join(CODE, "sync", "res", "benchmark_v1", "labels")
    lab_te = {k: v for k, v in json.load(open(os.path.join(LABD, "test_clear.json"))).items()
              if k.startswith("0508")}
    lab_tr = {}
    for fn in ("train", "val"):                            # = the dayfull checkpoint's own training pool
        for k, v in json.load(open(os.path.join(LABD, fn + ".json"))).items():
            if k.startswith("0508"): lab_tr.setdefault(k, []).extend(v)
    pairs = {int(r["pair_idx"]): int(r["camera_color_idx"])
             for r in csv.DictReader(open(os.path.join(RES, "camera_sync_pairs.csv")))}
    tr_stride = max(1, STRIDE * len(lab_tr) // max(1, len(lab_te)))   # sample train to ~the same frame count
    splits = [("TEST (test_clear, held out)", lab_te, sorted(lab_te)[::STRIDE], True),
              ("train (train+val: the trained head's own data)", lab_tr, sorted(lab_tr)[::tr_stride], False)]
    print(f"2D eval: TEST {len(splits[0][2])} frames (stride {STRIDE} of {len(lab_te)}), "
          f"train {len(splits[1][2])} frames (stride {tr_stride} of {len(lab_tr)}); "
          f"GT = native bbox_xyxy first, BEV reprojection fallback")
    heads, z0 = None, None
    ck = os.path.join(HERE, "camera_detr_dayfull.pt")
    if os.path.exists(ck):
        d = torch.load(ck, map_location="cpu")
        heads = CD.DetrHeads(); heads.load_state_dict(d["heads"]); heads.eval(); z0 = d["z0"]
        print("trained cls head: camera_detr_dayfull.pt")
    cb = CD.geo_calib(SEQ)
    os.makedirs(VIZDIR, exist_ok=True)
    for spname, lab, sids, do_viz in splits:
        print(f"\n================ [{spname}] ================")
        viz_every = max(1, len(sids) // max(1, VIZN))
        bins = [(0, 10), (10, 20), (20, 30), (30, 50)]
        pooled = {"zeroshot": [], "trained": []}
        n_all = n_cam = 0
        tpw = {k: dict(all=0, cam=0) for k in pooled}                       # matched GT counts per world
        ceil = {w: {b: [0, 0] for b in bins} for w in ("cam", "blind")}     # candidate IoU>=0.5 ceiling by range
        binrec = {b: [0, 0] for b in bins}                                  # trained recall by range, cam world
        for n, sid in enumerate(sids):
            p = int(sid.split("_")[1])
            if p not in pairs: continue
            fp = os.path.join(CAM, "color", f"{pairs[p]:06d}.jpg")
            if not os.path.exists(fp): fp = fp[:-3] + "png"
            img = cv2.imread(fp)
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
            gts = frame_gt(sid, lab, cb, (H, W))
            n_all += len(gts); n_cam += sum(g["native"] for g in gts)
            cands = sorted(CD.detr_queries(img), key=lambda c: -c["score"])
            kept = []
            for c in cands:
                if all(CD._iou2d(c["bbox2d"], k["bbox2d"]) < 0.7 for k in kept): kept.append(c)
            for g in gts:                                                   # proposal ceiling per world
                best = max((CD._iou2d(c["bbox2d"], g["box2d"]) for c in kept), default=0.0)
                w = "cam" if g["native"] else "blind"
                for lo, hi in bins:
                    if lo <= g["rng"] < hi: ceil[w][(lo, hi)][0] += best >= IOU2D; ceil[w][(lo, hi)][1] += 1
            B2 = [c["bbox2d"] for c in kept]
            scores = {"zeroshot": [c["score"] for c in kept]}
            if heads is not None:
                depth = CP.aligned_depth(SEQ, pairs[p])
                with torch.no_grad():
                    scores["trained"] = [float(torch.sigmoid(heads(torch.tensor(c["q"][None]),
                        torch.tensor(np.array([CD.geom_feat(c["bbox2d"], c["score"], c["label"], (H, W),
                                                            cb, z0[SEQ], depth)], np.float32)))[0]))
                                         for c in kept]
            matched_viz = set()
            for nm, sc in scores.items():
                keep = [i for i, s in enumerate(sc) if s >= 0.5]
                order = sorted(keep, key=lambda i: -sc[i]); usedg = set()
                for i in order:
                    best, bj = 0.0, -1
                    for j, g in enumerate(gts):
                        if j in usedg: continue
                        v = CD._iou2d(B2[i], g["box2d"])
                        if v > best: best, bj = v, j
                    if best >= IOU2D:
                        usedg.add(bj); pooled[nm].append((sc[i], 1))
                        tpw[nm]["all"] += 1; tpw[nm]["cam"] += gts[bj]["native"]
                    else:
                        pooled[nm].append((sc[i], 0))
                if nm == ("trained" if heads is not None else "zeroshot"):
                    matched_viz = usedg
                    for j, g in enumerate(gts):
                        if not g["native"]: continue
                        for lo, hi in bins:
                            if lo <= g["rng"] < hi: binrec[(lo, hi)][0] += j in usedg; binrec[(lo, hi)][1] += 1
            # ---- viz: GT blue(native)/red(reproj fallback); predictions green(matched)/orange ----
            if VIZN and do_viz and n % viz_every == 0:
                vim = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for g in gts:
                    col = (200, 120, 0) if g["native"] else (40, 40, 230)
                    b = g["box2d"]; cv2.rectangle(vim, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
                    cv2.putText(vim, f"{g['rng']:.0f}m", (int(b[0]), max(12, int(b[1]) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                sc = scores.get("trained", scores["zeroshot"])
                for i, b in enumerate(B2):
                    if sc[i] < 0.5: continue
                    on = any(CD._iou2d(b, g["box2d"]) >= IOU2D for g in gts)
                    col = (70, 180, 60) if on else (30, 150, 250)
                    cv2.rectangle(vim, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
                    cv2.putText(vim, f"{sc[i]:.2f}", (int(b[0]), min(H - 4, int(b[3]) + 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                cv2.putText(vim, f"{sid}  GT: BLUE=native RED=reproj-fallback | pred: GREEN=matched ORANGE=no",
                            (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(VIZDIR, f"{sid}.jpg"), vim)
            if n % 40 == 0: print(f"  {n}/{len(sids)}", flush=True)

        print(f"\nunion objects: {n_all} visible to 2D eval;  cam world (native box): {n_cam};"
              f"  camera-blind (reproj fallback): {n_all - n_cam}")
        print(f"\nCANDIDATE CEILING by FORWARD distance (any DETR box IoU>={IOU2D}):")
        print(f"{'range':>8s}   {'cam world':>12s}   {'cam-blind':>12s}")
        for b in bins:
            fmt = lambda x: f"{x[0]/x[1]:.2f} ({x[1]:3d})" if x[1] else "     -    "
            print(f"{b[0]:3d}-{b[1]:2d} m   {fmt(ceil['cam'][b]):>12s}   {fmt(ceil['blind'][b]):>12s}")
        print()
        for nm, pl in pooled.items():
            if not pl: continue
            tp = sum(t for _, t in pl); fp_ = len(pl) - tp
            prec = tp / max(1, tp + fp_)
            r_all = tpw[nm]["all"] / max(1, n_all); r_cam = tpw[nm]["cam"] / max(1, n_cam)
            f1 = 2 * prec * r_all / max(1e-9, prec + r_all)
            t_ = 0; ap = 0.0
            for i, (s, t) in enumerate(sorted(pl, key=lambda x: -x[0]), 1):
                if t: t_ += 1; ap += t_ / i
            print(f"{nm:9s}: P {prec:.2f}  recall_ALL {r_all:.2f}  recall_CAM {r_cam:.2f}  F1(all) {f1:.2f}  "
                  f"AP(all) {ap/max(1, n_all):.2f}")
        print("trained-row recall by FORWARD distance (cam world):")
        for b, (h, n_) in binrec.items():
            if n_: print(f"  {b[0]:2d}-{b[1]:2d} m: {h}/{n_} = {h/n_:.2f}")
    if VIZN: print(f"visualisations: {VIZDIR}/")


# ===================== L2 system ladder (was eval_l2.py) =====================
import os, sys, pickle, numpy as np, collections
import boxes as BX, associate as AS
from fusion_head import group_feat, train_fusion, score_groups, BASE_DIM
def l2_main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    # Per-class BEV match radius (m): 2 m fits car scale + the ~1-2 m cross-modal localisation error, but is a different
    # location for a person -> person-scale classes match at 1 m. Used EVERYWHERE a proposal meets GT (miss rate,
    # fusion-head training positives, detection metrics) so train and eval share one definition of "correct".
    RADIUS = {"car": 2.0, "truck": 2.0, "bus": 2.0, "person": 1.0, "bicycle": 1.0, "motorcycle": 1.0}
    SWEEP = (0.25, 0.5, 1.0, 2.0)   # mAP radius multipliers -> cars 0.5/1/2/4 m, persons 0.25/0.5/1/2 m (nuScenes-style)
    CONDS = ["night_clean", "night_glare", "day_clean", "day_smoke"]; MODS = ["cam", "lidar", "radar"]

    import data_l2
    P = pickle.load(open(f"{HERE}/proposals_l2.pkl", "rb")); frames = P["frames"]
    train_pids = data_l2.load_split()                       # canonical split (split.json), not the pkl copy
    te = [f for f in frames if f["pid"] not in train_pids]
    tr = [f for f in frames if f["pid"] in train_pids]
    print(f"L2 eval: {len(frames)} frames (train {len(tr)} / test {len(te)}), cam backend = {P['backend']}")

    def gt_info(fr):
        """-> (G (n,4) union-GT boxes, R (n,) per-class match radii)."""
        if not fr["union_gt"]: return np.zeros((0, 4)), np.zeros((0,))
        G = np.array([u["box"] for u in fr["union_gt"]])
        R = np.array([RADIUS.get(u.get("label", "car"), 2.0) for u in fr["union_gt"]])
        return G, R

    def prov_of(u):   # a union object's primary provenance
        s = set(u["provs"]); return "lidarradar" if "lidarradar" in s else ("verified" if "verified" in s else "cam")

    def detected_by(fr, m):
        G, R = gt_info(fr); props = fr["props"][m]
        if len(G) == 0: return np.zeros(0, bool)
        if len(props) == 0: return np.zeros(len(G), bool)
        B = np.array([p[0] for p in props]); return BX.center_dist_matrix(G, B).min(1) <= R

    def cam_coverable(fr):
        """Per-GT bool: does the object project into the camera image at all (geometric coverage, auto extrinsic)?
        Used ONLY for the stratified coverable-recall table -- the headline metrics count every miss regardless of why."""
        import camera_detector as CD
        cb = CD.geo_calib(fr["seq"])
        return np.array([CD.project_gt_box(u, cb) is not None for u in fr["union_gt"]], bool)

    # ---------- miss rate / recall / recoverability ----------
    def invisibility(flist):
        rec = {m: collections.defaultdict(lambda: [0, 0]) for m in MODS}        # cond -> [hit, total]
        recby_prov = {m: collections.defaultdict(lambda: [0, 0]) for m in MODS}  # prov -> [hit,total]
        recov = {m: [0, 0] for m in MODS}                                        # [recovered, missed]
        for fr in flist:
            det = {m: detected_by(fr, m) for m in MODS}
            for k, u in enumerate(fr["union_gt"]):
                pr = prov_of(u)
                other = {m: any(det[o][k] for o in MODS if o != m) for m in MODS}
                for m in MODS:
                    rec[m][fr["cls"]][0] += int(det[m][k]); rec[m][fr["cls"]][1] += 1
                    recby_prov[m][pr][0] += int(det[m][k]); recby_prov[m][pr][1] += 1
                    if not det[m][k]:
                        recov[m][1] += 1; recov[m][0] += int(other[m])
        return rec, recby_prov, recov

    # ---------- centre-matched detection metrics + TP box quality ----------
    def aligned_iou(d, g):
        """BEV IoU after centring the detection on the GT: extent-only agreement (nuScenes-ASE style)."""
        wd, ld = d[2] - d[0], d[3] - d[1]; wg, lg = g[2] - g[0], g[3] - g[1]
        inter = min(wd, wg) * min(ld, lg); union = wd * ld + wg * lg - inter
        return inter / union if union > 0 else 0.0

    def match_frame(dets_b, dets_s, G, R, mult=1.0):
        """Greedy score-ordered one-to-one matching at per-class radius R*mult.
        -> ([(score, tp, ate, ase)], n_gt): ate = TP centre error (m), ase = 1 - centre-aligned IoU."""
        order = np.argsort(dets_s)[::-1]; matched = set(); res = []
        for i in order:
            tp = 0; ate = ase = float("nan")
            if len(G):
                dist = BX.center_dist_matrix(dets_b[i:i + 1], G)[0]
                ok = [j for j in np.argsort(dist) if dist[j] <= R[j] * mult and j not in matched]
                if ok:
                    j = ok[0]; tp = 1; matched.add(j)
                    ate = float(dist[j]); ase = 1.0 - aligned_iou(dets_b[i], G[j])
            res.append((float(dets_s[i]), tp, ate, ase))
        return res, len(G)

    def detect_metrics(flist, get_dets):
        """get_dets(fr) -> (boxes (n,4), scores (n,)). Pooled over flist: P/R/F1/AP + TP quality (ATE, ASE) at the
        per-class radius, and mAP = mean AP over the SWEEP radius multipliers."""
        dets = [(get_dets(fr), gt_info(fr)) for fr in flist]
        def pool(mult):
            pooled = []; ngt = 0
            for (b, s), (G, R) in dets:
                res, n = match_frame(b, s, G, R, mult); pooled += res; ngt += n
            return pooled, ngt
        def ap_of(pooled, ngt):
            """VOC-style AP: detections sorted by score, AP = sum of precision-at-each-TP / n_GT. No credit beyond the
            achieved recall. (The earlier sklearn padding -- missed GT appended at score -inf, label 1 -- handed every
            policy phantom bottom-rank precision, the more the sparser it emits: a detector with recall 0.14 scored
            AP 0.58. All APs are lower under this definition; only this definition is comparable across policies.)"""
            if ngt == 0 or not pooled: return float("nan")
            tp = 0; ap = 0.0
            for i, (s, t, *_) in enumerate(sorted(pooled, key=lambda x: -x[0]), 1):
                if t: tp += 1; ap += tp / i
            return ap / ngt
        pooled, ngt = pool(1.0)
        if ngt == 0: return dict(precision=0, recall=0, f1=0, AP=0, mAP=0, ate=float("nan"), ase=float("nan"), n=0)
        TP = sum(t for _, t, _, _ in pooled); FP = len(pooled) - TP
        prec = TP / (TP + FP) if TP + FP else 1.0; rec = TP / ngt
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        ate = float(np.nanmean([a for _, t, a, _ in pooled if t])) if TP else float("nan")
        ase = float(np.nanmean([a for _, t, _, a in pooled if t])) if TP else float("nan")
        aps = [ap_of(*((pooled, ngt) if m == 1.0 else pool(m))) for m in SWEEP]
        return dict(precision=round(prec, 3), recall=round(rec, 3), f1=round(f1, 3),
                    AP=round(float(ap_of(pooled, ngt)), 3), mAP=round(float(np.nanmean(aps)), 3),
                    ate=round(ate, 2), ase=round(ase, 2), n=ngt)

    def single_dets(m):
        def g(fr):
            p = fr["props"][m]
            if not p: return np.zeros((0, 4)), np.zeros((0,))
            return np.array([x[0] for x in p]), np.array([x[1] for x in p])
        return g

    def groups_of(fr): return AS.associate(fr["props"])
    def equal_dets(fr):
        """EQUAL-fusion baseline: cross-modal groups, score = unweighted mean of the present modalities' scores.
        Same detectors, same association as learned fusion -- the only difference is that nothing is learned."""
        gs = groups_of(fr)
        if not gs: return np.zeros((0, 4)), np.zeros((0,))
        B = []; S = []
        for g in gs:
            gf = AS.group_features(g); B.append(gf["box"]); S.append(sum(gf["scores"]) / max(1, gf["nmod"]))
        return np.array(B), np.array(S)

    # ---------- train learned fusion head on TRAIN-scene groups ----------
    def group_dataset(flist):
        X = []; y = []
        for fr in flist:
            G, R = gt_info(fr)
            for g in groups_of(fr):
                gf, x = group_feat(g, fr.get("ev")); X.append(x)
                d = BX.center_dist_matrix(gf["box"][None], G)[0] if len(G) else np.zeros(0)
                y.append(int(bool((d <= R).any())))
        return np.array(X), np.array(y)

    Xtr, ytr = group_dataset(tr)
    head = train_fusion(Xtr, ytr)                       # 17-d: confidences + agreement + size + raw evidence
    print(f"fusion head trained on {len(ytr)} train groups ({int(ytr.sum())} positive), feat dim {Xtr.shape[1]}")
    def learned_dets(fr, h=None, dims=None):
        gs = groups_of(fr)
        if not gs: return np.zeros((0, 4)), np.zeros((0,))
        s, b = score_groups(h if h is not None else head, gs, fr.get("ev"), dims); keep = s > 0
        return b[keep], 1 / (1 + np.exp(-s[keep]))

    # ================= REPORT =================
    rec, recby, recov = invisibility(te)
    print("\n=== MISS RATE  (1 - recall of union-GT objects) by condition [sensor invisibility] ===")
    print(f"{'condition':12s} {'n':>4s}   {'cam':>6s} {'lidar':>6s} {'radar':>6s}")
    for c in CONDS:
        n = rec['cam'][c][1]
        mr = {m: (1 - rec[m][c][0] / rec[m][c][1]) if rec[m][c][1] else float('nan') for m in MODS}
        print(f"{c:12s} {n:4d}   {mr['cam']:6.2f} {mr['lidar']:6.2f} {mr['radar']:6.2f}")

    print("\n=== CAM recall on CAMERA-COVERABLE GT (objects that project into the image; extra stratified view --")
    print("    the headline tables above still count every miss. Gap between columns = pure coverage cost) ===")
    print(f"{'condition':12s} {'coverable':>10s}   {'recall(coverable)':>17s} {'recall(all GT)':>15s}")
    cam_cov = {}
    for c in CONDS:
        h = n = na = ha = 0
        for fr in te:
            if fr["cls"] != c: continue
            det = detected_by(fr, "cam"); cv = cam_coverable(fr)
            h += int((det & cv).sum()); n += int(cv.sum()); ha += int(det.sum()); na += len(cv)
        cam_cov[c] = dict(coverable=[h, n], all=[ha, na])
        rc = h / n if n else float("nan"); ra = ha / na if na else float("nan")
        print(f"{c:12s} {n:5d}/{na:3d}   {rc:17.2f} {ra:15.2f}")

    print("\n=== RECOVERABILITY  (of objects a modality misses, frac another modality catches) ===")
    for m in MODS:
        r = recov[m]; print(f"  {m:6s}: {r[0]}/{r[1]} = {r[0]/r[1]:.2f}" if r[1] else f"  {m}: no misses")

    print("\n=== RECALL by GT provenance [circularity view] ===")
    print(f"{'modality':8s}  {'all':>6s} {'cam-raised':>11s} {'lidar/radar':>12s}")
    for m in MODS:
        allh = sum(recby[m][p][0] for p in recby[m]); allt = sum(recby[m][p][1] for p in recby[m])
        def rp(p): return recby[m][p][0] / recby[m][p][1] if recby[m][p][1] else float('nan')
        print(f"{m:8s}  {allh/allt:6.2f} {rp('cam'):11.2f} {rp('lidarradar'):12.2f}")

    print("\n=== DETECTION (union GT, test, per-class centre radius) ===")
    print("mAP = AP averaged over radius x{0.25,0.5,1,2}; ATE = mean TP centre error (m); ASE = 1 - aligned IoU")
    print(f"{'policy':16s} {'prec':>5s} {'rec':>5s} {'F1':>5s} {'AP':>5s} {'mAP':>5s} {'ATE':>5s} {'ASE':>5s}")
    # The baseline ladder, all on the SAME three frozen detectors: per-modality -> equal fusion (no learning) ->
    # learned fusion (trained head). RL fusion (the rl_modality policy on these detectors) is the next rung.
    policies = [("cam-only", single_dets("cam")), ("lidar-only", single_dets("lidar")),
                ("radar-only", single_dets("radar")), ("equal-fusion", equal_dets),
                ("learned-fusion", learned_dets)]
    res = {}
    for nm, fn in policies:
        d = detect_metrics(te, fn); res[nm] = d
        print(f"{nm:16s} {d['precision']:5.2f} {d['recall']:5.2f} {d['f1']:5.2f} {d['AP']:5.2f} "
              f"{d['mAP']:5.2f} {d['ate']:5.2f} {d['ase']:5.2f}")
    print("\n=== MISS RATE by condition — FUSION policies (union GT with no emitted detection within radius) ===")
    def covered(fr, get_dets):
        b, s = get_dets(fr); G, R = gt_info(fr)
        if len(G) == 0: return np.zeros(0, bool)
        if len(b) == 0: return np.zeros(len(G), bool)
        return BX.center_dist_matrix(G, b).min(1) <= R
    fus_miss = {}
    print(f"{'condition':12s} {'n':>4s}   {'equal':>6s} {'learned':>8s}")
    for c in CONDS:
        row = {}
        for nm, fn in [("equal", equal_dets), ("learned", learned_dets)]:
            h = n = 0
            for fr in te:
                if fr["cls"] != c: continue
                cv = covered(fr, fn); h += int(cv.sum()); n += len(cv)
            row[nm] = (h, n)
        fus_miss[c] = row
        n = row["equal"][1]
        print(f"{c:12s} {n:4d}   {1 - row['equal'][0]/max(1,n):6.2f} {1 - row['learned'][0]/max(1,n):8.2f}")

    # ---------------- BEV visualisation: same render panel as the labelling tools (viz_labels) ----------------
    # raw lidar point scatter (grey, s=2), forward UP, +lateral LEFT (x inverted), equal aspect.
    VIZN = int(sys.argv[1]) if len(sys.argv) > 1 else 12          # python eval_l2.py [viz_n]; 0 = no viz
    if VIZN:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import data_l2
        VIZDIR = f"{HERE}/eval_l2_viz"; os.makedirs(VIZDIR, exist_ok=True)
        COLS = {"cam": "#B05A1E", "lidar": "#2E6E45", "radar": "#3A5CA8"}
        sel = te[::max(1, len(te) // VIZN)]
        want = {f["fid"] for f in sel}
        raw_pts = {f["fid"]: f["lidar_pts"] for f in data_l2.load_frames() if f["fid"] in want}
        def rect(ax, b, **kw):                                   # BEV box [f0,l0,f1,l1] -> (x=lat, y=fwd)
            ax.add_patch(Rectangle((b[1], b[0]), b[3] - b[1], b[2] - b[0], fill=False, **kw))
        for fr in sel:
            fig, ax = plt.subplots(figsize=(8, 9), facecolor="white")
            pts = raw_pts.get(fr["fid"])
            if pts is not None and len(pts):
                m = (pts[:, 0] > -2) & (pts[:, 0] < 34) & (np.abs(pts[:, 1]) < 18) & (pts[:, 2] > 0.05)
                ax.scatter(pts[m, 1], pts[m, 0], s=2, c="0.6", linewidths=0)
            for mname, col in COLS.items():                      # thin: each modality's raw proposals
                for b, sc in fr["props"][mname]:
                    rect(ax, b, ec=col, lw=1.0, alpha=0.85)
            fb, fs = learned_dets(fr)                            # thick teal: fused output (+score)
            for b, sc in zip(fb, fs):
                rect(ax, b, ec="#0E6E74", lw=2.6)
                ax.text(b[1], b[0] - 0.35, f"{sc:.2f}", color="#0E6E74", fontsize=8, ha="left")
            G, R = gt_info(fr)                                   # red dashed: union GT (+ class)
            for u, g in zip(fr["union_gt"], G):
                rect(ax, g, ec="#C0392B", lw=1.8, ls="--")
                ax.text(g[3], g[0] - 0.35, u.get("label", "?"), color="#C0392B", fontsize=8, ha="right")
            ax.set_xlim(17, -17); ax.set_ylim(-1, 33); ax.set_aspect("equal")
            ax.set_xlabel("lateral (m)  [+left]"); ax.set_ylabel("forward (m)")
            ax.set_title(f"{fr['fid']}   GT red-dashed | cam amber - lidar green - radar blue | FUSED teal (+score)",
                         fontsize=9.5)
            plt.tight_layout(); plt.savefig(f"{VIZDIR}/{fr['fid'].replace(':', '_')}.png", dpi=120); plt.close(fig)
        print(f"BEV visualisations: {VIZDIR}/ ({len(sel)} test frames; run `python eval_l2.py 0` to skip)")

    import json
    json.dump(dict(missrate={m: {c: (rec[m][c][0], rec[m][c][1]) for c in CONDS} for m in MODS},
                   fusion_missrate=fus_miss, cam_coverable=cam_cov,
                   recoverability={m: recov[m] for m in MODS}, detection=res),
              open(f"{HERE}/results_l2.json", "w"), indent=1)
    print(f"\nsaved {HERE}/results_l2.json")

# ===================== CLI =====================
if _CMD == "bev": bev_main()
elif _CMD == "cam2d": cam2d_main()
elif _CMD == "l2": l2_main()
