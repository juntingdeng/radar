"""TWO-VIEW late fusion: run the full single-view system at two sequential ego poses, bring the earlier view's
detections into the current frame via lidar-ICP ego-motion, and late-fuse the two detection sets.

  view pairs   consecutive benchmark frames of the same condition (sorted by pid). Night pairs are short-baseline
               (~0.2-0.3 s); day pairs are long-baseline (~14 s -- the cart has moved past the parked cars, so the
               two views see different aspects/occlusions of the same static objects).
  ego-motion   point-to-plane ICP between the two cached lidar scans (coarse 5 m -> fine 1 m correspondence),
               SE(3) -> the BEV SE(2) applied to view-1 boxes (corners transformed, axis-aligned hull kept).
  late fusion  UNION with metric dedupe: transformed view-1 detections join view-2's; within 1.5 m they merge and
               corroborate (s = 1-(1-s1)(1-s2)); view-1-only survivors are imported as rescues (occlusion change).
               No tracking -- objects that MOVED between the views import as false positives; that failure mode is
               reported, not hidden.

Eval: view-2 frames on TEST scenes only, against view-2's own GT, same per-class-radius matching as eval_l2.
Run: python two_view_fusion.py        -> prints single-view vs two-view tables, writes results_2view.json
"""
import os, json, pickle, collections, numpy as np, torch
import boxes as BX, associate as AS
from fusion_head import group_feat, train_fusion, score_groups
HERE = os.path.dirname(os.path.abspath(__file__)); torch.manual_seed(0); np.random.seed(0)
RADIUS = {"car": 2., "truck": 2., "bus": 2., "person": 1., "bicycle": 1., "motorcycle": 1.}
CONDS = ["night_clean", "night_glare", "day_clean", "day_smoke"]
LINK2V = 1.5                      # metric dedupe/corroboration radius across views (same as association)

def gt_info(fr):
    if not fr["union_gt"]: return np.zeros((0, 4)), np.zeros((0,))
    return (np.array([u["box"] for u in fr["union_gt"]]),
            np.array([RADIUS.get(u.get("label", "car"), 2.) for u in fr["union_gt"]]))

def icp_pair(p1, p2):
    """scan1 -> scan2 SE(3) via point-to-plane ICP, coarse->fine. Returns (T 4x4, fitness)."""
    import open3d as o3d
    def pc(P):
        c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[:, :3].astype(np.float64)))
        c = c.voxel_down_sample(0.3)
        c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        return c
    a, b = pc(p1), pc(p2); T = np.eye(4)
    for corr in (5.0, 1.0):
        r = o3d.pipelines.registration.registration_icp(
            a, b, corr, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = r.transformation
    return np.array(T), float(r.fitness)

def warp_boxes(B, T):
    """BEV boxes (n,4) through the SE(2) part of T: transform the 4 corners, keep the axis-aligned hull."""
    if len(B) == 0: return B
    R2, t2 = T[:2, :2], T[:2, 3]
    out = []
    for f0, l0, f1, l1 in B:
        c = np.array([[f0, l0], [f0, l1], [f1, l0], [f1, l1]]) @ R2.T + t2
        out.append([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()])
    return np.array(out)

def match_counts(b, s, G, R):
    """score-ordered one-to-one matching -> (tp, fp, fn, [ate], [ase])."""
    order = np.argsort(s)[::-1]; matched = set(); tp = 0; ates = []; ases = []
    for i in order:
        dist = BX.center_dist_matrix(b[i:i + 1], G)[0] if len(G) else np.zeros(0)
        ok = [j for j in np.argsort(dist) if len(G) and dist[j] <= R[j] and j not in matched]
        if ok:
            j = ok[0]; matched.add(j); tp += 1; ates.append(float(dist[j]))
            d, g = b[i], G[j]
            wd, ld = d[2]-d[0], d[3]-d[1]; wg, lg = g[2]-g[0], g[3]-g[1]
            inter = min(wd, wg)*min(ld, lg); u = wd*ld + wg*lg - inter
            ases.append(1 - inter/u if u > 0 else 1.0)
    return tp, len(b) - tp, len(G) - tp, ates, ases

if __name__ == "__main__":
    import data_l2
    P = pickle.load(open(f"{HERE}/proposals_l2.pkl", "rb")); frames = P["frames"]
    trp = data_l2.load_split()                          # canonical split (split.json)
    raw = {f"{f['cls']}:{f['pid']}": f for f in data_l2.load_frames()}

    # single-view fused detections for every frame (head trained on train groups, same recipe as eval_l2)
    Xg, yg = [], []
    for fr in [f for f in frames if f["pid"] in trp]:
        G, R = gt_info(fr)
        for g in AS.associate(fr["props"]):
            gf, x = group_feat(g, fr.get("ev")); Xg.append(x)
            d = BX.center_dist_matrix(gf["box"][None], G)[0] if len(G) else np.zeros(0)
            yg.append(int(bool((d <= R).any())))
    head = train_fusion(np.array(Xg), np.array(yg))
    DET = {}
    for fr in frames:
        gs = AS.associate(fr["props"])
        if not gs: DET[fr["fid"]] = (np.zeros((0, 4)), np.zeros(0)); continue
        s, b = score_groups(head, gs, fr.get("ev")); k = s > 0
        DET[fr["fid"]] = (b[k], 1 / (1 + np.exp(-s[k])))
    print(f"head trained ({len(yg)} groups); single-view detections cached for {len(frames)} frames")

    # sequential pairs (prev -> cur), evaluated on TEST cur frames
    bycls = collections.defaultdict(list)
    for fr in frames: bycls[fr["cls"]].append(fr)
    pairs = []
    for c, fl in bycls.items():
        fl = sorted(fl, key=lambda f: f["pid"])
        pairs += [(a, b) for a, b in zip(fl, fl[1:]) if b["pid"] not in trp]
    print(f"{len(pairs)} test view-pairs; running ICP ego-motion...")

    agg = {m: dict(tp=0, fp=0, fn=0, ates=[], ases=[]) for m in ("single", "two-view")}
    bycond = {m: collections.defaultdict(lambda: [0, 0]) for m in ("single", "two-view")}   # cond -> [hit, n]
    fits = []; disp = []; nimp = 0
    for k, (v1, v2) in enumerate(pairs):
        T, fit = icp_pair(raw[v1["fid"]]["lidar_pts"], raw[v2["fid"]]["lidar_pts"])
        fits.append(fit); disp.append(float(np.hypot(T[0, 3], T[1, 3])))
        b1, s1 = DET[v1["fid"]]; b2, s2 = DET[v2["fid"]]
        b1w = warp_boxes(b1, T)
        # union with metric dedupe: corroborated v2 dets get boosted; unmatched v1 dets import as rescues
        fb = list(map(np.array, b2)); fs = list(s2); used1 = set()
        for i in range(len(b2)):
            if len(b1w):
                d = BX.center_dist_matrix(b2[i:i + 1], b1w)[0]
                j = int(d.argmin())
                if d[j] <= LINK2V and j not in used1:
                    used1.add(j); fs[i] = 1 - (1 - fs[i]) * (1 - s1[j])
        for j in range(len(b1w)):
            if j not in used1: fb.append(b1w[j]); fs.append(float(s1[j] * 0.9)); nimp += 1
        fb = np.array(fb) if fb else np.zeros((0, 4)); fs = np.array(fs)
        G, R = gt_info(v2)
        for m, (bb, ss) in (("single", (b2, s2)), ("two-view", (fb, fs))):
            tp, fp, fn, at, ae = match_counts(bb, ss, G, R)
            A = agg[m]; A["tp"] += tp; A["fp"] += fp; A["fn"] += fn; A["ates"] += at; A["ases"] += ae
            cov = (BX.center_dist_matrix(G, bb).min(1) <= R) if len(G) and len(bb) else np.zeros(len(G), bool)
            bycond[m][v2["cls"]][0] += int(cov.sum()); bycond[m][v2["cls"]][1] += len(G)
        if k % 20 == 0: print(f"  pair {k+1}/{len(pairs)}  icp fitness {fit:.2f}  |t| {disp[-1]:.1f} m", flush=True)

    print(f"\nICP: fitness median {np.median(fits):.2f} (min {min(fits):.2f}); ego displacement median {np.median(disp):.1f} m, max {max(disp):.1f} m")
    print(f"view-1 detections imported as rescues: {nimp}")
    print(f"\n=== SINGLE-VIEW vs TWO-VIEW late fusion ({len(pairs)} test frames with a prior view) ===")
    print(f"{'policy':10s} {'prec':>5s} {'rec':>5s} {'F1':>5s} {'ATE':>5s} {'ASE':>5s}")
    res = {}
    for m in ("single", "two-view"):
        A = agg[m]; p = A["tp"] / max(1, A["tp"] + A["fp"]); r = A["tp"] / max(1, A["tp"] + A["fn"])
        res[m] = dict(prec=round(p, 2), rec=round(r, 2), f1=round(2*p*r/max(1e-9, p+r), 2),
                      ate=round(float(np.mean(A["ates"])), 2), ase=round(float(np.mean(A["ases"])), 2))
        d = res[m]
        print(f"{m:10s} {d['prec']:5.2f} {d['rec']:5.2f} {d['f1']:5.2f} {d['ate']:5.2f} {d['ase']:5.2f}")
    print("\n=== MISS RATE by condition ===")
    print(f"{'condition':12s} {'n':>4s} {'single':>7s} {'two-view':>9s}")
    mr = {}
    for c in CONDS:
        n = bycond["single"][c][1]
        if n == 0: continue
        mr[c] = {m: 1 - bycond[m][c][0] / n for m in ("single", "two-view")}
        print(f"{c:12s} {n:4d} {mr[c]['single']:7.2f} {mr[c]['two-view']:9.2f}")
    json.dump(dict(rows=res, missrate=mr, icp_fitness_median=float(np.median(fits)),
                   ego_disp_median_m=float(np.median(disp)), imported=nimp, pairs=len(pairs)),
              open(f"{HERE}/results_2view.json", "w"), indent=1)
    print(f"\nsaved {HERE}/results_2view.json")
