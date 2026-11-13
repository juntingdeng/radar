"""RL FUSION -- the final rung of the ladder: a contextual policy that picks WHICH of the three frozen detectors
to run per frame, from cheap probes, rewarded by detection quality minus compute cost.

  action   sensor subset: C, L, R, CL, CR, LR, CLR                      (same ACT as rl_modality/bandit50)
  context  6 cheap probes, no detector runs: camera brightness/contrast/near-laplacian, lidar near-point count,
           radar near-field BEV excess + strong-cell count               (same probes as rl_modality)
  fused    the chosen subset's cached proposals -> cross-modal groups -> the TRAINED 17-d fusion head (logit>0)
  reward   frame-F1 (per-class centre radius, one-to-one)  -  LAM * cost(subset)
           COST = [cam 0.3, lidar 0.4, radar 1.0], LAM = 0.15            (radar beamforming is the heavy step)

All detector outputs are cached in proposals_l2.pkl, so every action's reward is computable offline per frame
(complete feedback): the policy (probes -> 7 logits) trains by direct expected-reward maximisation on TRAIN
scenes, and is evaluated argmax on TEST scenes against every fixed subset and the per-frame oracle.
Run: python rl_fusion.py      -> prints tables, writes results_rl.json
"""
import os, json, pickle, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import boxes as BX, associate as AS
from fusion_head import group_feat, train_fusion, score_groups
HERE = os.path.dirname(os.path.abspath(__file__)); torch.manual_seed(0); np.random.seed(0)

ACT = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
ANM = ["C", "L", "R", "CL", "CR", "LR", "CLR"]
MODS = ["cam", "lidar", "radar"]
COST = np.array([0.3, 0.4, 1.0]); LAM = 0.15
RADIUS = {"car": 2.0, "truck": 2.0, "bus": 2.0, "person": 1.0, "bicycle": 1.0, "motorcycle": 1.0}  # = eval_l2

def gt_info(fr):
    if not fr["union_gt"]: return np.zeros((0, 4)), np.zeros((0,))
    G = np.array([u["box"] for u in fr["union_gt"]])
    R = np.array([RADIUS.get(u.get("label", "car"), 2.0) for u in fr["union_gt"]])
    return G, R

# ---------------- cheap probes (rl_modality conventions, computed from the raw frame, detectors NOT run) ----------
def probes(f):
    import cv2
    g = cv2.cvtColor(f["cam"].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    th = cv2.resize(g, (16, 12)).astype(float)
    lid = f["lidar_pts"]
    near = int(((lid[:, 0] > 0) & (lid[:, 0] < 10) & (np.abs(lid[:, 1]) < 10)
                & (lid[:, 2] > 0.3) & (lid[:, 2] < 4)).sum())
    b = np.nan_to_num(np.asarray(f["radar_bev"], float)); fa = np.asarray(f["fa"], float); la = np.asarray(f["la"], float)
    fi = np.where((fa >= 2) & (fa <= 18))[0]; li = np.where(np.abs(la) < 12)[0]
    sub = b[fi[0]:fi[-1] + 1, li[0]:li[-1] + 1] if (len(fi) and len(li)) else b
    peak = float(np.percentile(sub, 95) - np.median(b)); npk = int((sub > np.percentile(b, 90)).sum())
    return [float(th.mean()), float(th.std()),
            float(cv2.Laplacian(g[320:, :].astype(np.float32), cv2.CV_32F).var()), near, peak, npk]

# ---------------- fused detections of a subset + frame reward ----------------
def masked_ev(fr, mask):
    """Evidence rasters restricted to the SELECTED sensors -- a subset action must not consume raw data (lidar
    scan, radar beamforming) from sensors it didn't pay for. Zero raster -> evidence features read 0."""
    ev = fr.get("ev")
    if ev is None: return None
    z = np.zeros_like(np.asarray(ev["lid"]))
    return dict(lid=ev["lid"] if mask[1] else z, rad=ev["rad"] if mask[2] else z.astype(np.float16))

def subset_groups(fr, mask):
    props = {m: (fr["props"][m] if on else []) for m, on in zip(MODS, mask)}
    return AS.associate(props)

def fused_dets(fr, mask, head):
    gs = subset_groups(fr, mask)
    if not gs: return np.zeros((0, 4)), np.zeros((0,))
    s, b = score_groups(head, gs, masked_ev(fr, mask))
    keep = s > 0
    return b[keep], s[keep]

def frame_f1(dets_b, dets_s, G, R):
    """one-to-one score-ordered matching at per-class radius -> frame F1 (1.0 for empty-vs-empty)."""
    if len(G) == 0: return 1.0 if len(dets_b) == 0 else 0.0
    order = np.argsort(dets_s)[::-1]; matched = set(); tp = 0
    for i in order:
        dist = BX.center_dist_matrix(dets_b[i:i + 1], G)[0]
        ok = [j for j in np.argsort(dist) if dist[j] <= R[j] and j not in matched]
        if ok: matched.add(ok[0]); tp += 1
    fp = len(dets_b) - tp; fn = len(G) - tp
    return 2 * tp / max(1, 2 * tp + fp + fn)

class Policy(nn.Module):
    def __init__(self, d=6, a=len(ACT)):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, a))
    def forward(self, x): return self.net(x)

if __name__ == "__main__":
    import data_l2
    P = pickle.load(open(f"{HERE}/proposals_l2.pkl", "rb")); frames = P["frames"]
    trp = data_l2.load_split()                          # canonical split (split.json)
    raw = {f"{f['cls']}:{f['pid']}": f for f in data_l2.load_frames()}
    X = np.array([probes(raw[fr["fid"]]) for fr in frames], np.float32)
    tr = np.array([fr["pid"] in trp for fr in frames]); te = ~tr
    print(f"{len(frames)} frames (train {tr.sum()} / test {te.sum()}), probes {X.shape}")

    # PER-SUBSET fusion heads: each action's groups are judged by a head trained on THAT subset's groups (with
    # evidence masked to the subset). A single all-3 head is a corroboration gate that kills 1-sensor actions.
    heads = {}
    trf = [f for f in frames if f["pid"] in trp]
    for a, mask in enumerate(ACT):
        Xg, yg = [], []
        for fr in trf:
            G, R = gt_info(fr); ev = masked_ev(fr, mask)
            for g in subset_groups(fr, mask):
                gf, x = group_feat(g, ev); Xg.append(x)
                d = BX.center_dist_matrix(gf["box"][None], G)[0] if len(G) else np.zeros(0)
                yg.append(int(bool((d <= R).any())))
        heads[a] = train_fusion(np.array(Xg), np.array(yg), seed=a)
        print(f"  head[{ANM[a]:3s}]: {len(yg)} train groups ({int(np.sum(yg))} pos)", flush=True)

    # complete-feedback reward table: R[i,a] = frame_f1 - LAM*cost ; keep per-frame counts for pooled metrics
    nF, nA = len(frames), len(ACT)
    F1 = np.zeros((nF, nA)); CO = np.array([ (COST * np.array(a)).sum() for a in ACT ])
    DETS = {}
    for i, fr in enumerate(frames):
        G, R = gt_info(fr)
        for a, mask in enumerate(ACT):
            b, s = fused_dets(fr, mask, heads[a])
            F1[i, a] = frame_f1(b, s, G, R); DETS[(i, a)] = (b, s)
        if i % 40 == 0: print(f"  rewards {i+1}/{nF}", flush=True)
    RW = F1 - LAM * CO[None, :]

    # policy: probes -> action logits, trained by expected reward on TRAIN frames
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = torch.tensor((X - mu) / sd); Rt = torch.tensor(RW, dtype=torch.float32)
    pol = Policy(); opt = torch.optim.Adam(pol.parameters(), 1e-2)
    tri = torch.tensor(np.where(tr)[0])
    for ep in range(600):
        pi = F.softmax(pol(Xn[tri]), -1)
        loss = -(pi * Rt[tri]).sum(1).mean() - 0.01 * (-(pi * pi.clamp(1e-8).log()).sum(1).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    pol.eval()
    with torch.no_grad():
        act_te = pol(Xn[torch.tensor(np.where(te)[0])]).argmax(1).numpy()

    # ---------------- report on TEST: fixed subsets vs policy vs oracle ----------------
    tei = np.where(te)[0]
    def aligned_iou(d, g):
        wd, ld = d[2] - d[0], d[3] - d[1]; wg, lg = g[2] - g[0], g[3] - g[1]
        inter = min(wd, wg) * min(ld, lg); u = wd * ld + wg * lg - inter
        return inter / u if u > 0 else 0.0
    def pooled(action_of):           # pooled P/R/F1 + VOC AP + TP quality (ATE/ASE) + mean cost/reward/frameF1
        tp = fp = fn = 0; cost = rw = f1m = 0.0; pl = []; ngt = 0; ates = []; ases = []
        for k, i in enumerate(tei):
            a = action_of(k, i); G, R = gt_info(frames[i]); b, s = DETS[(i, a)]
            order = np.argsort(s)[::-1]; matched = set(); t = 0
            for j in order:
                dist = BX.center_dist_matrix(b[j:j + 1], G)[0] if len(G) else np.zeros(0)
                ok = [m for m in np.argsort(dist) if len(G) and dist[m] <= R[m] and m not in matched]
                hit = 0
                if ok:
                    m = ok[0]; matched.add(m); t += 1; hit = 1
                    ates.append(float(dist[m])); ases.append(1.0 - aligned_iou(b[j], G[m]))
                pl.append((float(s[j]), hit))
            tp += t; fp += len(b) - t; fn += len(G) - t; ngt += len(G)
            cost += CO[a]; rw += RW[i, a]; f1m += F1[i, a]
        n = len(tei); prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        t_ = 0; ap = 0.0
        for j, (s_, h) in enumerate(sorted(pl, key=lambda x: -x[0]), 1):
            if h: t_ += 1; ap += t_ / j
        return dict(prec=round(prec, 2), rec=round(rec, 2),
                    f1=round(2 * prec * rec / max(1e-9, prec + rec), 2), AP=round(ap / max(1, ngt), 2),
                    ate=round(float(np.mean(ates)), 2) if ates else float("nan"),
                    ase=round(float(np.mean(ases)), 2) if ases else float("nan"),
                    frameF1=round(f1m / n, 3), cost=round(cost / n, 2), reward=round(rw / n, 3))
    rows = {}
    for a, nm in enumerate(ANM): rows[nm] = pooled(lambda k, i, a=a: a)
    rows["RL-policy"] = pooled(lambda k, i: int(act_te[k]))
    rows["oracle"] = pooled(lambda k, i: int(RW[i].argmax()))
    print(f"\n=== RL FUSION on the 3 frozen detectors (test; reward = frameF1 - {LAM} x cost; C/L/R cost {COST.tolist()}) ===")
    print(f"{'policy':10s} {'prec':>5s} {'rec':>5s} {'F1':>5s} {'AP':>5s} {'ATE':>5s} {'ASE':>5s} {'frameF1':>8s} {'cost':>5s} {'reward':>7s}")
    for nm, d in rows.items():
        print(f"{nm:10s} {d['prec']:5.2f} {d['rec']:5.2f} {d['f1']:5.2f} {d['AP']:5.2f} {d['ate']:5.2f} {d['ase']:5.2f} "
              f"{d['frameF1']:8.3f} {d['cost']:5.2f} {d['reward']:7.3f}")

    print("\n=== policy action histogram by condition (test) ===")
    hist = collections.defaultdict(collections.Counter)
    for k, i in enumerate(tei): hist[frames[i]["cls"]][ANM[act_te[k]]] += 1
    for c, h in hist.items(): print(f"  {c:12s} {dict(h)}")

    # miss rate by condition: union GT with no emitted detection within its radius, under each policy's actions
    print("\n=== MISS RATE by condition (test) ===")
    conds = ["night_clean", "night_glare", "day_clean", "day_smoke"]
    def missrow(action_of):
        out = {}
        for c in conds:
            h = n = 0
            for k, i in enumerate(tei):
                if frames[i]["cls"] != c: continue
                G, R = gt_info(frames[i]); b, _ = DETS[(i, action_of(k, i))]
                if len(G):
                    cv = (BX.center_dist_matrix(G, b).min(1) <= R) if len(b) else np.zeros(len(G), bool)
                    h += int(cv.sum()); n += len(G)
            out[c] = (h, n)
        return out
    mm = {"CLR": missrow(lambda k, i: len(ACT) - 1), "RL-policy": missrow(lambda k, i: int(act_te[k])),
          "oracle": missrow(lambda k, i: int(RW[i].argmax()))}
    print(f"{'condition':12s}" + "".join(f" {nm:>10s}" for nm in mm))
    for c in conds:
        print(f"{c:12s}" + "".join(f" {1 - v[c][0]/max(1, v[c][1]):10.2f}" for v in mm.values()))

    json.dump(dict(rows=rows, actions={c: dict(h) for c, h in hist.items()},
                   missrate={nm: v for nm, v in mm.items()},
                   lam=LAM, cost=COST.tolist()), open(f"{HERE}/results_rl.json", "w"), indent=1)
    print(f"\nsaved {HERE}/results_rl.json")
