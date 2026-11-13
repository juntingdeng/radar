"""F1-based reward + precision/recall/AP for the modality selector (reviewer point #2).

The env now carries background NEGATIVE candidates (neg_logit_* from build_env_multi, K_NEG per frame) so false
positives are counted:  TP = GT object with fused>0 ; FP = background cand with fused>0 ; FN = GT object with fused<=0.
Detection of a candidate = (weighted logit sum  w . logit) > tau.  Reward = F1 - lambda*cost  (was recall - lambda*cost;
recall alone never penalised FPs, so a policy had no reason to reject an unreliable modality -- reviewer pt #2).

Reports, per policy, on a SCENE-split held-out set: micro precision / recall / F1 (at tau=0) and AP (sklearn,
threshold-independent, sweeps tau). Policies: fixed weights (camera / lidar / radar / equal), a LEARNED weighted
policy trained on a differentiable soft-F1, and a per-frame ORACLE (best simplex weights). Run:
  ENV_ROWS=env_multi_rows.json python eval_f1.py
"""
import os, json, numpy as np, torch, torch.nn as nn
from sklearn.metrics import average_precision_score
HERE=os.path.dirname(os.path.abspath(__file__)); rng=np.random.default_rng(0); torch.manual_seed(0)
rows=json.load(open(os.environ.get("ENV_ROWS", f"{HERE}/env_multi_rows.json")))
if "neg_logit_cam" not in rows[0]:
    raise SystemExit("env has no neg_logit_* -- re-run build_env_multi.py (needs the K_NEG background candidates).")
COST=np.array([0.3,0.4,1.0]); LAM=0.15; EPS=1e-9

# per-frame positive (GT) and negative (background) logit matrices, (n,3)
Pos=[np.stack([r["logit_cam"],r["logit_lidar"],r["logit_radar"]],1).astype(float) for r in rows]
Neg=[np.stack([r["neg_logit_cam"],r["neg_logit_lidar"],r["neg_logit_radar"]],1).astype(float) for r in rows]
pid=np.array([r["pid"] for r in rows]); cls=np.array([r["cls"] for r in rows]); N=len(rows)
# scene split (a scene's conditions stay together)
up=np.unique(pid); rng.shuffle(up); trp=set(up[:int(0.7*len(up))])
tr=np.array([p in trp for p in pid]); te=~tr; tri,tei=np.where(tr)[0],np.where(te)[0]
print(f"{N} frames, {len(up)} scenes -> train {tr.sum()} / test {te.sum()}  (K_neg/frame={Neg[0].shape[0]})")

# probe features for the learned policy
def probe_X():
    Xr=np.stack([[r["probe_bright"] for r in rows],[r["probe_contrast"] for r in rows],
                 np.log10(np.maximum([r["probe_near_lap"] for r in rows],1.)),
                 np.log10(np.maximum([r["probe_lidar_near"] for r in rows],10.)),
                 [r.get("probe_radar_bev",0.) for r in rows],
                 np.log10(np.maximum([r.get("probe_radar_pts",0.) for r in rows],1.))],1)
    Xr=(Xr-Xr.mean(0))/(Xr.std(0)+EPS)
    return torch.tensor(np.concatenate([Xr,np.ones((N,1))],1)).float()
X=probe_X()

# ---------- hard metrics (tau=0) for a per-frame weight function ----------
def metrics(wfun, idxs):
    TP=FP=FN=0; sc=[]; lb=[]; costs=[]
    for i in idxs:
        w=wfun(i); ps=Pos[i]@w; ns=Neg[i]@w
        TP+=int((ps>0).sum()); FP+=int((ns>0).sum()); FN+=int((ps<=0).sum())
        sc+=list(ps)+list(ns); lb+=[1]*len(ps)+[0]*len(ns); costs.append(float((w*COST).sum()))
    P=TP/(TP+FP) if TP+FP else 1.0; R=TP/(TP+FN) if TP+FN else 0.0
    F1=2*P*R/(P+R) if P+R>0 else 0.0
    ap=average_precision_score(lb,sc) if len(set(lb))>1 else float("nan")
    rec_only=TP/(TP+FN) if TP+FN else 0.0  # old reward's detection term
    c=float(np.mean(costs))
    return dict(precision=round(P,3),recall=round(R,3),f1=round(F1,3),AP=round(float(ap),3),
                cost=round(c,3),reward_F1=round(F1-LAM*c,3),reward_recall_only=round(rec_only-LAM*c,3))

fixed={"camera":np.array([1.,0,0]),"lidar":np.array([0,1.,0]),"radar":np.array([0,0,1.]),
       "equal":np.array([1,1,1.])/3}
res={nm:metrics(lambda i,w=w:w, tei) for nm,w in fixed.items()}

# ---------- per-frame ORACLE over a simplex grid (best F1 - lambda*cost) ----------
g=np.array([[a,b,c] for a in(0,.25,.5,.75,1) for b in(0,.25,.5,.75,1) for c in(0,.25,.5,.75,1)],float)
g=g[g.sum(1)>0]; grid=g/g.sum(1,keepdims=True)
def oracle_w(i):
    best=None; bestr=-1e9
    for w in grid:
        ps=Pos[i]@w; ns=Neg[i]@w
        tp=(ps>0).sum(); fp=(ns>0).sum(); fn=(ps<=0).sum()
        P=tp/(tp+fp) if tp+fp else 1.0; R=tp/(tp+fn) if tp+fn else 0.0
        f1=2*P*R/(P+R) if P+R>0 else 0.0; r=f1-LAM*(w*COST).sum()
        if r>bestr: bestr=r; best=w
    return best
oracle_cache={i:oracle_w(i) for i in tei}
res["oracle(F1)"]=metrics(lambda i:oracle_cache[i], tei)

# ---------- LEARNED weighted policy trained on differentiable soft-F1 ----------
Mp=max(len(p) for p in Pos); Mn=max(len(n) for n in Neg)
Lp=torch.zeros(N,Mp,3); Mpmask=torch.zeros(N,Mp); Ln=torch.zeros(N,Mn,3); Mnmask=torch.zeros(N,Mn)
for i in range(N):
    p=torch.tensor(Pos[i]).float().clamp(-12,12); Lp[i,:len(p)]=p; Mpmask[i,:len(p)]=1
    n=torch.tensor(Neg[i]).float().clamp(-12,12); Ln[i,:len(n)]=n; Mnmask[i,:len(n)]=1
COSTt=torch.tensor(COST).float()
def soft_reward(W,idx):                       # W:(len,3) simplex; differentiable soft-F1 - lam*cost
    fp=torch.sigmoid(torch.einsum('nmc,nc->nm',Lp[idx],W))      # P(detect) on positives
    fn=torch.sigmoid(torch.einsum('nmc,nc->nm',Ln[idx],W))      # P(detect) on negatives (-> FP)
    stp=(fp*Mpmask[idx]).sum(1); sfp=(fn*Mnmask[idx]).sum(1); sfn=((1-fp)*Mpmask[idx]).sum(1)
    P=stp/(stp+sfp+EPS); R=stp/(stp+sfn+EPS); F1=2*P*R/(P+R+EPS)
    cost=(W*COSTt).sum(1); return (F1-LAM*cost).mean()
class Pol(nn.Module):
    def __init__(s,d): super().__init__(); s.net=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,3))
    def forward(s,x): return torch.softmax(s.net(x),-1)
pol=Pol(X.shape[1]); opt=torch.optim.Adam(pol.parameters(),3e-3)
for ep in range(700):
    opt.zero_grad(); loss=-soft_reward(pol(X[tri]),tri); loss.backward()
    nn.utils.clip_grad_norm_(pol.parameters(),5.0); opt.step()
pol.eval()
with torch.no_grad(): Wall=pol(X).numpy()
res["learned(F1,weighted)"]=metrics(lambda i:Wall[i], tei)

# ---------- report ----------
order=["camera","lidar","radar","equal","learned(F1,weighted)","oracle(F1)"]
print("\n=== F1 reward: held-out precision / recall / F1 / AP (micro, scene-split) ===")
print(f"{'policy':22s} {'prec':>5s} {'rec':>5s} {'F1':>5s} {'AP':>5s} {'cost':>5s} | {'rwd(F1)':>8s} {'rwd(recall-only)':>16s}")
for k in order:
    v=res[k]; print(f"{k:22s} {v['precision']:5.2f} {v['recall']:5.2f} {v['f1']:5.2f} {v['AP']:5.2f} {v['cost']:5.2f} | {v['reward_F1']:8.3f} {v['reward_recall_only']:16.3f}")
# headroom
base=max(res['camera']['reward_F1'],res['equal']['reward_F1'])
print(f"\nF1-reward headroom: oracle {res['oracle(F1)']['reward_F1']:.3f} - best-baseline {base:.3f} = {res['oracle(F1)']['reward_F1']-base:+.3f}")
print(f"learned {res['learned(F1,weighted)']['reward_F1']:.3f} vs best-baseline {base:.3f} = {res['learned(F1,weighted)']['reward_F1']-base:+.3f}")
json.dump(res,open(f"{HERE}/results_f1.json","w"),indent=1)
print(f"saved {HERE}/results_f1.json")
