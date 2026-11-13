"""RL test on the 50-scene / 16-layer env, scored by the LEARNED per-modality detectors (env50_learned_rows.json
from models.py) -- NOT the circular YOLO-labeler/occupancy-threshold scoring (removed). Contextual bandit: cheap
probes -> sensor subset; reward = object RECALL of the chosen subset (per-object union of learned-detector hits) -
lambda*cost. Split by BASE SCENE (pid) so conditions of one scene can't leak train->test."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); rng=np.random.default_rng(0)
rows=json.load(open(os.environ.get("ENV_ROWS",f"{HERE}/env50_learned_rows.json"))); N=len(rows)
ACT=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]; ANM=["C","L","R","CL","CR","LR","CLR"]
COST=np.array([0.3,0.4,1.0]); LAM=0.15
REC=np.zeros((N,7)); Rt=np.zeros((N,7))
for i,r in enumerate(rows):
    H=np.array([r["hits_cam"],r["hits_lidar"],r["hits_radar"]],bool)        # (3, n_obj)
    for a,m in enumerate(ACT):
        mm=np.array(m,bool); rec=float(H[mm].any(0).mean()); REC[i,a]=rec; Rt[i,a]=rec-LAM*COST[mm].sum()
pid=np.array([r["pid"] for r in rows])
Xr=np.stack([[r["probe_bright"] for r in rows],[r["probe_contrast"] for r in rows],
             np.log10(np.maximum([r["probe_near_lap"] for r in rows],1.)),
             np.log10(np.maximum([r["probe_lidar_near"] for r in rows],10.)),
             [r.get("probe_radar_bev",0.) for r in rows],
             np.log10(np.maximum([r.get("probe_radar_pts",0.) for r in rows],1.))],1)
BINS=6; feats=[]
for j in range(Xr.shape[1]):
    e=np.quantile(Xr[:,j],np.linspace(0,1,BINS+1)[1:-1]); feats.append(np.eye(BINS)[np.digitize(Xr[:,j],e)])
X=np.concatenate(feats+[np.ones((N,1))],1); D=X.shape[1]
up=np.unique(pid); rng.shuffle(up); trp=set(up[:int(0.7*len(up))]); tr=np.array([p in trp for p in pid]); te=~tr
print(f"{N} samples, {len(up)} scenes -> train {tr.sum()} ({len(trp)} scenes) / test {te.sum()} ({len(up)-len(trp)} scenes)")
W=np.zeros((7,D)); curve=[]
for ep in range(800):
    eps=max(0.03,0.5*(1-ep/500)); lr=0.08/(1+ep/80); order=np.where(tr)[0]; rng.shuffle(order)
    for i in order:
        a=rng.integers(7) if rng.random()<eps else int(np.argmax(W@X[i])); W[a]+=lr*(Rt[i,a]-W[a]@X[i])*X[i]
    if ep%20==0: curve.append((ep,float(Rt[te,np.argmax(X[te]@W.T,1)].mean())))
ate=np.argmax(X[te]@W.T,1)
def st(ix):
    cost=np.array([COST[np.array(ACT[a],bool)].sum() for a in ix]) if hasattr(ix,"__len__") else COST[np.array(ACT[ix],bool)].sum()
    return dict(reward=float(Rt[te,ix].mean()),recall=float(REC[te,ix].mean()),cost=float(np.mean(cost)))
res={"learned":st(ate),"oracle":st(np.argmax(Rt[te],1))}
for a,nm in enumerate(ANM):
    if nm in ("C","L","R","CLR"): res[f"always_{nm}"]=st(a)
res["random"]=dict(reward=float(Rt[te][np.arange(te.sum()),rng.integers(0,7,te.sum())].mean()))
print(json.dumps({k:{kk:round(vv,3) for kk,vv in v.items()} for k,v in res.items()},indent=1))
json.dump(dict(results=res,n=N,scenes=len(up),curve=curve),open(f"{HERE}/results50.json","w"),indent=1)
# figure: per-condition recall + policy per condition + reward bars
import collections
fig=plt.figure(figsize=(19,5.5))
CONDS=["clean","light","dense","lidaronly"]
cc=np.array([r["cls"] for r in rows])          # condition class stored by models.py (clean/light/dense/lidaronly)
ax1=fig.add_subplot(1,3,1)
for j,(nm,col) in enumerate([("cam","tab:blue"),("lidar","tab:orange"),("radar","tab:green")]):
    v=[np.mean([rows[i][["rec_cam","rec_lidar","rec_radar"][j]] for i in range(N) if cc[i]==c]) for c in CONDS]
    ax1.bar(np.arange(len(CONDS))+(j-1)*0.25,v,0.25,color=col,label=nm)
ax1.set_xticks(range(len(CONDS))); ax1.set_xticklabels(CONDS,fontsize=8); ax1.legend(fontsize=8); ax1.set_ylabel("object recall")
ax1.set_title("per-modality recall by condition (physics, no FLUX)",fontsize=10)
ax2=fig.add_subplot(1,3,2); aa=np.argmax(X@W.T,1); M=np.zeros((len(CONDS),7))
for ci,c in enumerate(CONDS):
    sel=cc==c
    for a in range(7): M[ci,a]=np.mean(aa[sel]==a) if sel.any() else 0
ax2.imshow(M,cmap="Blues",aspect="auto",vmin=0,vmax=1); ax2.set_xticks(range(7)); ax2.set_xticklabels(ANM); ax2.set_yticks(range(len(CONDS))); ax2.set_yticklabels(CONDS,fontsize=8)
for ci in range(len(CONDS)):
    for a in range(7):
        if M[ci,a]>0.05: ax2.text(a,ci,f"{M[ci,a]:.2f}",ha="center",va="center",fontsize=7,color="white" if M[ci,a]>0.5 else "black")
ax2.set_title("learned policy: action share per condition",fontsize=10)
ax3=fig.add_subplot(1,3,3); ks=[k for k in res if k!="random"]; vals=[res[k]["reward"] for k in ks]
ax3.bar(ks,vals,color=["tab:red" if k=="learned" else ("tab:green" if k=="oracle" else "0.6") for k in ks])
ax3.axhline(res["random"]["reward"],color="k",ls="--",lw=0.8,label="random"); ax3.legend(fontsize=8); ax3.tick_params(axis="x",labelsize=8)
ax3.set_title(f"held-out reward (recall - {LAM}x cost)",fontsize=10)
fig.suptitle(f"RL modality selection on the 50-scene / 16-layer env (physics composites). learned {res['learned']['reward']:.3f} vs always-C {res['always_C']['reward']:.3f} / always-R {res['always_R']['reward']:.3f}, oracle {res['oracle']['reward']:.3f}.",fontsize=10,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(f"{HERE}/rl_modality_env50.png",dpi=85); print("saved rl_modality_env50.png")
