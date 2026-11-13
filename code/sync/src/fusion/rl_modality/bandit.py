"""Contextual-bandit modality selector on the measured environment (env_table.npz).

State  = cheap probes only (thumbnail brightness/contrast, near-band sharpness, lidar near-count)
Action = subset of {camera, lidar, radar} (7 non-empty subsets)
Reward = scene coverage of the chosen subset  -  lambda * compute cost
         coverage = max over chosen modalities of its MEASURED visibility this frame
         (a modality with no valid measurement that frame contributes 0 -- real sensor dropout)
Costs reflect this stack's compute: radar BEV beamforming >> lidar scan decode > camera probe.

Train: epsilon-greedy linear Q (numpy SGD), contiguous-block train/test split (no temporal leakage).
Baselines: fixed single modalities, always-all, random, and the per-frame ORACLE subset.
Outputs: results.json + rl_modality_policy.png next to this script.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
rng=np.random.default_rng(0)

E=np.load(f"{HERE}/env_table.npz",allow_pickle=True)
N=len(E["el"])
vis=np.stack([np.nan_to_num(E["vis_cam"],nan=0.0),
              np.nan_to_num(E["vis_lidar"],nan=0.0),
              np.nan_to_num(E["vis_radar"],nan=0.0)],1)          # (N,3) measured; NaN->0 = dropout
ACTIONS=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
ANAMES=["C","L","R","CL","CR","LR","CLR"]
COST=np.array([0.3,0.4,1.0]); LAM=0.15
def reward_all(v):  # (N,3) -> (N,7)
    out=np.zeros((len(v),len(ACTIONS)))
    for a,m in enumerate(ACTIONS):
        mm=np.array(m,bool)
        out[:,a]=v[:,mm].max(1)-LAM*COST[mm].sum()
    return out
Rtab=reward_all(vis)

# state: probes only, standardized; lidar-near NaN (dropout) -> median + flag
pb=E["probe_bright"]; pc=E["probe_contrast"]; pl=np.log10(np.maximum(E["probe_near_lap"],1.0))
pn=E["probe_lidar_near"].copy(); nanmask=~np.isfinite(pn); pn[nanmask]=np.nanmedian(pn); pn=np.log10(np.maximum(pn,10))
Xraw=np.stack([pb,pc,pl,pn],1)
# binned one-hot features (6 quantile bins per probe) -- the visibility structure is thresholdy, a plain
# linear Q underfits it (first version scored below always-camera)
BINS=6
feats=[]
edges=[]
for j in range(Xraw.shape[1]):
    e=np.quantile(Xraw[:,j],np.linspace(0,1,BINS+1)[1:-1]); edges.append(e)
    idx=np.digitize(Xraw[:,j],e)
    oh=np.zeros((N,BINS)); oh[np.arange(N),idx]=1
    feats.append(oh)
feats.append(nanmask[:,None].astype(float))
X=np.concatenate(feats+[np.ones((N,1))],1)
D=X.shape[1]

# contiguous-block split (block 25) -> ~70/30
blocks=np.arange(N)//25
ub=np.unique(blocks); rng.shuffle(ub)
tr_b=set(ub[:int(0.7*len(ub))]); tr=np.array([b in tr_b for b in blocks]); te=~tr
print(f"train {tr.sum()} / test {te.sum()} frames")

W=np.zeros((len(ACTIONS),D))
EPOCHS=600; eps0=0.5
curve=[]
for ep in range(EPOCHS):
    eps=max(0.03,eps0*(1-ep/400))
    lr=0.08/(1+ep/60)
    order=np.where(tr)[0]; rng.shuffle(order)
    for i in order:
        a=rng.integers(len(ACTIONS)) if rng.random()<eps else int(np.argmax(W@X[i]))
        err=Rtab[i,a]-W[a]@X[i]
        W[a]+=lr*err*X[i]
    if ep%15==0 or ep==EPOCHS-1:
        a_te=np.argmax(X[te]@W.T,1)
        curve.append((ep,float(Rtab[te,a_te].mean())))
a_te=np.argmax(X[te]@W.T,1)

res={}
res["learned"]=dict(reward=float(Rtab[te,a_te].mean()),
    coverage=float(np.array([vis[i][np.array(ACTIONS[a],bool)].max() for i,a in zip(np.where(te)[0],a_te)]).mean()),
    cost=float(np.array([COST[np.array(ACTIONS[a],bool)].sum() for a in a_te]).mean()))
orc=np.argmax(Rtab[te],1)
res["oracle"]=dict(reward=float(Rtab[te,orc].mean()),
    coverage=float(np.array([vis[i][np.array(ACTIONS[a],bool)].max() for i,a in zip(np.where(te)[0],orc)]).mean()),
    cost=float(np.array([COST[np.array(ACTIONS[a],bool)].sum() for a in orc]).mean()))
for a,nm in enumerate(ANAMES):
    if nm in ("C","L","R","CLR"):
        res[f"always_{nm}"]=dict(reward=float(Rtab[te,a].mean()),
            coverage=float(vis[te][:,np.array(ACTIONS[a],bool)].max(1).mean()),
            cost=float(COST[np.array(ACTIONS[a],bool)].sum()))
res["random"]=dict(reward=float(Rtab[te][np.arange(te.sum()),rng.integers(0,7,te.sum())].mean()))
print(json.dumps(res,indent=1))
json.dump(dict(results=res,lambda_=LAM,costs=COST.tolist(),actions=ANAMES,
               curve=curve,n_train=int(tr.sum()),n_test=int(te.sum())),
          open(f"{HERE}/results.json","w"),indent=1)

# ---- figure: visibility curves + learned policy over time + bars ----
fig=plt.figure(figsize=(19,10)); el=E["el"]; vid=E["video"]
ax1=fig.add_subplot(2,2,1)
off=np.where(vid=="fog1",-1550+120,0)   # draw fog window (el 1618-1647) left of smoke on one axis via offset
xx=np.where(vid=="smoke1",el,el+off)
for j,(nm,col) in enumerate([("camera","tab:blue"),("lidar","tab:orange"),("radar","tab:green")]):
    ax1.plot(xx,vis[:,j],".",ms=3,color=col,label=f"vis {nm}")
ax1.axvline(1766,color="k",lw=0.5); ax1.text(190,1.08,"fog1 window",fontsize=8); ax1.text(1790,1.08,"smoke1 window",fontsize=8)
ax1.set_ylim(0,1.15); ax1.legend(fontsize=8); ax1.set_title("measured per-modality visibility (reward ingredients)",fontsize=10)
ax2=fig.add_subplot(2,2,2)
a_all=np.argmax(X@W.T,1)
ax2.scatter(xx,a_all,c=["tab:red" if te[i] else "0.7" for i in range(N)],s=8)
ax2.set_yticks(range(7)); ax2.set_yticklabels(ANAMES); ax2.set_title("learned policy per frame (red = held-out test)",fontsize=10)
ax3=fig.add_subplot(2,2,3)
ks=[k for k in res if k!="random"]; vals=[res[k]["reward"] for k in ks]
ax3.bar(ks,vals,color=["tab:red" if k=="learned" else ("tab:green" if k=="oracle" else "0.6") for k in ks])
ax3.axhline(res["random"]["reward"],color="k",ls="--",lw=0.8,label="random")
ax3.set_ylabel("mean test reward"); ax3.legend(fontsize=8); ax3.tick_params(axis="x",labelsize=8)
ax3.set_title(f"test reward (coverage - {LAM} x cost)",fontsize=10)
ax4=fig.add_subplot(2,2,4)
ax4.plot(*zip(*curve),"-o",ms=3); ax4.set_xlabel("epoch"); ax4.set_ylabel("held-out reward"); ax4.set_title("learning curve",fontsize=10)
fig.suptitle("RL (contextual bandit) modality selection on 495 REAL measured frames: cheap probes -> choose sensor subset. "
             "Clean: camera (cheapest) suffices; smoke: camera+lidar visibility collapses -> radar worth its cost; fog: camera stays adequate.",
             fontsize=11,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(f"{HERE}/rl_modality_policy.png",dpi=85)
print(f"saved {HERE}/rl_modality_policy.png")
