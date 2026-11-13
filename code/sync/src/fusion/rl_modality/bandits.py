"""Contextual bandits (merged; two envs). Subcommands: v495 | scenes50"""
import sys

# ======== src/fusion/rl_modality/bandit.py ========
# Contextual-bandit modality selector on the measured environment (env_table.npz).
# 
# State  = cheap probes only (thumbnail brightness/contrast, near-band sharpness, lidar near-count)
# Action = subset of {camera, lidar, radar} (7 non-empty subsets)
# Reward = scene coverage of the chosen subset  -  lambda * compute cost
#          coverage = max over chosen modalities of its MEASURED visibility this frame
#          (a modality with no valid measurement that frame contributes 0 -- real sensor dropout)
# Costs reflect this stack's compute: radar BEV beamforming >> lidar scan decode > camera probe.
# 
# Train: epsilon-greedy linear Q (numpy SGD), contiguous-block train/test split (no temporal leakage).
# Baselines: fixed single modalities, always-all, random, and the per-frame ORACLE subset.
# Outputs: results.json + rl_modality_policy.png next to this script.
def v495_main():
    import os, json
    import numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
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
    COST=np.array([0.3,0.4,1.0]); LAM=0.15
    def reward_all(v):  # (N,3) -> (N,7)
        out=np.zeros((len(v),len(ACTIONS)))
        for a,m in enumerate(ACTIONS):
            mm=np.array(m,bool)
            out[:,a]=v[:,mm].max(1)-LAM*COST[mm].sum()
        return out
    Rtab=reward_all(vis)
    pb=E["probe_bright"]; pc=E["probe_contrast"]; pl=np.log10(np.maximum(E["probe_near_lap"],1.0))
    pb=E["probe_bright"]; pc=E["probe_contrast"]; pl=np.log10(np.maximum(E["probe_near_lap"],1.0))
    pb=E["probe_bright"]; pc=E["probe_contrast"]; pl=np.log10(np.maximum(E["probe_near_lap"],1.0))
    pn=E["probe_lidar_near"].copy(); nanmask=~np.isfinite(pn); pn[nanmask]=np.nanmedian(pn); pn=np.log10(np.maximum(pn,10))
    pn=E["probe_lidar_near"].copy(); nanmask=~np.isfinite(pn); pn[nanmask]=np.nanmedian(pn); pn=np.log10(np.maximum(pn,10))
    pn=E["probe_lidar_near"].copy(); nanmask=~np.isfinite(pn); pn[nanmask]=np.nanmedian(pn); pn=np.log10(np.maximum(pn,10))
    pn=E["probe_lidar_near"].copy(); nanmask=~np.isfinite(pn); pn[nanmask]=np.nanmedian(pn); pn=np.log10(np.maximum(pn,10))
    Xraw=np.stack([pb,pc,pl,pn],1)
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
    blocks=np.arange(N)//25
    ub=np.unique(blocks); rng.shuffle(ub)
    ub=np.unique(blocks); rng.shuffle(ub)
    tr_b=set(ub[:int(0.7*len(ub))]); tr=np.array([b in tr_b for b in blocks]); te=~tr
    tr_b=set(ub[:int(0.7*len(ub))]); tr=np.array([b in tr_b for b in blocks]); te=~tr
    tr_b=set(ub[:int(0.7*len(ub))]); tr=np.array([b in tr_b for b in blocks]); te=~tr
    print(f"train {tr.sum()} / test {te.sum()} frames")
    W=np.zeros((len(ACTIONS),D))
    EPOCHS=600; eps0=0.5
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
    fig=plt.figure(figsize=(19,10)); el=E["el"]; vid=E["video"]
    fig=plt.figure(figsize=(19,10)); el=E["el"]; vid=E["video"]
    fig=plt.figure(figsize=(19,10)); el=E["el"]; vid=E["video"]
    ax1=fig.add_subplot(2,2,1)
    off=np.where(vid=="fog1",-1550+120,0)   # draw fog window (el 1618-1647) left of smoke on one axis via offset
    xx=np.where(vid=="smoke1",el,el+off)
    for j,(nm,col) in enumerate([("camera","tab:blue"),("lidar","tab:orange"),("radar","tab:green")]):
        ax1.plot(xx,vis[:,j],".",ms=3,color=col,label=f"vis {nm}")
    ax1.axvline(1766,color="k",lw=0.5); ax1.text(190,1.08,"fog1 window",fontsize=8); ax1.text(1790,1.08,"smoke1 window",fontsize=8)
    ax1.axvline(1766,color="k",lw=0.5); ax1.text(190,1.08,"fog1 window",fontsize=8); ax1.text(1790,1.08,"smoke1 window",fontsize=8)
    ax1.axvline(1766,color="k",lw=0.5); ax1.text(190,1.08,"fog1 window",fontsize=8); ax1.text(1790,1.08,"smoke1 window",fontsize=8)
    ax1.set_ylim(0,1.15); ax1.legend(fontsize=8); ax1.set_title("measured per-modality visibility (reward ingredients)",fontsize=10)
    ax1.set_ylim(0,1.15); ax1.legend(fontsize=8); ax1.set_title("measured per-modality visibility (reward ingredients)",fontsize=10)
    ax1.set_ylim(0,1.15); ax1.legend(fontsize=8); ax1.set_title("measured per-modality visibility (reward ingredients)",fontsize=10)
    ax2=fig.add_subplot(2,2,2)
    a_all=np.argmax(X@W.T,1)
    ax2.scatter(xx,a_all,c=["tab:red" if te[i] else "0.7" for i in range(N)],s=8)
    ax2.set_yticks(range(7)); ax2.set_yticklabels(ANAMES); ax2.set_title("learned policy per frame (red = held-out test)",fontsize=10)
    ax2.set_yticks(range(7)); ax2.set_yticklabels(ANAMES); ax2.set_title("learned policy per frame (red = held-out test)",fontsize=10)
    ax2.set_yticks(range(7)); ax2.set_yticklabels(ANAMES); ax2.set_title("learned policy per frame (red = held-out test)",fontsize=10)
    ax3=fig.add_subplot(2,2,3)
    ks=[k for k in res if k!="random"]; vals=[res[k]["reward"] for k in ks]
    ks=[k for k in res if k!="random"]; vals=[res[k]["reward"] for k in ks]
    ax3.bar(ks,vals,color=["tab:red" if k=="learned" else ("tab:green" if k=="oracle" else "0.6") for k in ks])
    ax3.axhline(res["random"]["reward"],color="k",ls="--",lw=0.8,label="random")
    ax3.set_ylabel("mean test reward"); ax3.legend(fontsize=8); ax3.tick_params(axis="x",labelsize=8)
    ax3.set_ylabel("mean test reward"); ax3.legend(fontsize=8); ax3.tick_params(axis="x",labelsize=8)
    ax3.set_ylabel("mean test reward"); ax3.legend(fontsize=8); ax3.tick_params(axis="x",labelsize=8)
    ax3.set_title(f"test reward (coverage - {LAM} x cost)",fontsize=10)
    ax4=fig.add_subplot(2,2,4)
    ax4.plot(*zip(*curve),"-o",ms=3); ax4.set_xlabel("epoch"); ax4.set_ylabel("held-out reward"); ax4.set_title("learning curve",fontsize=10)
    ax4.plot(*zip(*curve),"-o",ms=3); ax4.set_xlabel("epoch"); ax4.set_ylabel("held-out reward"); ax4.set_title("learning curve",fontsize=10)
    ax4.plot(*zip(*curve),"-o",ms=3); ax4.set_xlabel("epoch"); ax4.set_ylabel("held-out reward"); ax4.set_title("learning curve",fontsize=10)
    ax4.plot(*zip(*curve),"-o",ms=3); ax4.set_xlabel("epoch"); ax4.set_ylabel("held-out reward"); ax4.set_title("learning curve",fontsize=10)
    fig.suptitle("RL (contextual bandit) modality selection on 495 REAL measured frames: cheap probes -> choose sensor subset. "
                 "Clean: camera (cheapest) suffices; smoke: camera+lidar visibility collapses -> radar worth its cost; fog: camera stays adequate.",
                 fontsize=11,fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(f"{HERE}/rl_modality_policy.png",dpi=85)
    plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(f"{HERE}/rl_modality_policy.png",dpi=85)
    print(f"saved {HERE}/rl_modality_policy.png")

# ======== src/fusion/rl_modality/bandit50.py ========
# RL test on the 50-scene / 16-layer env, scored by the LEARNED per-modality detectors (env50_learned_rows.json
# from models.py) -- NOT the circular YOLO-labeler/occupancy-threshold scoring (removed). Contextual bandit: cheap
# probes -> sensor subset; reward = object RECALL of the chosen subset (per-object union of learned-detector hits) -
# lambda*cost. Split by BASE SCENE (pid) so conditions of one scene can't leak train->test.
def scenes50_main():
    import os, json, numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    HERE=os.path.dirname(os.path.abspath(__file__)); rng=np.random.default_rng(0)
    HERE=os.path.dirname(os.path.abspath(__file__)); rng=np.random.default_rng(0)
    rows=json.load(open(os.environ.get("ENV_ROWS",f"{HERE}/env50_learned_rows.json"))); N=len(rows)
    rows=json.load(open(os.environ.get("ENV_ROWS",f"{HERE}/env50_learned_rows.json"))); N=len(rows)
    ACT=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]; ANM=["C","L","R","CL","CR","LR","CLR"]
    ACT=[(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]; ANM=["C","L","R","CL","CR","LR","CLR"]
    COST=np.array([0.3,0.4,1.0]); LAM=0.15
    COST=np.array([0.3,0.4,1.0]); LAM=0.15
    REC=np.zeros((N,7)); Rt=np.zeros((N,7))
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
    BINS=6; feats=[]
    for j in range(Xr.shape[1]):
        e=np.quantile(Xr[:,j],np.linspace(0,1,BINS+1)[1:-1]); feats.append(np.eye(BINS)[np.digitize(Xr[:,j],e)])
    X=np.concatenate(feats+[np.ones((N,1))],1); D=X.shape[1]
    X=np.concatenate(feats+[np.ones((N,1))],1); D=X.shape[1]
    up=np.unique(pid); rng.shuffle(up); trp=set(up[:int(0.7*len(up))]); tr=np.array([p in trp for p in pid]); te=~tr
    up=np.unique(pid); rng.shuffle(up); trp=set(up[:int(0.7*len(up))]); tr=np.array([p in trp for p in pid]); te=~tr
    up=np.unique(pid); rng.shuffle(up); trp=set(up[:int(0.7*len(up))]); tr=np.array([p in trp for p in pid]); te=~tr
    up=np.unique(pid); rng.shuffle(up); trp=set(up[:int(0.7*len(up))]); tr=np.array([p in trp for p in pid]); te=~tr
    up=np.unique(pid); rng.shuffle(up); trp=set(up[:int(0.7*len(up))]); tr=np.array([p in trp for p in pid]); te=~tr
    print(f"{N} samples, {len(up)} scenes -> train {tr.sum()} ({len(trp)} scenes) / test {te.sum()} ({len(up)-len(trp)} scenes)")
    W=np.zeros((7,D)); curve=[]
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
    import collections
    fig=plt.figure(figsize=(19,5.5))
    CONDS=["clean","light","dense","lidaronly"]
    cc=np.array([r["cls"] for r in rows])          # condition class stored by models.py (clean/light/dense/lidaronly)
    ax1=fig.add_subplot(1,3,1)
    for j,(nm,col) in enumerate([("cam","tab:blue"),("lidar","tab:orange"),("radar","tab:green")]):
        v=[np.mean([rows[i][["rec_cam","rec_lidar","rec_radar"][j]] for i in range(N) if cc[i]==c]) for c in CONDS]
        ax1.bar(np.arange(len(CONDS))+(j-1)*0.25,v,0.25,color=col,label=nm)
    ax1.set_xticks(range(len(CONDS))); ax1.set_xticklabels(CONDS,fontsize=8); ax1.legend(fontsize=8); ax1.set_ylabel("object recall")
    ax1.set_xticks(range(len(CONDS))); ax1.set_xticklabels(CONDS,fontsize=8); ax1.legend(fontsize=8); ax1.set_ylabel("object recall")
    ax1.set_xticks(range(len(CONDS))); ax1.set_xticklabels(CONDS,fontsize=8); ax1.legend(fontsize=8); ax1.set_ylabel("object recall")
    ax1.set_xticks(range(len(CONDS))); ax1.set_xticklabels(CONDS,fontsize=8); ax1.legend(fontsize=8); ax1.set_ylabel("object recall")
    ax1.set_title("per-modality recall by condition (physics, no FLUX)",fontsize=10)
    ax2=fig.add_subplot(1,3,2); aa=np.argmax(X@W.T,1); M=np.zeros((len(CONDS),7))
    ax2=fig.add_subplot(1,3,2); aa=np.argmax(X@W.T,1); M=np.zeros((len(CONDS),7))
    ax2=fig.add_subplot(1,3,2); aa=np.argmax(X@W.T,1); M=np.zeros((len(CONDS),7))
    for ci,c in enumerate(CONDS):
        sel=cc==c
        for a in range(7): M[ci,a]=np.mean(aa[sel]==a) if sel.any() else 0
    ax2.imshow(M,cmap="Blues",aspect="auto",vmin=0,vmax=1); ax2.set_xticks(range(7)); ax2.set_xticklabels(ANM); ax2.set_yticks(range(len(CONDS))); ax2.set_yticklabels(CONDS,fontsize=8)
    ax2.imshow(M,cmap="Blues",aspect="auto",vmin=0,vmax=1); ax2.set_xticks(range(7)); ax2.set_xticklabels(ANM); ax2.set_yticks(range(len(CONDS))); ax2.set_yticklabels(CONDS,fontsize=8)
    ax2.imshow(M,cmap="Blues",aspect="auto",vmin=0,vmax=1); ax2.set_xticks(range(7)); ax2.set_xticklabels(ANM); ax2.set_yticks(range(len(CONDS))); ax2.set_yticklabels(CONDS,fontsize=8)
    ax2.imshow(M,cmap="Blues",aspect="auto",vmin=0,vmax=1); ax2.set_xticks(range(7)); ax2.set_xticklabels(ANM); ax2.set_yticks(range(len(CONDS))); ax2.set_yticklabels(CONDS,fontsize=8)
    ax2.imshow(M,cmap="Blues",aspect="auto",vmin=0,vmax=1); ax2.set_xticks(range(7)); ax2.set_xticklabels(ANM); ax2.set_yticks(range(len(CONDS))); ax2.set_yticklabels(CONDS,fontsize=8)
    for ci in range(len(CONDS)):
        for a in range(7):
            if M[ci,a]>0.05: ax2.text(a,ci,f"{M[ci,a]:.2f}",ha="center",va="center",fontsize=7,color="white" if M[ci,a]>0.5 else "black")
    ax2.set_title("learned policy: action share per condition",fontsize=10)
    ax3=fig.add_subplot(1,3,3); ks=[k for k in res if k!="random"]; vals=[res[k]["reward"] for k in ks]
    ax3=fig.add_subplot(1,3,3); ks=[k for k in res if k!="random"]; vals=[res[k]["reward"] for k in ks]
    ax3=fig.add_subplot(1,3,3); ks=[k for k in res if k!="random"]; vals=[res[k]["reward"] for k in ks]
    ax3.bar(ks,vals,color=["tab:red" if k=="learned" else ("tab:green" if k=="oracle" else "0.6") for k in ks])
    ax3.axhline(res["random"]["reward"],color="k",ls="--",lw=0.8,label="random"); ax3.legend(fontsize=8); ax3.tick_params(axis="x",labelsize=8)
    ax3.axhline(res["random"]["reward"],color="k",ls="--",lw=0.8,label="random"); ax3.legend(fontsize=8); ax3.tick_params(axis="x",labelsize=8)
    ax3.axhline(res["random"]["reward"],color="k",ls="--",lw=0.8,label="random"); ax3.legend(fontsize=8); ax3.tick_params(axis="x",labelsize=8)
    ax3.set_title(f"held-out reward (recall - {LAM}x cost)",fontsize=10)
    fig.suptitle(f"RL modality selection on the 50-scene / 16-layer env (physics composites). learned {res['learned']['reward']:.3f} vs always-C {res['always_C']['reward']:.3f} / always-R {res['always_R']['reward']:.3f}, oracle {res['oracle']['reward']:.3f}.",fontsize=10,fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(f"{HERE}/rl_modality_env50.png",dpi=85); print("saved rl_modality_env50.png")
    plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(f"{HERE}/rl_modality_env50.png",dpi=85); print("saved rl_modality_env50.png")
    plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(f"{HERE}/rl_modality_env50.png",dpi=85); print("saved rl_modality_env50.png")

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'v495': v495_main, 'scenes50': scenes50_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
