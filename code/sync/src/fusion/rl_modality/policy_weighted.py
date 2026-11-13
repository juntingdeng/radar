"""Weighted (continuous) modality policy -- the action is a WEIGHT per modality, not a binary on/off subset.

Reliability-gated late fusion: policy pi_theta(probes) -> w in [0,1]^3 (sigmoid per modality). Per object, the
learned per-modality detector LOGITS (from models.py) are fused  fused = sum_m w_m * logit_m ; detected iff
fused>0. Cost is paid in proportion to use: cost = sum_m w_m * cost_m. Reward = recall(fused) - lambda*cost.
The binary subset policy is the special case w in {0,1}^3; weights let the agent partially trust a modality.

Trained by direct gradient ascent on a SOFT reward (soft_recall = mean sigmoid(fused), differentiable in w), then
reported with HARD recall (fused>0) so it is comparable to the discrete bandit. Baselines: fixed weights
(all-on / camera / lidar / radar / equal) and a per-frame oracle (grid over w). Split by base scene."""
import os, json, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); rng=np.random.default_rng(0); torch.manual_seed(0)
rows=json.load(open(os.environ.get("ENV_ROWS",f"{HERE}/env50_learned_rows.json")))
COST=torch.tensor([0.3,0.4,1.0]); LAM=0.15
# per-sample object logits L (n_obj,3) and probe features
L=[torch.tensor(np.stack([r["logit_cam"],r["logit_lidar"],r["logit_radar"]],1)).float() for r in rows]
pid=np.array([r["pid"] for r in rows])
Xr=np.stack([[r["probe_bright"] for r in rows],[r["probe_contrast"] for r in rows],
             np.log10(np.maximum([r["probe_near_lap"] for r in rows],1.)),
             np.log10(np.maximum([r["probe_lidar_near"] for r in rows],10.)),
             [r.get("probe_radar_bev",0.) for r in rows],
             np.log10(np.maximum([r.get("probe_radar_pts",0.) for r in rows],1.))],1)
Xr=(Xr-Xr.mean(0))/(Xr.std(0)+1e-9); X=torch.tensor(np.concatenate([Xr,np.ones((len(rows),1))],1)).float()
up=np.unique(pid); rng.shuffle(up); trp=set(up[:int(0.7*len(up))]); tr=np.array([p in trp for p in pid]); te=~tr
tri,tei=np.where(tr)[0],np.where(te)[0]
print(f"{len(rows)} samples, {len(up)} scenes -> train {tr.sum()} / test {te.sum()}")

# pad per-sample object logits to (N, M, 3) + mask; clip to a sane range
M=max(x.shape[0] for x in L); N=len(L)
Lpad=torch.zeros(N,M,3); Lmask=torch.zeros(N,M)
for i,x in enumerate(L): Lpad[i,:x.shape[0]]=x.clamp(-12,12); Lmask[i,:x.shape[0]]=1
def batch_reward(W,idx,hard=False):                  # W:(len(idx),3) or (3,); -> mean reward,recall,cost over idx
    Wb=W if W.dim()==2 else W[None].expand(len(idx),3)
    fused=torch.einsum('nmc,nc->nm',Lpad[idx],Wb)     # (len,M)
    rec=(((fused>0).float() if hard else torch.sigmoid(fused))*Lmask[idx]).sum(1)/Lmask[idx].sum(1)
    cost=(Wb*COST).sum(1)
    r=rec-LAM*cost; return r.mean(),rec.mean(),cost.mean()
class Policy(nn.Module):
    # SIMPLEX weights (softmax): a reliability DISTRIBUTION summing to 1 -> cost = sum(w*c) in [0.3,1.0] cannot be
    # gamed to ~0 (independent [0,1]^3 + sign-fusion let the policy shrink weights toward 0 and still detect on a
    # confident modality -> fake cost<0.3). Softmax = the standard attention/reliability-gate interpretation.
    def __init__(s,d): super().__init__(); s.net=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,3))
    def forward(s,x): return torch.softmax(s.net(x),-1)
pol=Policy(X.shape[1]); opt=torch.optim.Adam(pol.parameters(),3e-3)
for ep in range(600):
    opt.zero_grad(); W=pol(X[tri]); loss=-batch_reward(W,tri)[0]; loss.backward()
    torch.nn.utils.clip_grad_norm_(pol.parameters(),5.0); opt.step()
pol.eval()
with torch.no_grad():
    Wte=pol(X[tei])
    def agg(W):
        r,rec,c=batch_reward(W,tei,hard=True); return dict(reward=r.item(),recall=rec.item(),cost=c.item())
    res={"learned(weighted)":agg(Wte)}
    for nm,wv in [("camera",[1,0,0]),("lidar",[0,1,0]),("radar",[0,0,1]),("equal",[1/3,1/3,1/3])]:   # simplex points
        res[f"fixed_{nm}"]=agg(torch.tensor(wv).float())
    g=torch.tensor([[a,b,c] for a in (0,.25,.5,.75,1) for b in (0,.25,.5,.75,1) for c in (0,.25,.5,.75,1)]).float()
    g=g[g.sum(1)>0]; grid=g/g.sum(1,keepdim=True)                       # SIMPLEX grid (best reliability distribution/frame)
    Ro=0.
    for i in tei:
        fused=Lpad[i][:int(Lmask[i].sum())]@grid.T
        rec=(fused>0).float().mean(0); r=rec-LAM*(grid*COST).sum(1); Ro+=r.max().item()
    res["oracle(simplex)"]={"reward":Ro/len(tei)}
# discrete bandit result for reference
disc=json.load(open(f"{HERE}/results50.json"))["results"]["learned"]["reward"]
print(json.dumps({k:{kk:round(vv,3) for kk,vv in v.items()} for k,v in res.items()},indent=1))
print(f"(discrete binary-subset bandit, for reference: {disc:.3f})")
json.dump(dict(results=res,discrete_binary=disc),open(f"{HERE}/results50_weighted.json","w"),indent=1)
# figure: mean learned weight per condition
with torch.no_grad(): Wall=pol(X).numpy()
cc=np.array([r["cls"] for r in rows]); CONDS=["clean","light","dense","lidaronly"]
fig,ax=plt.subplots(1,2,figsize=(14,5))
Mw=np.array([[Wall[cc==c,j].mean() for j in range(3)] for c in CONDS])
for j,(nm,col) in enumerate([("cam","tab:blue"),("lidar","tab:orange"),("radar","tab:green")]):
    ax[0].bar(np.arange(len(CONDS))+(j-1)*0.25,Mw[:,j],0.25,color=col,label=nm)
ax[0].set_xticks(range(len(CONDS))); ax[0].set_xticklabels(CONDS,fontsize=8); ax[0].legend(fontsize=8); ax[0].set_ylabel("mean learned weight")
ax[0].set_title("learned continuous weights per condition (reliability gate)",fontsize=10)
ks=[k for k in res]; ax[1].bar(ks,[res[k]["reward"] for k in ks],color=["tab:red" if "learned" in k else ("tab:green" if "oracle" in k else "0.6") for k in ks])
ax[1].axhline(disc,ls="--",c="k",lw=0.8,label=f"discrete binary ({disc:.2f})"); ax[1].legend(fontsize=8); ax[1].tick_params(axis="x",labelsize=7,rotation=20); ax[1].set_title("held-out reward (hard recall - 0.15 cost)",fontsize=10)
fig.suptitle("Weighted (continuous) modality policy: action = weight per modality (generalizes binary on/off). Reliability-gated logit fusion, weighted cost.",fontsize=10,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig(f"{HERE}/rl_modality_weighted.png",dpi=85); print("saved rl_modality_weighted.png")
