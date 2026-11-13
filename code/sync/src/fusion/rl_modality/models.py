"""Learned per-modality object detectors for a FAIR cross-modal recall (replaces the hand thresholds + the
YOLO-is-also-the-labeler circularity).

Each detector is an object-vs-background classifier over a candidate region:
  CameraNet : conv stem -> ATTENTION head over the spatial tokens of the image crop.
  LidarNet  : POINTS branch (PointNet-style per-point MLP + masked max-pool) + BEV branch (CNN on a rasterized
              height patch), fused.
  RadarNet  : POINTS branch (top-k BEV peak cells as (dfwd,dlat,power)) + BEV branch (CNN on the power patch), fused.

FAIRNESS: trained ONLY on CLEAN frames (positives = GT boxes, negatives = random background boxes), so each head
learns its modality's object appearance independent of how the GT was labeled. Evaluated on DEGRADED frames: a head
fires iff the object signal survives in its modality -> recall reflects that modality's own information, not the
labeler's. Run: `python models.py` trains on the 50 clean scenes and prints fair per-modality recall by condition,
next to the old threshold recall.
"""
import os, sys, json, numpy as np
import torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)
HERE=os.path.dirname(os.path.abspath(__file__))
R10=os.path.abspath(os.path.join(HERE,"..","..","..","..","sync","res","2026.10.15"))
SCR=next((p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)),os.path.expanduser("~/.cache/ti_setup_frames/rl_env"))  # persistent local first; session scratchpad fallback

# ---------------- models ----------------
class AttnHead(nn.Module):
    """learned-query attention pooling over tokens -> logit."""
    def __init__(self,d):
        super().__init__(); self.q=nn.Parameter(torch.randn(d)*0.1); self.k=nn.Linear(d,d); self.v=nn.Linear(d,d)
        self.head=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,t):                               # t: (B,N,d)
        a=torch.softmax(self.k(t)@self.q,1).unsqueeze(-1)
        return self.head((a*self.v(t)).sum(1)).squeeze(-1)

class CameraNet(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
                                nn.Conv2d(32,d,3,2,1),nn.ReLU())          # 48->6x6
        self.attn=AttnHead(d)
    def forward(self,x):
        f=self.stem(x); t=f.flatten(2).transpose(1,2); return self.attn(t)

class LidarNet(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.pt=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,d),nn.ReLU())
        self.bev=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.AdaptiveMaxPool2d(1))
        self.head=nn.Sequential(nn.Linear(d+32,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,pts,mask,bev):                    # pts (B,N,3), mask (B,N) bool, bev (B,1,H,W)
        pf=self.pt(pts).masked_fill(~mask.unsqueeze(-1),-1e4).amax(1)
        pf=torch.where(mask.any(1,keepdim=True),pf,torch.zeros_like(pf))  # no points -> zero feature
        return self.head(torch.cat([pf,self.bev(bev).flatten(1)],1)).squeeze(-1)

class RadarNet(nn.Module):
    def __init__(self,d=48):
        super().__init__()
        self.pt=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,d),nn.ReLU())
        self.bev=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.AdaptiveMaxPool2d(1))
        self.head=nn.Sequential(nn.Linear(d+32,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,pts,mask,bev):
        pf=self.pt(pts).masked_fill(~mask.unsqueeze(-1),-1e4).amax(1)
        pf=torch.where(mask.any(1,keepdim=True),pf,torch.zeros_like(pf))
        return self.head(torch.cat([pf,self.bev(bev).flatten(1)],1)).squeeze(-1)

# ---------------- feature extraction ----------------
import cv2
NP=64   # max points per region
def cam_crop(img,bbox):
    x0,y0,x1,y1=[int(v) for v in bbox]; x0,y0=max(0,x0),max(0,y0); x1,y1=min(640,x1),min(480,y1)
    if x1<=x0 or y1<=y0: return np.zeros((48,48,3),np.float32)
    c=cv2.resize(img[y0:y1,x0:x1].astype(np.float32)/255.,(48,48)); return c
def lidar_feat(P,fwd,lat,m=2.0):
    f0,f1,l0,l1=fwd[0]-m,fwd[1]+m,lat[0]-m,lat[1]+m; cx,cy=(fwd[0]+fwd[1])/2,(lat[0]+lat[1])/2
    s=P[(P[:,0]>=f0)&(P[:,0]<=f1)&(P[:,1]>=l0)&(P[:,1]<=l1)&(P[:,2]>-1)&(P[:,2]<3)]
    pts=np.zeros((NP,3),np.float32); mask=np.zeros(NP,bool)
    if len(s):
        idx=np.random.permutation(len(s))[:NP]; q=s[idx]
        pts[:len(q)]=np.stack([(q[:,0]-cx)/4,(q[:,1]-cy)/4,q[:,2]/2],1); mask[:len(q)]=True
    # BEV height patch 24x24 over box+-m
    G=24; bev=np.zeros((G,G),np.float32)
    if len(s):
        gx=np.clip(((s[:,0]-f0)/(f1-f0)*G).astype(int),0,G-1); gy=np.clip(((s[:,1]-l0)/(l1-l0)*G).astype(int),0,G-1)
        np.maximum.at(bev,(gx,gy),(s[:,2]+1)/4)
    return pts,mask,bev
def radar_feat(bev,fwd,lat,la,fa,m=2.0):
    f0,f1,l0,l1=fwd[0]-m,fwd[1]+m,lat[0]-m,lat[1]+m
    fi0,fi1=np.searchsorted(fa,f0),np.searchsorted(fa,f1); li0,li1=np.searchsorted(la,l0),np.searchsorted(la,l1)
    fi0,fi1=max(0,fi0),min(len(fa),max(fi0+1,fi1)); li0,li1=max(0,li0),min(len(la),max(li0+1,li1))
    patch=np.nan_to_num(bev[fi0:fi1,li0:li1])
    bp=cv2.resize(patch.astype(np.float32),(24,24)) if patch.size else np.zeros((24,24),np.float32)
    mx=bp.max()+1e-6; bp=bp/mx
    # top-k peak cells as points (dfwd,dlat,power)
    pts=np.zeros((NP,3),np.float32); mask=np.zeros(NP,bool)
    if patch.size:
        flat=patch.flatten(); thr=np.percentile(flat,90); yy,xx=np.where(patch>=thr)
        k=min(NP,len(yy))
        if k:
            order=np.argsort(patch[yy,xx])[::-1][:k]; yy,xx=yy[order],xx[order]
            pts[:k]=np.stack([(yy/max(1,patch.shape[0])-0.5),(xx/max(1,patch.shape[1])-0.5),patch[yy,xx]/mx],1); mask[:k]=True
    return pts,mask,bp

def radar_probes(bev,la,fa):
    # CHEAP radar context for the policy (NO RadarNet run): near-field BEV strength + strong-return count.
    # bev[fi,li]: fi<->fa (forward), li<->la (lateral), both ascending. Near field = fwd 2-18m, |lat|<12m.
    # BEV is dB, floor-dominated -> use EXCESS of the strong near return over the global floor, not raw mean.
    b=np.nan_to_num(np.asarray(bev,float)); fa=np.asarray(fa,float); la=np.asarray(la,float)
    fi=np.where((fa>=2)&(fa<=18))[0]; li=np.where(np.abs(la)<12)[0]
    sub=b[fi[0]:fi[-1]+1,li[0]:li[-1]+1] if (len(fi) and len(li)) else b
    peak=float(np.percentile(sub,95)-np.median(b))   # probe_radar_bev : strong near return above floor (dB)
    npk=int((sub>np.percentile(b,90)).sum())         # probe_radar_pts : strong near-field cell count (point-like)
    return peak, npk

# ---------------- dataset from env (clean train / degraded eval) ----------------
def build_data():
    sys.path.insert(0,R10); import reconstruct_smoke as RS
    S=np.load(f"{SCR}/scenes50.npz",allow_pickle=True); meta=json.loads(str(S["meta"])); la,fa=S["la"],S["fa"]
    L=np.load(f"{SCR}/smoke_library_corrected.npz",allow_pickle=True); lib=list(L["lib"])
    clean_bev10=np.nan_to_num(L["clean_bev"].astype(float)); A=L["A"].astype(float); Blum=float(np.percentile(L["clean_cam"].astype(float).mean(2),50))
    LIDC=os.path.expanduser("~/.cache/ti_setup_frames/lidar_scan_cache")
    def rand_bg(boxes):   # a background box not overlapping any GT
        for _ in range(20):
            cf=np.random.uniform(7,24); cl=np.random.uniform(-12,12)
            fwd=(cf-2.2,cf+2.2); lat=(cl-1,cl+1)
            bb=[np.clip(320+cl*12,0,580),240,np.clip(320+cl*12+60,20,640),300]
            if all(not(abs((b['fwd'][0]+b['fwd'][1])/2-cf)<4 and abs((b['lat'][0]+b['lat'][1])/2-cl)<3) for b in boxes):
                return dict(bbox=bb,fwd=fwd,lat=lat)
        return dict(bbox=[0,240,60,300],fwd=(20,24),lat=(10,12))
    train={m:{"X":[],"y":[]} for m in ["cam","lid","rad"]}       # cam: crops; lid/rad: (pts,mask,bev)
    ev=[]                                                         # degraded eval samples
    for si,mt in enumerate(meta):
        cam0=S["cams"][si].astype(float)
        if cam0.max()==0: continue
        lid0=np.load(f"{LIDC}/l_{mt['lidar_idx']}.npy").astype(float); bev0=S["bevs"][si].astype(float)
        boxes=mt["boxes"]; ratio=float(np.percentile(cam0.mean(2),50))/Blum
        flip=float(np.mean([(b['bbox'][0]+b['bbox'][2])/2 for b in boxes]))<320
        # ---- CLEAN positives + negatives ----
        for b in boxes:
            train["cam"]["X"].append(cam_crop(cam0.astype(np.uint8),b["bbox"])); train["cam"]["y"].append(1)
            train["lid"]["X"].append(lidar_feat(lid0,b["fwd"],b["lat"])); train["lid"]["y"].append(1)
            train["rad"]["X"].append(radar_feat(bev0,b["fwd"],b["lat"],la,fa)); train["rad"]["y"].append(1)
        for _ in range(3*len(boxes)):
            nb=rand_bg(boxes)
            train["cam"]["X"].append(cam_crop(cam0.astype(np.uint8),nb["bbox"])); train["cam"]["y"].append(0)
            train["lid"]["X"].append(lidar_feat(lid0,nb["fwd"],nb["lat"])); train["lid"]["y"].append(0)
            train["rad"]["X"].append(radar_feat(bev0,nb["fwd"],nb["lat"],la,fa)); train["rad"]["y"].append(0)
        # ---- DEGRADED eval: clean + 4 layers (same rotation as env50) ----
        for cond,k in [("clean",-1)]+[(f"layer{(si*4+j)%16}",(si*4+j)%16) for j in range(4)]:
            if k<0: cam,lid,bev=cam0,lid0,bev0; dens=0; camok=True
            else:
                lay=lib[k]; camok=bool(lay.get("cam_ok",True)) and lay["alpha"] is not None; dens=int(lay["density"])
                a=(lay["alpha"].astype(float) if camok else None)
                if camok:
                    aa=a[:,::-1] if flip else a; Af=(lay["airlight"].astype(float)[:,::-1] if flip else lay["airlight"].astype(float))*ratio
                    cam=np.clip(cam0*(1-aa[...,None])+Af*aa[...,None],0,255)
                else: cam=cam0
                bs=lay["backscatter"].astype(float).copy();
                if flip: bs[:,1]*=-1
                lid=RS._lidar_apply(lid0,bs)
                w=lay["bev_w"].astype(float); d=np.nan_to_num(lay["bev_smoke"].astype(float))-clean_bev10
                if flip: w=w[:,::-1]; d=d[:,::-1]
                bev=np.nan_to_num(bev0)+w*np.clip(d,0,None)
            feats=[]
            for b in boxes:
                feats.append((cam_crop(cam.astype(np.uint8),b["bbox"]),lidar_feat(lid,b["fwd"],b["lat"]),radar_feat(bev,b["fwd"],b["lat"],la,fa)))
            cls="clean" if k<0 else ("lidaronly" if not camok else ("dense" if dens>8000 else "light"))
            g=cv2.cvtColor(cam.astype(np.uint8),cv2.COLOR_RGB2GRAY); th=cv2.resize(g,(16,12)).astype(float)
            near=int(((lid[:,0]>0)&(lid[:,0]<10)&(np.abs(lid[:,1])<10)&(lid[:,2]>0.3)&(lid[:,2]<4)).sum())
            rb,rp=radar_probes(bev,la,fa)
            probes=dict(probe_bright=float(th.mean()),probe_contrast=float(th.std()),
                        probe_near_lap=float(cv2.Laplacian(g[320:,:],cv2.CV_32F).var()),probe_lidar_near=near,
                        probe_radar_bev=rb,probe_radar_pts=rp)
            ev.append(dict(pid=mt["pid"],cond=cond,cls=cls,feats=feats,probes=probes))
        if si%10==0: print(f"  data scene {si+1}/{len(meta)}",flush=True)
    return train,ev

def to_cam(xs): return torch.tensor(np.stack(xs)).permute(0,3,1,2).float()
def to_pb(xs):
    pts=torch.tensor(np.stack([x[0] for x in xs])).float(); mask=torch.tensor(np.stack([x[1] for x in xs]))
    bev=torch.tensor(np.stack([x[2] for x in xs])).unsqueeze(1).float(); return pts,mask,bev

def fit(net,Xtr,ytr,fwd,epochs=40):
    opt=torch.optim.Adam(net.parameters(),1e-3); y=torch.tensor(ytr).float(); lossf=nn.BCEWithLogitsLoss()
    for e in range(epochs):
        net.train(); perm=torch.randperm(len(y))
        for i in range(0,len(y),64):
            idx=perm[i:i+64]; opt.zero_grad(); out=fwd(net,Xtr,idx); l=lossf(out,y[idx]); l.backward(); opt.step()
    return net

if __name__=="__main__":
    print("building data (clean train + degraded eval)...",flush=True)
    train,ev=build_data()
    # camera
    Xc=to_cam(train["cam"]["X"]); cam=fit(CameraNet(),Xc,train["cam"]["y"],lambda n,X,ix:n(X[ix]))
    # lidar / radar
    Pl=to_pb(train["lid"]["X"]); lidn=fit(LidarNet(),Pl,train["lid"]["y"],lambda n,X,ix:n(X[0][ix],X[1][ix],X[2][ix]))
    Pr=to_pb(train["rad"]["X"]); radn=fit(RadarNet(),Pr,train["rad"]["y"],lambda n,X,ix:n(X[0][ix],X[1][ix],X[2][ix]))
    for m in (cam,lidn,radn): m.eval()
    torch.save({"cam":cam.state_dict(),"lid":lidn.state_dict(),"rad":radn.state_dict()},f"{HERE}/detectors.pt")
    # eval: fair recall per condition (fire = sigmoid>0.5 on the GT box)
    import collections
    agg=collections.defaultdict(lambda:[0.,0.,0.,0])
    rows_out=[]
    with torch.no_grad():
        for s in ev:
            lcl,lll,lrl=[],[],[]                                     # per-object detector LOGITS (for weighted fusion)
            for (cc,lf,rf) in s["feats"]:
                lcl.append(float(cam(to_cam([cc])).item()))
                lll.append(float(lidn(*to_pb([lf])).item())); lrl.append(float(radn(*to_pb([rf])).item()))
            hcl=[int(x>0) for x in lcl]; hll=[int(x>0) for x in lll]; hrl=[int(x>0) for x in lrl]   # hit = logit>0
            n=len(lcl); a=agg[s["cls"]]; a[0]+=sum(hcl)/n; a[1]+=sum(hll)/n; a[2]+=sum(hrl)/n; a[3]+=1
            rows_out.append(dict(pid=s["pid"],cond=s["cond"],cls=s["cls"],
                logit_cam=lcl,logit_lidar=lll,logit_radar=lrl,hits_cam=hcl,hits_lidar=hll,hits_radar=hrl,
                rec_cam=sum(hcl)/n,rec_lidar=sum(hll)/n,rec_radar=sum(hrl)/n,**s["probes"]))
    json.dump(rows_out,open(f"{HERE}/env50_learned_rows.json","w"))
    print("\nFAIR per-modality recall (learned detectors; cf. old threshold recall in parentheses):")
    OLD={"clean":(1.00,1.00,0.95),"light":(0.70,1.00,0.97),"dense":(0.98,0.08,1.00),"lidaronly":(1.00,0.00,0.90)}
    for cls in ["clean","light","dense","lidaronly"]:
        c,l,r,n=agg[cls]; o=OLD.get(cls,(0,0,0))
        print(f"  {cls:10s} n={n:3d}  cam {c/n:.2f}({o[0]:.2f}) lidar {l/n:.2f}({o[1]:.2f}) radar {r/n:.2f}({o[2]:.2f})")
    print(f"\nsaved {HERE}/detectors.pt + env50_learned_rows.json")
