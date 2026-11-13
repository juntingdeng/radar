"""Multi-condition env for the modality selector: 4 regimes of comparable size, scored by the LEARNED detectors.
  night_clean  (05.11)           all modalities work
  night_glare  (05.11, REAL)     camera glare-blind, lidar/radar fine   <- the camera-FAILS regime
  day_clean    (05.08)           all modalities work
  day_smoke    (05.08 + corrected 10.15 smoke composite)  camera+lidar degrade
Detectors (from models.py) trained on CLEAN (night+day) object-vs-background, evaluated on all 4. Writes
env_multi_rows.json (per-object logits + probes + cls + base_id). Run bandit50/policy_weighted on it."""
import os, sys, json, numpy as np, cv2, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from models import (CameraNet,LidarNet,RadarNet,cam_crop,lidar_feat,radar_feat,radar_probes,to_cam,to_pb,fit)
R10="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code/sync/res/2026.10.15"
sys.path.insert(0,R10); import reconstruct_smoke as RS
SCR=next((p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)),os.path.expanduser("~/.cache/ti_setup_frames/rl_env"))  # persistent local first; session scratchpad fallback
torch.manual_seed(0); np.random.seed(0)
K_NEG=6   # background (non-object) candidates per frame -> FP denominator for precision / F1 / AP
NEGRNG=np.random.RandomState(123)   # dedicated RNG for EVAL negatives -> leaves the global RNG (training negatives/detectors) unchanged
NCACHE=os.path.expanduser("~/.cache/ti_setup_frames/lidar_scan_cache")
DCACHE=os.path.expanduser("~/.cache/ti_setup_frames/day08_lidar")
LIBc=np.load(f"{SCR}/smoke_library_corrected.npz",allow_pickle=True); lib=list(LIBc["lib"])
clean_bev10=np.nan_to_num(LIBc["clean_bev"].astype(float)); A10=LIBc["A"].astype(float)
Blum=float(np.percentile(LIBc["clean_cam"].astype(float).mean(2),50))

def load(npz):
    S=np.load(npz,allow_pickle=True); return S,json.loads(str(S["meta"])),S["la"],S["fa"]
NC,ncm,la,fa=load(f"{SCR}/scenes50.npz")
NG,ngm,_,_=load(f"{SCR}/glare.npz")
DY,dym,_,_=load(f"{SCR}/day.npz")

def probes(cam,lid,bev):
    g=cv2.cvtColor(cam.astype(np.uint8),cv2.COLOR_RGB2GRAY); th=cv2.resize(g,(16,12)).astype(float)
    near=int(((lid[:,0]>0)&(lid[:,0]<10)&(np.abs(lid[:,1])<10)&(lid[:,2]>0.3)&(lid[:,2]<4)).sum())
    rb,rp=radar_probes(bev,la,fa)
    return dict(probe_bright=float(th.mean()),probe_contrast=float(th.std()),
                probe_near_lap=float(cv2.Laplacian(g[320:,:],cv2.CV_32F).var()),probe_lidar_near=near,
                probe_radar_bev=rb,probe_radar_pts=rp)
def feats(cam,lid,bev,boxes): return [(cam_crop(cam.astype(np.uint8),b["bbox"]),lidar_feat(lid,b["fwd"],b["lat"]),radar_feat(bev,b["fwd"],b["lat"],la,fa)) for b in boxes]
def rand_bg(boxes,rng=np.random):
    for _ in range(20):
        cf=rng.uniform(7,24); cl=rng.uniform(-12,12)
        bb=[np.clip(320+cl*12,0,580),240,np.clip(320+cl*12+60,20,640),300]
        if all(not(abs((b['fwd'][0]+b['fwd'][1])/2-cf)<4 and abs((b['lat'][0]+b['lat'][1])/2-cl)<3) for b in boxes):
            return dict(bbox=bb,fwd=(cf-2.2,cf+2.2),lat=(cl-1,cl+1))
    return dict(bbox=[0,240,60,300],fwd=(20,24),lat=(10,12))

train={m:{"X":[],"y":[]} for m in ["cam","lid","rad"]}; ev=[]
def add_clean_train(cam,lid,bev,boxes):
    for b in boxes:
        c,lf,rf=feats(cam,lid,bev,[b])[0]
        train["cam"]["X"].append(c); train["cam"]["y"].append(1); train["lid"]["X"].append(lf); train["lid"]["y"].append(1); train["rad"]["X"].append(rf); train["rad"]["y"].append(1)
    for _ in range(3*len(boxes)):
        nb=rand_bg(boxes); c,lf,rf=feats(cam,lid,bev,[nb])[0]
        train["cam"]["X"].append(c); train["cam"]["y"].append(0); train["lid"]["X"].append(lf); train["lid"]["y"].append(0); train["rad"]["X"].append(rf); train["rad"]["y"].append(0)

# ---- night clean + night glare ----
for S,meta,cond,train_on in [(NC,ncm,"night_clean",True),(NG,ngm,"night_glare",False)]:
    for i,mt in enumerate(meta):
        cam=S["cams"][i].astype(float)
        if cam.max()==0: continue
        lid=np.load(f"{NCACHE}/l_{mt['lidar_idx']}.npy").astype(float); bev=S["bevs"][i].astype(float); boxes=mt["boxes"]
        if train_on: add_clean_train(cam,lid,bev,boxes)
        ev.append(dict(base_id=f"n{mt['pid']}",cls=cond,feats=feats(cam,lid,bev,boxes),
                       neg_feats=feats(cam,lid,bev,[rand_bg(boxes,NEGRNG) for _ in range(K_NEG)]),**probes(cam,lid,bev)))
# ---- day clean + day smoke ----
for i,mt in enumerate(dym):
    cam0=DY["cams"][i].astype(float)
    if cam0.max()==0: continue
    lid0=np.load(f"{DCACHE}/l_{mt['lidar_idx']}.npy").astype(float); bev0=np.nan_to_num(DY["bevs"][i].astype(float)); boxes=mt["boxes"]
    add_clean_train(cam0,lid0,bev0,boxes)
    ev.append(dict(base_id=f"d{mt['pid']}",cls="day_clean",feats=feats(cam0,lid0,bev0,boxes),
                   neg_feats=feats(cam0,lid0,bev0,[rand_bg(boxes,NEGRNG) for _ in range(K_NEG)]),**probes(cam0,lid0,bev0)))
    # day smoke: composite a corrected 10.15 layer (flip onto car side)
    ratio=float(np.percentile(cam0.mean(2),50))/Blum; flip=float(np.mean([(b['bbox'][0]+b['bbox'][2])/2 for b in boxes]))<320
    lay=lib[(i*5)%len(lib)]
    a=(lay["alpha"].astype(float)[:,::-1] if flip else lay["alpha"].astype(float)); Af=(lay["airlight"].astype(float)[:,::-1] if flip else lay["airlight"].astype(float))*ratio
    cam=np.clip(cam0*(1-a[...,None])+Af*a[...,None],0,255)
    bs=lay["backscatter"].astype(float).copy();
    if flip: bs[:,1]*=-1
    lid=RS._lidar_apply(lid0,bs)
    w=lay["bev_w"].astype(float); d=np.nan_to_num(lay["bev_smoke"].astype(float))-clean_bev10
    if flip: w=w[:,::-1]; d=d[:,::-1]
    bev=bev0+w*np.clip(d,0,None)
    ev.append(dict(base_id=f"d{mt['pid']}",cls="day_smoke",feats=feats(cam,lid,bev,boxes),
                   neg_feats=feats(cam,lid,bev,[rand_bg(boxes,NEGRNG) for _ in range(K_NEG)]),**probes(cam,lid,bev)))
    if i%10==0: print(f"day {i+1}/{len(dym)}",flush=True)
print(f"samples: {len(ev)}  train positives: {sum(train['cam']['y'])}",flush=True)

# ---- train detectors on CLEAN, score all ----
cam=fit(CameraNet(),to_cam(train["cam"]["X"]),train["cam"]["y"],lambda n,X,ix:n(X[ix]))
lidn=fit(LidarNet(),to_pb(train["lid"]["X"]),train["lid"]["y"],lambda n,X,ix:n(X[0][ix],X[1][ix],X[2][ix]))
radn=fit(RadarNet(),to_pb(train["rad"]["X"]),train["rad"]["y"],lambda n,X,ix:n(X[0][ix],X[1][ix],X[2][ix]))
for m in (cam,lidn,radn): m.eval()
rows=[]
import collections; agg=collections.defaultdict(lambda:[0.,0.,0.,0])
with torch.no_grad():
    for s in ev:
        lc=[float(cam(to_cam([f[0]])).item()) for f in s["feats"]]
        ll=[float(lidn(*to_pb([f[1]])).item()) for f in s["feats"]]
        lr=[float(radn(*to_pb([f[2]])).item()) for f in s["feats"]]
        nlc=[float(cam(to_cam([f[0]])).item()) for f in s["neg_feats"]]   # background candidates
        nll=[float(lidn(*to_pb([f[1]])).item()) for f in s["neg_feats"]]
        nlr=[float(radn(*to_pb([f[2]])).item()) for f in s["neg_feats"]]
        n=len(s["feats"]); hc=[int(x>0) for x in lc]; hl=[int(x>0) for x in ll]; hr=[int(x>0) for x in lr]
        a=agg[s["cls"]]; a[0]+=sum(hc)/n; a[1]+=sum(hl)/n; a[2]+=sum(hr)/n; a[3]+=1
        rows.append(dict(base_id=s["base_id"],pid=s["base_id"],cls=s["cls"],logit_cam=lc,logit_lidar=ll,logit_radar=lr,
            neg_logit_cam=nlc,neg_logit_lidar=nll,neg_logit_radar=nlr,
            hits_cam=hc,hits_lidar=hl,hits_radar=hr,
            probe_bright=s["probe_bright"],probe_contrast=s["probe_contrast"],probe_near_lap=s["probe_near_lap"],probe_lidar_near=s["probe_lidar_near"],
            probe_radar_bev=s["probe_radar_bev"],probe_radar_pts=s["probe_radar_pts"]))
json.dump(rows,open(f"{HERE}/env_multi_rows.json","w"))
print("\nFAIR recall by condition (learned detectors):")
for c in ["night_clean","night_glare","day_clean","day_smoke"]:
    if agg[c][3]: cc,l,r,nn=agg[c]; print(f"  {c:12s} n={nn:3d}  cam {cc/nn:.2f}  lidar {l/nn:.2f}  radar {r/nn:.2f}")
print(f"saved {HERE}/env_multi_rows.json ({len(rows)} samples)")
