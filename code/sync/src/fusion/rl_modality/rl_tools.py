"""RL modality tooling (merged). Subcommands: f1 (F1 eval) | label (static labeling) | viz (env sample)"""
import sys

# ======== src/fusion/rl_modality/eval_f1.py ========
# F1-based reward + precision/recall/AP for the modality selector (reviewer point #2).
# 
# The env now carries background NEGATIVE candidates (neg_logit_* from build_env_multi, K_NEG per frame) so false
# positives are counted:  TP = GT object with fused>0 ; FP = background cand with fused>0 ; FN = GT object with fused<=0.
# Detection of a candidate = (weighted logit sum  w . logit) > tau.  Reward = F1 - lambda*cost  (was recall - lambda*cost;
# recall alone never penalised FPs, so a policy had no reason to reject an unreliable modality -- reviewer pt #2).
# 
# Reports, per policy, on a SCENE-split held-out set: micro precision / recall / F1 (at tau=0) and AP (sklearn,
# threshold-independent, sweeps tau). Policies: fixed weights (camera / lidar / radar / equal), a LEARNED weighted
# policy trained on a differentiable soft-F1, and a per-frame ORACLE (best simplex weights). Run:
#   ENV_ROWS=env_multi_rows.json python eval_f1.py
def f1_main():
    import os, json, numpy as np, torch, torch.nn as nn
    from sklearn.metrics import average_precision_score
    HERE=os.path.dirname(os.path.abspath(__file__)); rng=np.random.default_rng(0); torch.manual_seed(0)
    HERE=os.path.dirname(os.path.abspath(__file__)); rng=np.random.default_rng(0); torch.manual_seed(0)
    HERE=os.path.dirname(os.path.abspath(__file__)); rng=np.random.default_rng(0); torch.manual_seed(0)
    rows=json.load(open(os.environ.get("ENV_ROWS", f"{HERE}/env_multi_rows.json")))
    if "neg_logit_cam" not in rows[0]:
        raise SystemExit("env has no neg_logit_* -- re-run build_env_multi.py (needs the K_NEG background candidates).")
    COST=np.array([0.3,0.4,1.0]); LAM=0.15; EPS=1e-9
    COST=np.array([0.3,0.4,1.0]); LAM=0.15; EPS=1e-9
    COST=np.array([0.3,0.4,1.0]); LAM=0.15; EPS=1e-9
    Pos=[np.stack([r["logit_cam"],r["logit_lidar"],r["logit_radar"]],1).astype(float) for r in rows]
    Neg=[np.stack([r["neg_logit_cam"],r["neg_logit_lidar"],r["neg_logit_radar"]],1).astype(float) for r in rows]
    pid=np.array([r["pid"] for r in rows]); cls=np.array([r["cls"] for r in rows]); N=len(rows)
    pid=np.array([r["pid"] for r in rows]); cls=np.array([r["cls"] for r in rows]); N=len(rows)
    pid=np.array([r["pid"] for r in rows]); cls=np.array([r["cls"] for r in rows]); N=len(rows)
    up=np.unique(pid); rng.shuffle(up); trp=set(up[:int(0.7*len(up))])
    up=np.unique(pid); rng.shuffle(up); trp=set(up[:int(0.7*len(up))])
    up=np.unique(pid); rng.shuffle(up); trp=set(up[:int(0.7*len(up))])
    tr=np.array([p in trp for p in pid]); te=~tr; tri,tei=np.where(tr)[0],np.where(te)[0]
    tr=np.array([p in trp for p in pid]); te=~tr; tri,tei=np.where(tr)[0],np.where(te)[0]
    tr=np.array([p in trp for p in pid]); te=~tr; tri,tei=np.where(tr)[0],np.where(te)[0]
    print(f"{N} frames, {len(up)} scenes -> train {tr.sum()} / test {te.sum()}  (K_neg/frame={Neg[0].shape[0]})")
    def probe_X():
        Xr=np.stack([[r["probe_bright"] for r in rows],[r["probe_contrast"] for r in rows],
                     np.log10(np.maximum([r["probe_near_lap"] for r in rows],1.)),
                     np.log10(np.maximum([r["probe_lidar_near"] for r in rows],10.)),
                     [r.get("probe_radar_bev",0.) for r in rows],
                     np.log10(np.maximum([r.get("probe_radar_pts",0.) for r in rows],1.))],1)
        Xr=(Xr-Xr.mean(0))/(Xr.std(0)+EPS)
        return torch.tensor(np.concatenate([Xr,np.ones((N,1))],1)).float()
    X=probe_X()
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
    g=np.array([[a,b,c] for a in(0,.25,.5,.75,1) for b in(0,.25,.5,.75,1) for c in(0,.25,.5,.75,1)],float)
    g=g[g.sum(1)>0]; grid=g/g.sum(1,keepdims=True)
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
    Mp=max(len(p) for p in Pos); Mn=max(len(n) for n in Neg)
    Mp=max(len(p) for p in Pos); Mn=max(len(n) for n in Neg)
    Lp=torch.zeros(N,Mp,3); Mpmask=torch.zeros(N,Mp); Ln=torch.zeros(N,Mn,3); Mnmask=torch.zeros(N,Mn)
    Lp=torch.zeros(N,Mp,3); Mpmask=torch.zeros(N,Mp); Ln=torch.zeros(N,Mn,3); Mnmask=torch.zeros(N,Mn)
    Lp=torch.zeros(N,Mp,3); Mpmask=torch.zeros(N,Mp); Ln=torch.zeros(N,Mn,3); Mnmask=torch.zeros(N,Mn)
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
    pol=Pol(X.shape[1]); opt=torch.optim.Adam(pol.parameters(),3e-3)
    for ep in range(700):
        opt.zero_grad(); loss=-soft_reward(pol(X[tri]),tri); loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(),5.0); opt.step()
    pol.eval()
    with torch.no_grad(): Wall=pol(X).numpy()
    res["learned(F1,weighted)"]=metrics(lambda i:Wall[i], tei)
    order=["camera","lidar","radar","equal","learned(F1,weighted)","oracle(F1)"]
    print("\n=== F1 reward: held-out precision / recall / F1 / AP (micro, scene-split) ===")
    print(f"{'policy':22s} {'prec':>5s} {'rec':>5s} {'F1':>5s} {'AP':>5s} {'cost':>5s} | {'rwd(F1)':>8s} {'rwd(recall-only)':>16s}")
    for k in order:
        v=res[k]; print(f"{k:22s} {v['precision']:5.2f} {v['recall']:5.2f} {v['f1']:5.2f} {v['AP']:5.2f} {v['cost']:5.2f} | {v['reward_F1']:8.3f} {v['reward_recall_only']:16.3f}")
    base=max(res['camera']['reward_F1'],res['equal']['reward_F1'])
    print(f"\nF1-reward headroom: oracle {res['oracle(F1)']['reward_F1']:.3f} - best-baseline {base:.3f} = {res['oracle(F1)']['reward_F1']-base:+.3f}")
    print(f"learned {res['learned(F1,weighted)']['reward_F1']:.3f} vs best-baseline {base:.3f} = {res['learned(F1,weighted)']['reward_F1']-base:+.3f}")
    json.dump(res,open(f"{HERE}/results_f1.json","w"),indent=1)
    print(f"saved {HERE}/results_f1.json")

# ======== src/fusion/rl_modality/label_objects.py ========
# Auto-label the STATIC parked-car objects of the 10.15 scene on the clean reference (one-time).
# Stationary rig + static objects => each object has a fixed image bbox + fixed lidar BEV box + radar footprint;
# per-frame detection outcomes against these are then real measurements. Cross-confirmed: YOLO image box (az-matched)
# <-> lidar cluster (8-22m) <-> radar strong cells. Writes objects.json next to this script.
def label_main():
    import sys, os, json
    import numpy as np, cv2
    HERE=os.path.dirname(os.path.abspath(__file__))
    REPO=os.path.abspath(os.path.join(HERE,"..","..","..",".."))
    SCR=[p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)][0]
    L=np.load(f"{SCR}/smoke_library_multisensor.npz",allow_pickle=True)
    cam=L["clean_cam"].astype(np.uint8); lid=L["clean_lid"].astype(float); bev=L["clean_bev"].astype(float)
    cam=L["clean_cam"].astype(np.uint8); lid=L["clean_lid"].astype(float); bev=L["clean_bev"].astype(float)
    cam=L["clean_cam"].astype(np.uint8); lid=L["clean_lid"].astype(float); bev=L["clean_bev"].astype(float)
    la,fa=L["la"],L["fa"]; fx,cx=float(L["fx"]),float(L["cx"])
    la,fa=L["la"],L["fa"]; fx,cx=float(L["fx"]),float(L["cx"])
    lid2=np.load(f"{SCR}/fog1_layers.npz")["clean_lid"].astype(float)
    cal=json.load(open(f"{REPO}/sync/res/2026.10.15/13.26.18/calibration.json"))
    L2C=np.linalg.inv(np.array(cal["camera_to_lidar_auto"]))
    from ultralytics import YOLO
    yolo=YOLO(f"{REPO}/yolov8n.pt")
    res=yolo.predict(source=cam[...,::-1],conf=0.25,verbose=False)[0]
    dets=[]
    for b,c,cf in zip(res.boxes.xyxy.cpu().numpy(),res.boxes.cls.cpu().numpy(),res.boxes.conf.cpu().numpy()):
        if res.names[int(c)] in ("car","truck","bus"): dets.append((b,float(cf)))
    print(f"YOLO cars on clean: {len(dets)}")
    m=(lid[:,0]>7)&(lid[:,0]<26)&(np.abs(lid[:,1])<16)&(lid[:,2]>-0.5)&(lid[:,2]<2.2)
    P=lid[m]
    gx=((P[:,0]-7)/0.4).astype(int); gy=((P[:,1]+16)/0.4).astype(int)
    gx=((P[:,0]-7)/0.4).astype(int); gy=((P[:,1]+16)/0.4).astype(int)
    H=np.zeros((int(19/0.4)+2,int(32/0.4)+2),int); np.add.at(H,(gx,gy),1)
    H=np.zeros((int(19/0.4)+2,int(32/0.4)+2),int); np.add.at(H,(gx,gy),1)
    occ=(H>=3).astype(np.uint8)
    n,labm=cv2.connectedComponents(occ,8)
    objs=[]
    RL,RF=np.meshgrid(la,fa); strong_thr=np.nanpercentile(bev,99.0)
    RL,RF=np.meshgrid(la,fa); strong_thr=np.nanpercentile(bev,99.0)
    def ct_in(Larr,f0,f1,l0,l1):
        return int(((Larr[:,0]>=f0)&(Larr[:,0]<=f1)&(Larr[:,1]>=l0)&(Larr[:,1]<=l1)&(Larr[:,2]>-0.5)&(Larr[:,2]<2.2)).sum())
    for k in range(1,n):
        cells=np.argwhere(labm==k)
        if len(cells)<4: continue
        sel=(labm[gx,gy]==k)
        Q=P[sel]
        f0,f1=float(Q[:,0].min()),float(Q[:,0].max()); l0,l1=float(Q[:,1].min()),float(Q[:,1].max())
        if (f1-f0)*(l1-l0)<0.4 or len(Q)<15: continue
        # STATIC check: cluster must persist in the independent fog-window clean scan
        n2=ct_in(lid2,f0-0.3,f1+0.3,l0-0.3,l1+0.3)
        if n2<0.3*len(Q):
            print(f"  dropped mover/transient at fwd {f0:.1f}-{f1:.1f} lat {l0:.1f}-{l1:.1f} ({len(Q)} pts vs {n2} in ref2)")
            continue
        # project cluster center to image for YOLO match
        ctr=np.array([Q[:,0].mean(),Q[:,1].mean(),Q[:,2].mean(),1.0])
        cc=L2C@ctr; u=fx*cc[0]/cc[2]+cx if cc[2]>0 else -1
        best=None
        for b,cf in dets:
            if b[0]-40<=u<=b[2]+40: best=(b,cf) if best is None or cf>best[1] else best
        # radar strong cells in footprint
        inbox=(RF>=f0-0.5)&(RF<=f1+0.5)&(RL>=l0-0.5)&(RL<=l1+0.5)
        ncells=int(np.nansum(bev[inbox]>strong_thr))
        if best is None: continue
        objs.append(dict(obj_id=f"car{len(objs)}",n_lidar=int(len(Q)),
            bev_box=[round(f0,2),round(f1,2),round(l0,2),round(l1,2)],
            img_box=[round(float(x),1) for x in best[0]],yolo_conf=round(best[1],3),
            radar_strong_clean=ncells,u_px=round(float(u),1)))
    print(json.dumps(objs,indent=1))
    json.dump(dict(objects=objs,strong_thr=float(strong_thr),
        note="static parked-car objects on the clean 10.15 scene (stationary rig). bev_box=[fwd0,fwd1,lat0,lat1]; "
             "img_box=xyxy on 640x480; radar_strong_clean=cells>p99(clean bev) in footprint"),
        open(f"{HERE}/objects.json","w"),indent=1)
    vis=cam.copy()
    for o in objs:
        x0,y0,x1,y1=[int(v) for v in o["img_box"]]
        cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,0),2); cv2.putText(vis,o["obj_id"],(x0,y0-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv2.imwrite(f"{HERE}/objects_clean.png",vis[...,::-1])
    print(f"saved {HERE}/objects.json + objects_clean.png ({len(objs)} objects)")

# ======== src/fusion/rl_modality/viz_placement.py ========
# Sample of what the object-level RL env actually feeds the detectors: clean 05.11 night base frames with the
# 10.15 smoke/fog layers PLACED ON the GT objects, across severities, all 3 modalities, GT boxes + real detector
# outcomes drawn. 2 base frames x [clean, smoke_mid, smoke_dense, fog_mid] -> rl_env_sample.png.
def viz_main():
    import os, sys, json, types
    import numpy as np, cv2
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    HERE=os.path.dirname(os.path.abspath(__file__))
    REPO=os.path.abspath(os.path.join(HERE,"..","..","..",".."))
    SCR=next((p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)),os.path.expanduser("~/.cache/ti_setup_frames/rl_env"))  # persistent local first; session scratchpad fallback
    sys.path.insert(0,f"{REPO}/sync/src"); sys.path.insert(0,f"{REPO}/sync/src/camera_projection")
    sys.path.insert(0,f"{REPO}/sync/src"); sys.path.insert(0,f"{REPO}/sync/src/camera_projection")
    sys.path.insert(0,f"{REPO}/sync/src/calibration"); sys.path.insert(0,f"{REPO}/sync/src/fusion/radar_first")
    sys.path.insert(0,f"{REPO}/sync/src/calibration"); sys.path.insert(0,f"{REPO}/sync/src/fusion/radar_first")
    sys.path.insert(0,f"{REPO}/sync/res/2026.10.15")
    os.chdir(REPO)
    import reconstruct_smoke as RS
    from yolo_to_lidar import _radar_setup, _radar_panel, _extract_frames, DEFAULT_COLOR_TOPIC
    from pathlib import Path
    RES="sync/res/2026.05.11/21.37.07"; RAW="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11"
    RES="sync/res/2026.05.11/21.37.07"; RAW="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11"
    LIDCACHE=os.path.expanduser("~/.cache/ti_setup_frames/lidar_scan_cache")
    SMOKE=f"{SCR}/smoke1_layers.npz"; FOG=f"{SCR}/fog1_layers.npz"
    SMOKE=f"{SCR}/smoke1_layers.npz"; FOG=f"{SCR}/fog1_layers.npz"
    Vs=np.load(SMOKE)
    okS=np.where(Vs["lidar_valid"]&Vs["radar_valid"])[0]; ld=np.nan_to_num(Vs["lidar_density"].astype(float),nan=-1)[okS]
    okS=np.where(Vs["lidar_valid"]&Vs["radar_valid"])[0]; ld=np.nan_to_num(Vs["lidar_density"].astype(float),nan=-1)[okS]
    def pick(t): return int(okS[np.argmin(np.abs(ld-t))])
    Vf=np.load(FOG); okF=np.where(Vf["lidar_valid"]&Vf["radar_valid"])[0]; kf=int(okF[np.argmax(Vf["density"][okF])])
    Vf=np.load(FOG); okF=np.where(Vf["lidar_valid"]&Vf["radar_valid"])[0]; kf=int(okF[np.argmax(Vf["density"][okF])])
    Vf=np.load(FOG); okF=np.where(Vf["lidar_valid"]&Vf["radar_valid"])[0]; kf=int(okF[np.argmax(Vf["density"][okF])])
    CONDS=[("clean",None,None),("smoke_mid",SMOKE,pick(0.5)),("smoke_dense",SMOKE,pick(1.0)),("fog_mid",FOG,kf)]
    B_lum=float(np.percentile(Vs["B"].astype(float).mean(2),50))
    rows_env=json.load(open(f"{HERE}/env_obj_rows.json"))
    pids=sorted(set(r["pid"] for r in rows_env))
    L={}
    for f in ["train","val","test_clear"]:
        for k,v in json.load(open("sync/res/benchmark_v1/labels/{}.json".format(f))).items(): L[k]=v
    frames=[]
    for line in open("sync/res/benchmark_v1/samples.jsonl"):
        s=json.loads(line)
        if s["seq"]!="2026.05.11" or s["pair_idx"] not in pids: continue
        bs=[b for b in L.get(s["sample_id"],[]) if b["forward"][0]>4 and b["forward"][1]<28 and b.get("bbox_xyxy")]
        if len(bs)>=2: frames.append((s["pair_idx"],s["modality"],bs))
    frames=sorted(frames)[:2]
    print("base frames:",[f[0] for f in frames],flush=True)
    cal=json.load(open(f"sync/res/benchmark_v1/calib/2026.05.11.json"))
    import detector as det_mod; det_mod.H5=f"{RAW}/radar/2026.05.11-21.37.07.h5"
    import detector as det_mod; det_mod.H5=f"{RAW}/radar/2026.05.11-21.37.07.h5"
    args=types.SimpleNamespace(radar_h5=det_mod.H5,cfg_file="mmWaveStudio/mobile_collection/rng40_res0004_tx3_rx4.lua",
      sync_csv=f"{RES}/sync_pairs.csv",sync_summary=f"{RES}/sync_summary.json",lidar_pcap=None,lidar_metadata=None,
      radar_bev_res=0.1,radar_percentile=99.9)
    loader,R11=_radar_setup(args,cal,doppler_mode="all",processing="legacy"); R11=np.array(R11)
    loader,R11=_radar_setup(args,cal,doppler_mode="all",processing="legacy"); R11=np.array(R11)
    bag_ts=np.load(f"{RES}/camera_index.npz")["color_bag_ts_ns"]
    camcache=Path(os.path.expanduser("~/.cache/ti_setup_frames/bench_2026.05.11"))
    from ultralytics import YOLO
    yolo=YOLO(f"{REPO}/yolov8n.pt")
    la,fa=Vs["la"],Vs["fa"]
    def cam_at(ci):
        d=_extract_frames(f"{RAW}/camera/20260511_213724.bag",DEFAULT_COLOR_TOPIC,[int(ci)],camcache,"color",bag_ts)
        im=d.get(int(ci)); return im.reshape(480,640,3) if (im is not None and im.ndim==1) else im
    def lid_at(li): return np.load(f"{LIDCACHE}/l_{int(li)}.npy").astype(float)
    def yb(img):
        r=yolo.predict(source=img[...,::-1],conf=0.25,verbose=False)[0]
        return [b for b,c in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy()) if r.names[int(c)] in ("car","truck","bus")]
    def iou(a,b):
        x0,y0,x1,y1=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
        inter=max(0,x1-x0)*max(0,y1-y0)
        return inter/max(1e-6,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
    NF=len(frames); fig,ax=plt.subplots(3*NF,len(CONDS),figsize=(5*len(CONDS),3.6*3*NF))
    NF=len(frames); fig,ax=plt.subplots(3*NF,len(CONDS),figsize=(5*len(CONDS),3.6*3*NF))
    for fi,(pid,mod,boxes) in enumerate(frames):
        cam0=cam_at(mod["camera_color_idx"]).astype(float); lid0=lid_at(mod["lidar_idx"])
        bev0,_,_=_radar_panel(loader,pid); bev0f=np.where(np.isfinite(bev0),bev0,0.0)
        ratio=float(np.percentile(cam0.mean(2),50))/B_lum
        pcol=float(np.mean([(b["bbox_xyxy"][0]+b["bbox_xyxy"][2])/2 for b in boxes]))
        paz=float(np.degrees(np.arctan2(np.mean([(b["lateral"][0]+b["lateral"][1])/2 for b in boxes]),
                                        np.mean([(b["forward"][0]+b["forward"][1])/2 for b in boxes]))))
        for c,(cname,npz,k) in enumerate(CONDS):
            if npz is None: camX,lidX,bevX=cam0.astype(np.uint8),lid0,bev0f
            else:
                camX,lidX,bevX,_=RS.reconstruct_from_video(npz,k,base_cam=cam0,base_lid=lid0,base_bev=bev0f,
                                  airlight_scale=ratio,place_col=pcol,place_az_deg=paz); camX=camX.astype(np.uint8)
            dets=yb(camX)
            axc=ax[3*fi,c]; axc.imshow(camX); axc.axis("off")
            nhit=0
            for b in boxes:
                gt=np.array(b["bbox_xyxy"]); hit=any(iou(gt,y)>=0.3 for y in dets); nhit+=hit
                x0,y0,x1,y1=[int(v) for v in gt]
                axc.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec="lime" if hit else "red",lw=2))
            axc.set_title(f"{'pid'+str(pid)+'  ' if c==0 else ''}{cname}\ncamera YOLO {nhit}/{len(boxes)}",fontsize=9)
            axl=ax[3*fi+1,c]; m=(lidX[:,0]>-2)&(lidX[:,0]<26)&(np.abs(lidX[:,1])<22)
            axl.scatter(lidX[m,1],lidX[m,0],s=1.2,c="0.5")
            for b in boxes:
                axl.add_patch(plt.Rectangle((b["lateral"][0],b["forward"][0]),b["lateral"][1]-b["lateral"][0],
                              b["forward"][1]-b["forward"][0],fill=False,ec="deepskyblue",lw=1.3))
            nl=sum(int(((lidX[:,0]>=b["forward"][0])&(lidX[:,0]<=b["forward"][1])&(lidX[:,1]>=b["lateral"][0])
                       &(lidX[:,1]<=b["lateral"][1])&(lidX[:,2]>-0.5)&(lidX[:,2]<2.5)).sum()) for b in boxes)
            axl.set_xlim(22,-22); axl.set_ylim(0,26); axl.set_aspect("equal"); axl.set_title(f"lidar (pts in GT {nl})",fontsize=9)
            axr=ax[3*fi+2,c]; RS.radar_axes(axr,bevX,la,fa,R11); axr.set_title("radar BEV",fontsize=9)
            if c==0:
                ax[3*fi,0].set_ylabel("camera",fontsize=10); ax[3*fi+1,0].set_ylabel("lidar BEV",fontsize=10)
                ax[3*fi+2,0].set_ylabel("radar BEV",fontsize=10)
    fig.suptitle("What the object-level RL env feeds the detectors: 10.15 smoke/fog layers PLACED ON the GT objects of clean 05.11 night frames.\n"
                 "Camera box lime=YOLO hit / red=miss; lidar cyan=GT box (pts-in-box drives the lidar hit); radar BEV (house render). Plume placed at the objects' column+bearing.",
                 fontsize=11,fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(f"{HERE}/rl_env_sample.png",dpi=80)
    plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(f"{HERE}/rl_env_sample.png",dpi=80)
    print(f"saved {HERE}/rl_env_sample.png")

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'f1': f1_main, 'label': label_main, 'viz': viz_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
