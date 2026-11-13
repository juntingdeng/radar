"""RL env builders (merged; PARALLEL studies). Subcommands: measured | scenes50 | multi | corrected"""
import sys

# ======== src/fusion/rl_modality/build_env.py ========
# Build the RL modality-selection environment table from MEASURED per-frame quantities (no invented data).
# 
# 495 real frames (smoke1 298 + fog1 197, clean -> dense smoke -> fog). Per frame:
#   VISIBILITY (the reward's ingredients; full-measurement, computed offline):
#     vis_cam   = far-band sharpness ratio vs the clean background (Laplacian variance, rows 120..300)
#                 -- real smoke BLURS + washes the scene (the discriminator lesson); clipped 0..1
#     vis_lidar = far-field survival: lidar_far / clean base_far (stored counts), clipped 0..1
#     vis_radar = strong-cell recall vs the window's median clean radar reference (stored crops, r<16m)
#                 normalized by the clean-frame median recall, clipped 0..1
#   PROBES (the agent's cheap state; must NOT include the full measurements):
#     thumbnail mean brightness / contrast (16x12 gray), near-band Laplacian, lidar NEAR count (cheap range-image
#     stat), frame-of-day index. The far-band measurements and radar quantities are NOT in the state -- the agent
#     must learn their value from experience (that is the selection problem).
# Writes env_table.npz next to this script.
def measured_main():
    import os, sys, json
    import numpy as np, cv2
    HERE=os.path.dirname(os.path.abspath(__file__))
    REPO=os.path.abspath(os.path.join(HERE,"..","..","..",".."))
    SCR=next((p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)),os.path.expanduser("~/.cache/ti_setup_frames/rl_env"))  # persistent local first; session scratchpad fallback
    FARBAND=slice(120,300)   # image rows holding the far scene (trees + road across)
    rows=[]
    for video in ["smoke1","fog1"]:
        src=f"{SCR}/{video}_layers.npz"
        if not os.path.exists(src): src=f"{REPO}/sync/res/2026.10.15/{video}_layers.npz"
        V=np.load(src)
        B=V["B"].astype(np.uint8)
        gB=cv2.cvtColor(B,cv2.COLOR_RGB2GRAY)
        lapB=float(cv2.Laplacian(gB[FARBAND],cv2.CV_32F).var())
        # radar: per-frame strong-cell recall vs the stored median clean reference (same machinery as the proof)
        MREF=V["clean_bev_median_crop"].astype(np.float32)
        f0,f1,l0,l1=[int(x) for x in V["crop"]]
        la,fa=V["la"],V["fa"]; RL,RF=np.meshgrid(la,fa); RNGc=np.hypot(RF,RL)[f0:f1,l0:l1]
        valid=(RNGc>2)&(RNGc<16)
        strong=valid&(MREF>np.percentile(MREF[valid],99))
        sm=V["bev_smoke_crop"].astype(np.float32)
        rec=np.array([float(np.mean(sm[k][strong]>=MREF[strong]-3.0)) for k in range(len(sm))])
        base_far=float(V["base_far"]); lidar_far=V["lidar_far"].astype(float)
        lidar_near=V["lidar_near"].astype(float)
        vsrc=f"{SCR}/{video}.mp4"
        if not os.path.exists(vsrc): vsrc=f"{REPO}/sync/res/2026.10.15/{video}.mp4"
        # sequential decode (seek-based reads glitch): layer frames are every STRIDE-th video frame
        cap=cv2.VideoCapture(vsrc)
        frames={}
        want=set(int(x) for x in V["frame_idx"]); i=0
        while True:
            ok,fr=cap.read()
            if not ok: break
            if i in want: frames[i]=fr
            i+=1
        assert len(frames)==len(want),(len(frames),len(want))
        N=len(V["frame_idx"])
        rec_clean=[]
        tmp=[]
        for k in range(N):
            fr=frames[int(V["frame_idx"][k])]
            g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
            lap=float(cv2.Laplacian(g[FARBAND],cv2.CV_32F).var())
            th=cv2.resize(g,(16,12)).astype(float)
            tmp.append(dict(video=video,k=k,el=float(V["el"][k]),
                cam_density=float(V["density"][k]),
                lidar_valid=bool(V["lidar_valid"][k]),radar_valid=bool(V["radar_valid"][k]),
                vis_cam=lap/lapB,
                vis_lidar=min(1.0,lidar_far[k]/base_far) if V["lidar_valid"][k] and lidar_far[k]>=0 else np.nan,
                rec_radar=rec[k] if V["radar_valid"][k] else np.nan,
                probe_bright=float(th.mean()),probe_contrast=float(th.std()),
                probe_near_lap=float(cv2.Laplacian(g[320:,:],cv2.CV_32F).var()),
                probe_lidar_near=float(lidar_near[k]) if lidar_near[k]>=0 else np.nan))
            if V["density"][k]<0.05 and V["lidar_valid"][k] and lidar_near[k]<3200 and V["radar_valid"][k]:
                rec_clean.append(rec[k])
        recn=float(np.median(rec_clean))
        for r in tmp:
            r["vis_radar"]=min(1.0,r["rec_radar"]/recn) if np.isfinite(r["rec_radar"]) else np.nan
            r["vis_cam"]=min(1.0,r["vis_cam"])
        rows+=tmp
        print(f"{video}: {N} frames, clean radar recall {recn:.3f}, "
              f"vis_cam {np.nanmin([r['vis_cam'] for r in tmp]):.2f}-{np.nanmax([r['vis_cam'] for r in tmp]):.2f}, "
              f"vis_lidar min {np.nanmin([r['vis_lidar'] for r in tmp]):.2f}",flush=True)
    keys=["cam_density","vis_cam","vis_lidar","vis_radar","probe_bright","probe_contrast","probe_near_lap","probe_lidar_near","el"]
    out={k:np.array([r.get(k,np.nan) for r in rows],float) for k in keys}
    out["video"]=np.array([r["video"] for r in rows]); out["k"]=np.array([r["k"] for r in rows])
    out["video"]=np.array([r["video"] for r in rows]); out["k"]=np.array([r["k"] for r in rows])
    out["lidar_valid"]=np.array([r["lidar_valid"] for r in rows]); out["radar_valid"]=np.array([r["radar_valid"] for r in rows])
    out["lidar_valid"]=np.array([r["lidar_valid"] for r in rows]); out["radar_valid"]=np.array([r["radar_valid"] for r in rows])
    np.savez(f"{HERE}/env_table.npz",**out,
        note="per-frame measured visibility (reward ingredients) + cheap probes (state). vis_* in 0..1; "
             "NaN vis = modality had no valid same-moment measurement (frame excluded from training).")
    print(f"saved {HERE}/env_table.npz ({len(rows)} frames)")

# ======== src/fusion/rl_modality/build_env50.py ========
# Generate the 50-scene / 16-layer composites and save a cam|lidar|radar inspection panel per sample.
# 
# SCORING LIVES IN models.py (learned per-modality detectors) -- this script does NOT score detection. The earlier
# version used YOLO (the GT labeler) for camera + occupancy thresholds for lidar/radar, which is CIRCULAR and inflated
# camera/radar; that code is removed. Panels here are annotated with RAW DESCRIPTIVE stats only (lidar point-count in
# the GT box, radar strong-cell count) -- measurements, not detection claims.
# 
# Env = 50 clean night base scenes x {clean + 4 of the 16 real layers (rotating, all 16 covered)}; physics composites
# (no FLUX); per-scene flip places the 10.15 dense core on the scene's car side. Writes panels to rl_env50_frames/.
# To build the RL training data + fair recall, run models.py (it recomputes the same composites and scores them with
# the learned detectors).
def scenes50_main():
    import os, sys, json, numpy as np, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import os, sys, json, numpy as np, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import os, sys, json, numpy as np, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R10="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code/sync/res/2026.10.15"
    sys.path.insert(0,R10); import reconstruct_smoke as RS
    sys.path.insert(0,R10); import reconstruct_smoke as RS
    SCR=next((p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)),os.path.expanduser("~/.cache/ti_setup_frames/rl_env"))  # persistent local first; session scratchpad fallback
    OUT=f"{R10}/rl_env50_frames"; os.makedirs(OUT,exist_ok=True)
    OUT=f"{R10}/rl_env50_frames"; os.makedirs(OUT,exist_ok=True)
    S=np.load(f"{SCR}/scenes50.npz",allow_pickle=True); meta=json.loads(str(S["meta"])); la,fa,R=S["la"],S["fa"],S["R"]
    S=np.load(f"{SCR}/scenes50.npz",allow_pickle=True); meta=json.loads(str(S["meta"])); la,fa,R=S["la"],S["fa"],S["R"]
    S=np.load(f"{SCR}/scenes50.npz",allow_pickle=True); meta=json.loads(str(S["meta"])); la,fa,R=S["la"],S["fa"],S["R"]
    L=np.load(f"{SCR}/smoke_library_corrected.npz",allow_pickle=True); lib=list(L["lib"])
    L=np.load(f"{SCR}/smoke_library_corrected.npz",allow_pickle=True); lib=list(L["lib"])
    clean_bev10=L["clean_bev"].astype(float); A=L["A"].astype(float); Blum=float(np.percentile(L["clean_cam"].astype(float).mean(2),50))
    clean_bev10=L["clean_bev"].astype(float); A=L["A"].astype(float); Blum=float(np.percentile(L["clean_cam"].astype(float).mean(2),50))
    clean_bev10=L["clean_bev"].astype(float); A=L["A"].astype(float); Blum=float(np.percentile(L["clean_cam"].astype(float).mean(2),50))
    LIDC=os.path.expanduser("~/.cache/ti_setup_frames/lidar_scan_cache")
    RL,RF=np.meshgrid(la,fa); LF=R[0,0]*RF+R[0,1]*RL+R[0,3]; LLw=R[1,0]*RF+R[1,1]*RL+R[1,3]
    RL,RF=np.meshgrid(la,fa); LF=R[0,0]*RF+R[0,1]*RL+R[0,3]; LLw=R[1,0]*RF+R[1,1]*RL+R[1,3]
    RL,RF=np.meshgrid(la,fa); LF=R[0,0]*RF+R[0,1]*RL+R[0,3]; LLw=R[1,0]*RF+R[1,1]*RL+R[1,3]
    def pts_in(P,b): return int(((P[:,0]>=b["fwd"][0])&(P[:,0]<=b["fwd"][1])&(P[:,1]>=b["lat"][0])&(P[:,1]<=b["lat"][1])&(P[:,2]>-0.5)&(P[:,2]<2.5)).sum())
    def cells_in(bev,b,thr): m=(RF>=b["fwd"][0]-0.5)&(RF<=b["fwd"][1]+0.5)&(RL>=b["lat"][0]-0.5)&(RL<=b["lat"][1]+0.5); return int((bev[m]>thr).sum())
    def flipp(x,kind,fl):
        if not fl: return x
        return x[:,::-1] if kind!="pts" else (lambda p:(p.__setitem__((slice(None),1),-p[:,1]),p)[1])(x.copy())
    n_panels=0
    for si,m in enumerate(meta):
        cam0=S["cams"][si].astype(float)
        if cam0.max()==0: continue
        lid0=np.load(f"{LIDC}/l_{m['lidar_idx']}.npy").astype(float); bev0=S["bevs"][si].astype(float)
        boxes=m["boxes"]; ratio=float(np.percentile(cam0.mean(2),50))/Blum
        flip=float(np.mean([(b['bbox'][0]+b['bbox'][2])/2 for b in boxes]))<320
        thr=float(np.percentile(bev0[bev0>0],99.0)); vmn,vmx=np.percentile(bev0[bev0>0],[2,99.8])
        def panel(tag,cam,lid,bev):
            fig,ax=plt.subplots(1,3,figsize=(15,4.2))
            ax[0].imshow(cam.astype(np.uint8)); ax[0].axis("off")
            for b in boxes: ax[0].add_patch(plt.Rectangle((b["bbox"][0],b["bbox"][1]),b["bbox"][2]-b["bbox"][0],b["bbox"][3]-b["bbox"][1],fill=False,ec="lime",lw=1.5))
            ax[0].set_title("camera",fontsize=9)
            mm=(lid[:,0]>-2)&(lid[:,0]<26)&(np.abs(lid[:,1])<22); ax[1].scatter(lid[mm,1],lid[mm,0],s=1,c="0.55")
            for b in boxes: ax[1].add_patch(plt.Rectangle((b["lat"][0],b["fwd"][0]),b["lat"][1]-b["lat"][0],b["fwd"][1]-b["fwd"][0],fill=False,ec="deepskyblue",lw=1.3))
            ax[1].set_xlim(22,-22); ax[1].set_ylim(0,26); ax[1].set_aspect("equal"); ax[1].set_title(f"lidar (box pts: {[pts_in(lid,b) for b in boxes]})",fontsize=8)
            b2=np.where(np.isfinite(bev)&(bev!=0),bev,np.nan); ax[2].pcolormesh(LLw,LF,b2,cmap="viridis",shading="gouraud",vmin=vmn,vmax=vmx)
            ax[2].set_xlim(25,-25); ax[2].set_ylim(0,25); ax[2].set_aspect("equal"); ax[2].set_title(f"radar (box strong cells: {[cells_in(bev,b,thr) for b in boxes]})",fontsize=8)
            fig.suptitle(tag+"  (raw stats, not detection -- score via models.py)",fontsize=8); plt.tight_layout(rect=[0,0,1,0.93]); plt.savefig(f"{OUT}/{tag}.png",dpi=70); plt.close(fig)
        panel(f"s{m['pid']:05d}_clean",cam0,lid0,bev0); n_panels+=1
        for j in range(4):
            k=(si*4+j)%16; lay=lib[k]
            if lay.get("cam_ok",True) and lay["alpha"] is not None:
                a=flipp(lay["alpha"].astype(float),"img",flip); Af=flipp(lay["airlight"].astype(float),"img",flip)*ratio
                cam=np.clip(cam0*(1-a[...,None])+Af*a[...,None],0,255)
            else: cam=cam0
            bs=flipp(lay["backscatter"].astype(float),"pts",flip); lid=RS._lidar_apply(lid0,bs)
            w=flipp(lay["bev_w"].astype(float),"bev",flip); d=flipp((lay["bev_smoke"].astype(float)-clean_bev10),"bev",flip)
            bev=bev0+w*np.clip(d,0,None)
            panel(f"s{m['pid']:05d}_layer{k:02d}",cam,lid,bev); n_panels+=1
        if si%10==0: print(f"scene {si+1}/{len(meta)} ({n_panels} panels)",flush=True)
    print(f"DONE {n_panels} inspection panels -> {OUT}/ (scoring: models.py)")

# ======== src/fusion/rl_modality/build_env_multi.py ========
# Multi-condition env for the modality selector: 4 regimes of comparable size, scored by the LEARNED detectors.
#   night_clean  (05.11)           all modalities work
#   night_glare  (05.11, REAL)     camera glare-blind, lidar/radar fine   <- the camera-FAILS regime
#   day_clean    (05.08)           all modalities work
#   day_smoke    (05.08 + corrected 10.15 smoke composite)  camera+lidar degrade
# Detectors (from models.py) trained on CLEAN (night+day) object-vs-background, evaluated on all 4. Writes
# env_multi_rows.json (per-object logits + probes + cls + base_id). Run bandit50/policy_weighted on it.
def multi_main():
    import os, sys, json, numpy as np, cv2, torch
    HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
    HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
    from models import (CameraNet,LidarNet,RadarNet,cam_crop,lidar_feat,radar_feat,radar_probes,to_cam,to_pb,fit)
    R10="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code/sync/res/2026.10.15"
    sys.path.insert(0,R10); import reconstruct_smoke as RS
    sys.path.insert(0,R10); import reconstruct_smoke as RS
    SCR=next((p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)),os.path.expanduser("~/.cache/ti_setup_frames/rl_env"))  # persistent local first; session scratchpad fallback
    torch.manual_seed(0); np.random.seed(0)
    torch.manual_seed(0); np.random.seed(0)
    K_NEG=6   # background (non-object) candidates per frame -> FP denominator for precision / F1 / AP
    NEGRNG=np.random.RandomState(123)   # dedicated RNG for EVAL negatives -> leaves the global RNG (training negatives/detectors) unchanged
    NCACHE=os.path.expanduser("~/.cache/ti_setup_frames/lidar_scan_cache")
    DCACHE=os.path.expanduser("~/.cache/ti_setup_frames/day08_lidar")
    LIBc=np.load(f"{SCR}/smoke_library_corrected.npz",allow_pickle=True); lib=list(LIBc["lib"])
    LIBc=np.load(f"{SCR}/smoke_library_corrected.npz",allow_pickle=True); lib=list(LIBc["lib"])
    clean_bev10=np.nan_to_num(LIBc["clean_bev"].astype(float)); A10=LIBc["A"].astype(float)
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
    train={m:{"X":[],"y":[]} for m in ["cam","lid","rad"]}; ev=[]
    def add_clean_train(cam,lid,bev,boxes):
        for b in boxes:
            c,lf,rf=feats(cam,lid,bev,[b])[0]
            train["cam"]["X"].append(c); train["cam"]["y"].append(1); train["lid"]["X"].append(lf); train["lid"]["y"].append(1); train["rad"]["X"].append(rf); train["rad"]["y"].append(1)
        for _ in range(3*len(boxes)):
            nb=rand_bg(boxes); c,lf,rf=feats(cam,lid,bev,[nb])[0]
            train["cam"]["X"].append(c); train["cam"]["y"].append(0); train["lid"]["X"].append(lf); train["lid"]["y"].append(0); train["rad"]["X"].append(rf); train["rad"]["y"].append(0)
    for S,meta,cond,train_on in [(NC,ncm,"night_clean",True),(NG,ngm,"night_glare",False)]:
        for i,mt in enumerate(meta):
            cam=S["cams"][i].astype(float)
            if cam.max()==0: continue
            lid=np.load(f"{NCACHE}/l_{mt['lidar_idx']}.npy").astype(float); bev=S["bevs"][i].astype(float); boxes=mt["boxes"]
            if train_on: add_clean_train(cam,lid,bev,boxes)
            ev.append(dict(base_id=f"n{mt['pid']}",cls=cond,feats=feats(cam,lid,bev,boxes),
                           neg_feats=feats(cam,lid,bev,[rand_bg(boxes,NEGRNG) for _ in range(K_NEG)]),**probes(cam,lid,bev)))
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
    cam=fit(CameraNet(),to_cam(train["cam"]["X"]),train["cam"]["y"],lambda n,X,ix:n(X[ix]))
    lidn=fit(LidarNet(),to_pb(train["lid"]["X"]),train["lid"]["y"],lambda n,X,ix:n(X[0][ix],X[1][ix],X[2][ix]))
    radn=fit(RadarNet(),to_pb(train["rad"]["X"]),train["rad"]["y"],lambda n,X,ix:n(X[0][ix],X[1][ix],X[2][ix]))
    for m in (cam,lidn,radn): m.eval()
    rows=[]
    import collections; agg=collections.defaultdict(lambda:[0.,0.,0.,0])
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

# ======== src/fusion/rl_modality/build_corrected_lib.py ========
# Rebuild the smoke layer library from the CORRECTLY-SYNCED smoke1_layers.npz (built with camera_sync_pairs.csv,
# carries lidar_valid/radar_valid). Only lidar_valid & radar_valid frames -> same-instant camera+lidar+radar triplets.
# Replaces the stale smoke_library_multisensor.npz (built from the old span_map manual pairing w the dropout mispair).
# Output structure matches what models.py expects (lib[*] with alpha/airlight/backscatter/bev_smoke/bev_w/density/
# cam_ok + shared clean_cam/clean_lid/clean_bev/A/la/fa/R). Radar stored as the measured near-field DELTA in bev_smoke
# with clean_bev=0 so models.py's d=bev_smoke-clean_bev gives the delta directly.
def corrected_main():
    import os, numpy as np, cv2
    SCR=next((p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)),os.path.expanduser("~/.cache/ti_setup_frames/rl_env"))  # persistent local first; session scratchpad fallback
    V=np.load(f"{SCR}/smoke1_layers.npz")
    el=V["el"]; lv=V["lidar_valid"]; rv=V["radar_valid"]; near=V["lidar_near"]
    el=V["el"]; lv=V["lidar_valid"]; rv=V["radar_valid"]; near=V["lidar_near"]
    el=V["el"]; lv=V["lidar_valid"]; rv=V["radar_valid"]; near=V["lidar_near"]
    el=V["el"]; lv=V["lidar_valid"]; rv=V["radar_valid"]; near=V["lidar_near"]
    la,fa=V["la"],V["fa"]; f0,f1,l0,l1=[int(x) for x in V["crop"]]
    la,fa=V["la"],V["fa"]; f0,f1,l0,l1=[int(x) for x in V["crop"]]
    mref=V["clean_bev_median_crop"].astype(np.float32)             # 10.15 clean radar median ref (for the delta)
    cand=[k for k in range(len(el)) if 1770.4<=el[k]<=1776 and lv[k] and rv[k]]
    sel=[cand[i] for i in np.linspace(0,len(cand)-1,16).astype(int)]   # 16 valid frames spanning the event
    print("corrected library layers (el, cam_alpha, lidar_near):")
    LIB=[]
    for k in sel:
        bs=V["bs_pts"][int(V["bs_off"][k]):int(V["bs_off"][k+1])].astype(np.float16)
        sm=np.zeros((len(fa),len(la)),np.float32); sm[f0:f1,l0:l1]=V["bev_smoke_crop"][k].astype(np.float32)-mref   # DELTA
        w =np.zeros((len(fa),len(la)),np.float32); w[f0:f1,l0:l1]=V["bev_w_crop"][k].astype(np.float32)
        air=cv2.resize(V["airlight_ds4"][k].astype(np.float32),(640,480))
        LIB.append(dict(el=float(el[k]),alpha=(V["alpha"][k].astype(np.float32)/255),airlight=air,
                        backscatter=bs,bev_smoke=np.nan_to_num(sm),bev_w=np.nan_to_num(w),
                        density=int(near[k]),cam_ok=True))
        print(f"  el {el[k]:.1f}  cam_a {float(V['density'][k]):.2f}  near {int(near[k])}")
    np.savez_compressed(f"{SCR}/smoke_library_corrected.npz",lib=np.array(LIB,dtype=object),
        clean_cam=V["B"],                                  # daytime clean background (for Blum airlight scaling)
        clean_lid=V["clean_lid"],clean_bev=np.zeros((len(fa),len(la)),np.float32),   # 0 -> models.py d = bev_smoke(delta)
        A=V["A"],la=la,fa=fa,R=V["R"])
    print(f"saved {SCR}/smoke_library_corrected.npz ({len(LIB)} valid same-instant layers; stale 16-layer lib retired)")

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'measured': measured_main, 'scenes50': scenes50_main, 'multi': multi_main, 'corrected': corrected_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
