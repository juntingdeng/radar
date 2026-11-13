"""Generate the 50-scene / 16-layer composites and save a cam|lidar|radar inspection panel per sample.

SCORING LIVES IN models.py (learned per-modality detectors) -- this script does NOT score detection. The earlier
version used YOLO (the GT labeler) for camera + occupancy thresholds for lidar/radar, which is CIRCULAR and inflated
camera/radar; that code is removed. Panels here are annotated with RAW DESCRIPTIVE stats only (lidar point-count in
the GT box, radar strong-cell count) -- measurements, not detection claims.

Env = 50 clean night base scenes x {clean + 4 of the 16 real layers (rotating, all 16 covered)}; physics composites
(no FLUX); per-scene flip places the 10.15 dense core on the scene's car side. Writes panels to rl_env50_frames/.
To build the RL training data + fair recall, run models.py (it recomputes the same composites and scores them with
the learned detectors)."""
import os, sys, json, numpy as np, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R10="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code/sync/res/2026.10.15"
sys.path.insert(0,R10); import reconstruct_smoke as RS
SCR=next((p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)),os.path.expanduser("~/.cache/ti_setup_frames/rl_env"))  # persistent local first; session scratchpad fallback
OUT=f"{R10}/rl_env50_frames"; os.makedirs(OUT,exist_ok=True)
S=np.load(f"{SCR}/scenes50.npz",allow_pickle=True); meta=json.loads(str(S["meta"])); la,fa,R=S["la"],S["fa"],S["R"]
L=np.load(f"{SCR}/smoke_library_corrected.npz",allow_pickle=True); lib=list(L["lib"])
clean_bev10=L["clean_bev"].astype(float); A=L["A"].astype(float); Blum=float(np.percentile(L["clean_cam"].astype(float).mean(2),50))
LIDC=os.path.expanduser("~/.cache/ti_setup_frames/lidar_scan_cache")
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
