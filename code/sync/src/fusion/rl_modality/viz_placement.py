"""Sample of what the object-level RL env actually feeds the detectors: clean 05.11 night base frames with the
10.15 smoke/fog layers PLACED ON the GT objects, across severities, all 3 modalities, GT boxes + real detector
outcomes drawn. 2 base frames x [clean, smoke_mid, smoke_dense, fog_mid] -> rl_env_sample.png."""
import os, sys, json, types
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
REPO=os.path.abspath(os.path.join(HERE,"..","..","..",".."))
SCR=next((p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)),os.path.expanduser("~/.cache/ti_setup_frames/rl_env"))  # persistent local first; session scratchpad fallback
sys.path.insert(0,f"{REPO}/sync/src"); sys.path.insert(0,f"{REPO}/sync/src/camera_projection")
sys.path.insert(0,f"{REPO}/sync/src/calibration"); sys.path.insert(0,f"{REPO}/sync/src/fusion/radar_first")
sys.path.insert(0,f"{REPO}/sync/res/2026.10.15")
os.chdir(REPO)
import reconstruct_smoke as RS
from yolo_to_lidar import _radar_setup, _radar_panel, _extract_frames, DEFAULT_COLOR_TOPIC
from pathlib import Path

RES="sync/res/2026.05.11/21.37.07"; RAW="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11"
LIDCACHE=os.path.expanduser("~/.cache/ti_setup_frames/lidar_scan_cache")
SMOKE=f"{SCR}/smoke1_layers.npz"; FOG=f"{SCR}/fog1_layers.npz"
Vs=np.load(SMOKE)
okS=np.where(Vs["lidar_valid"]&Vs["radar_valid"])[0]; ld=np.nan_to_num(Vs["lidar_density"].astype(float),nan=-1)[okS]
def pick(t): return int(okS[np.argmin(np.abs(ld-t))])
Vf=np.load(FOG); okF=np.where(Vf["lidar_valid"]&Vf["radar_valid"])[0]; kf=int(okF[np.argmax(Vf["density"][okF])])
CONDS=[("clean",None,None),("smoke_mid",SMOKE,pick(0.5)),("smoke_dense",SMOKE,pick(1.0)),("fog_mid",FOG,kf)]
B_lum=float(np.percentile(Vs["B"].astype(float).mean(2),50))

# two base frames (reuse the ones the env already processed = camera cached)
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
args=types.SimpleNamespace(radar_h5=det_mod.H5,cfg_file="mmWaveStudio/mobile_collection/rng40_res0004_tx3_rx4.lua",
  sync_csv=f"{RES}/sync_pairs.csv",sync_summary=f"{RES}/sync_summary.json",lidar_pcap=None,lidar_metadata=None,
  radar_bev_res=0.1,radar_percentile=99.9)
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
print(f"saved {HERE}/rl_env_sample.png")
