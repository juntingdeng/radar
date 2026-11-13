"""Auto-label the STATIC parked-car objects of the 10.15 scene on the clean reference (one-time).
Stationary rig + static objects => each object has a fixed image bbox + fixed lidar BEV box + radar footprint;
per-frame detection outcomes against these are then real measurements. Cross-confirmed: YOLO image box (az-matched)
<-> lidar cluster (8-22m) <-> radar strong cells. Writes objects.json next to this script."""
import sys, os, json
import numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__))
REPO=os.path.abspath(os.path.join(HERE,"..","..","..",".."))
SCR=[p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)][0]
L=np.load(f"{SCR}/smoke_library_multisensor.npz",allow_pickle=True)
cam=L["clean_cam"].astype(np.uint8); lid=L["clean_lid"].astype(float); bev=L["clean_bev"].astype(float)
la,fa=L["la"],L["fa"]; fx,cx=float(L["fx"]),float(L["cx"])
# second, time-independent clean reference (fog window, ~3min earlier): STATIC objects must persist in BOTH;
# the 1786.8 frame's moving SUV (and any transient) drops out by data.
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

# lidar clusters 8-26m in the camera FOV: grid-based connected components
m=(lid[:,0]>7)&(lid[:,0]<26)&(np.abs(lid[:,1])<16)&(lid[:,2]>-0.5)&(lid[:,2]<2.2)
P=lid[m]
gx=((P[:,0]-7)/0.4).astype(int); gy=((P[:,1]+16)/0.4).astype(int)
H=np.zeros((int(19/0.4)+2,int(32/0.4)+2),int); np.add.at(H,(gx,gy),1)
occ=(H>=3).astype(np.uint8)
n,labm=cv2.connectedComponents(occ,8)
objs=[]
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
# viz
vis=cam.copy()
for o in objs:
    x0,y0,x1,y1=[int(v) for v in o["img_box"]]
    cv2.rectangle(vis,(x0,y0),(x1,y1),(0,255,0),2); cv2.putText(vis,o["obj_id"],(x0,y0-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
cv2.imwrite(f"{HERE}/objects_clean.png",vis[...,::-1])
print(f"saved {HERE}/objects.json + objects_clean.png ({len(objs)} objects)")
