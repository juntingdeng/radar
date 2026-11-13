import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0,"sync/src/camera_projection"); sys.path.insert(0,"sync/src/fusion/radar_first")
from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
from common import load_pairs, load_calib, LidarSource, LOCAL, RES
CACHE=os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM=f"{LOCAL}/camera/20260511_213724.bag"; CIDX=f"{RES}/camera_index.npz"
F=int(sys.argv[1])
pairs=load_pairs(); cal=load_calib(); ci=cal["color_intrinsics"]
C2L=np.array(cal["camera_to_lidar_auto"]); L2C=np.linalg.inv(C2L)
fx,fy,cx,cy=ci["fx"],ci["fy"],ci["cx"],ci["cy"]; Hh,Ww=ci["height"],ci["width"]
def proj(Xl):
    Pc=(L2C@np.c_[Xl,np.ones(len(Xl))].T).T[:,:3]; m=Pc[:,2]>0.3; Pc=Pc[m]
    return fx*Pc[:,0]/Pc[:,2]+cx, fy*Pc[:,1]/Pc[:,2]+cy, m
bag_ts=np.load(CIDX)["color_bag_ts_ns"]
img=_extract_frames(CAM,DEFAULT_COLOR_TOPIC,[pairs[F]["color_idx"]],Path(CACHE),"color",bag_ts).get(pairs[F]["color_idx"])
if img is not None and img.ndim==1: img=img.reshape(Hh,Ww,3)
src=LidarSource(); P=src.xyz(pairs[F]["lidar_idx"]); P=P[np.linalg.norm(P[:,:2],axis=1)<30]
b=[x for x in json.load(open(f"{RES}/annotations/yolo_full_stride1.json"))["objects"][str(F)] if x.get("label")=="car" and x.get("box3d")][0]
b3=b["box3d"]; cxw,cyw,czw=b3["cx"],b3["cy"],b3["cz"]; L,W,Hc,yaw=b3["length"],b3["width"],b3["height"],b3.get("yaw",0)
# box3d corners in lidar frame
dx=np.array([L/2,L/2,-L/2,-L/2]*2); dy=np.array([W/2,-W/2,-W/2,W/2]*2); dz=np.array([-Hc/2]*4+[Hc/2]*4)
cs,sn=np.cos(yaw),np.sin(yaw)
cor=np.c_[cxw+dx*cs-dy*sn, cyw+dx*sn+dy*cs, czw+dz]
# lidar points inside the box footprint (the car cluster)
Xl=P[:,:2]-[cxw,cyw]; pr=Xl@np.array([[cs,sn],[-sn,cs]]).T
incar=(np.abs(pr[:,0])<L/2+0.3)&(np.abs(pr[:,1])<W/2+0.3)&(np.abs(P[:,2]-czw)<Hc/2+0.3)
u,v,m=proj(P); dep=np.hypot(P[m,0],P[m,1])
uc_,vc_,mc=proj(P[incar])
fig,ax=plt.subplots(figsize=(13,9)); ax.imshow(img)
ax.scatter(u,v,c=dep,s=5,cmap="jet",alpha=0.35,vmin=0,vmax=30)
ax.scatter(uc_,vc_,c="magenta",s=16,label="lidar points INSIDE box3d (the car)")
x0,y0,x1,y1=b["bbox_xyxy"]; ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec="white",lw=2,label="YOLO 2D bbox"))
ue,ve,me=proj(cor); E=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
uu=fx*((L2C@np.c_[cor,np.ones(8)].T).T[:,0])/((L2C@np.c_[cor,np.ones(8)].T).T[:,2])+cx
vv=fy*((L2C@np.c_[cor,np.ones(8)].T).T[:,1])/((L2C@np.c_[cor,np.ones(8)].T).T[:,2])+cy
for a,bb in E: ax.plot([uu[a],uu[bb]],[vv[a],vv[bb]],"lime",lw=1.5)
ax.set_xlim(0,Ww); ax.set_ylim(Hh,0); ax.legend(loc="upper right")
ax.set_title(f"pair {F}: car box3d (green) + its lidar points (magenta) vs YOLO bbox (white).\nMagenta on the car body & green enclosing it => calib azimuth OK")
cen_mag=(uc_.mean(),vc_.mean()); cen_yolo=(0.5*(x0+x1),0.5*(y0+y1))
print(f"car lidar-centroid proj = ({cen_mag[0]:.0f},{cen_mag[1]:.0f}) px ; YOLO bbox center = ({cen_yolo[0]:.0f},{cen_yolo[1]:.0f}) px")
print(f"offset du={cen_mag[0]-cen_yolo[0]:+.0f}px dv={cen_mag[1]-cen_yolo[1]:+.0f}px  -> azimuth {np.degrees(np.arctan((cen_mag[0]-cen_yolo[0])/fx)):+.2f}deg")
out=f"/tmp/box3d_{F}.png"; plt.savefig(out,dpi=95,bbox_inches="tight"); print("saved",out)
