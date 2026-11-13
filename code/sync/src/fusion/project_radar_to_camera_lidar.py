import sys, csv, os, json
import numpy as np, open3d as o3d
from sklearn.cluster import DBSCAN
from collections import defaultdict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT); sys.path.insert(0,'.'); sys.path.insert(0,'sync/src')
from utils.parse_config import radarConfig
from utils.cfar import cfar
from lib.bev_render import (load_radar_adc_frame_at_packet_row, sync_radar_idx_to_packet_row,
                            _adc_to_range_doppler, resolve_radar_processing)
from lib.lidar_io import (open_pcap_scan_source, sensor_info_from_source,
                          scan_range_flat, scan_to_xyz, _scan_from_scan_set)
LOCAL="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11"; RES="sync/res/2026.05.11/21.37.07"
H5=f"{LOCAL}/radar/2026.05.11-21.37.07.h5"; CFG=f"{ROOT}/mmWaveStudio/mobile_collection/rng40_res0004_tx3_rx4.lua"
PCAP=f"{LOCAL}/lidar/20260511_2137_35_OS-0-128_122519000479.pcap"; META=f"{LOCAL}/lidar/20260511_2137_35_OS-0-128_122519000479.json"
CSV=f"{RES}/camera_sync_pairs.csv"; OUT=f"{RES}/fusion_step13_project_cam.png"; P0,NP=13850,60; SPPF=45
cache=os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
cal=json.load(open(f"{RES}/calibration.json")); R2L=np.array(cal['radar_to_lidar_auto'])
C2L=np.array(cal['camera_to_lidar_auto']); L2C=np.linalg.inv(C2L); ci=cal['color_intrinsics']
rows={int(r['pair_idx']):r for r in csv.DictReader(open(CSV))}
pairs=[p for p in range(P0,P0+NP) if p in rows]; N=len(pairs)
t=np.array([float(rows[p]['radar_t']) for p in pairs]); t-=t[0]
radar=radarConfig(); radar.parse_radar(cfg_file=CFG); rres=radar.range_resolution
sv=radar.steering_vectors[1]; angs=np.linspace(-90,90,sv.shape[0]); proc=resolve_radar_processing('ti')
src=open_pcap_scan_source(PCAP,META,index=True); meta=sensor_info_from_source(src)
def loadfull(p):
    s=_scan_from_scan_set(src[int(rows[p]['lidar_idx'])]); xyz=scan_to_xyz(meta,s).reshape(-1,3); rr=scan_range_flat(s).reshape(-1)
    return xyz[np.isfinite(xyz).all(1)&(rr>=500)]
def mkpcd(a):
    q=o3d.geometry.PointCloud(); q.points=o3d.utility.Vector3dVector(a); q=q.voxel_down_sample(0.25)
    q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.75,max_nn=30)); return q
raw=[loadfull(p) for p in pairs]; poses=[np.eye(4)]; prev=mkpcd(raw[0])
for i in range(1,N):
    cur=mkpcd(raw[i]); reg=o3d.pipelines.registration.registration_icp(cur,prev,0.6,np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
    poses.append(poses[-1]@reg.transformation); prev=cur
inv_pose=[np.linalg.inv(p) for p in poses]
def toworld(xy,i): h=np.c_[xy[:,0],xy[:,1],np.zeros(len(xy)),np.ones(len(xy))]; return (poses[i]@h.T).T[:,:2]
def detect(pid,i):
    ridx=int(rows[pid]['radar_idx']); adc,_=load_radar_adc_frame_at_packet_row(H5,sync_radar_idx_to_packet_row(ridx,SPPF),radar)
    rd=_adc_to_range_doppler(adc.astype(np.float64),proc); rdp=10*np.log10(np.sum(np.abs(rd)**2,axis=1)+1e-9); maxR=int(25/rres)
    det,_,_=cfar(2,10,1e-4,rdp[:maxR]); ri,di=np.where(det>0); rng=ri*rres; pw=rdp[ri,di]
    az=np.array([angs[np.argmax(np.abs(sv@rd[r,:,d])**2)] for r,d in zip(ri,di)]) if len(ri) else np.array([])
    kk=(rng>=2)&(rng<=25)&(pw>np.percentile(rdp,99.6)); rng,az=rng[kk],az[kk]; azr=np.radians(-az); X=rng*np.cos(azr); Y=rng*np.sin(azr)  # FIX: azimuth sign (BEV fliplr)
    lf=R2L[0,0]*X+R2L[0,1]*Y+R2L[0,3]; ll=R2L[1,0]*X+R2L[1,1]*Y+R2L[1,3]
    if not len(lf): return np.zeros((0,2))
    L=DBSCAN(eps=1.2,min_samples=1).fit_predict(np.c_[lf,ll]); out=np.array([[lf[L==c].mean(),ll[L==c].mean()] for c in set(L)])
    return toworld(out,i)
frames=[detect(p,i) for i,p in enumerate(pairs)]
# lidar dynamic world
WL=[]
for i,a in enumerate(raw):
    gz=np.percentile(a[np.linalg.norm(a[:,:2],axis=1)<25][:,2],3); a=a[(np.linalg.norm(a[:,:2],axis=1)<25)&(a[:,2]>gz+0.3)&(a[:,2]<gz+2.2)]
    st=max(1,len(a)//6000); a=a[::st]; WL.append(np.c_[toworld(a[:,:2],i),np.full(len(a[::1]),i)][:len(a)])
WL=np.vstack(WL); k=np.floor(WL[:,:2]/0.3).astype(np.int64); vk,iv=np.unique(k,axis=0,return_inverse=True)
vf=defaultdict(set); [vf[int(v)].add(int(f)) for v,f in zip(iv,WL[:,2])]
occ=np.array([len(vf[i]) for i in range(len(vk))])[iv]; LDYN=[WL[(WL[:,2]==i)&(occ<=0.15*N)][:,:2] for i in range(N)]
# track
GATE=1.8; MISS=2; tracks=[]; active=[]
for f,dets in enumerate(frames):
    used=set()
    for tr in active:
        dt=t[f]-t[tr['lf']]; pred=np.array(tr['xy'])+np.array(tr['vel'])*dt
        if len(dets)==0: continue
        d=np.hypot(dets[:,0]-pred[0],dets[:,1]-pred[1]); j=int(np.argmin(d))
        if d[j]<=GATE and j not in used:
            used.add(j); prev=np.array(tr['xy']); tr['xy']=dets[j].tolist(); tr['lf']=f
            if dt>0: tr['vel']=((dets[j]-prev)/dt*0.5+np.array(tr['vel'])*0.5).tolist()
            tr['pts'].append((f,*dets[j]))
    for j,d in enumerate(dets):
        if j not in used: tracks.append({'xy':d.tolist(),'lf':f,'vel':[0,0],'pts':[(f,*d)]}); active.append(tracks[-1])
    active=[tr for tr in active if f-tr['lf']<=MISS]
conf=[]
for tr in tracks:
    P=np.array(tr['pts'])
    if len(P)<6: continue
    tt=t[P[:,0].astype(int)]; xy=P[:,1:3]; vx=np.polyfit(tt,xy[:,0],1)[0]; vy=np.polyfit(tt,xy[:,1],1)[0]; spd=np.hypot(vx,vy)
    net=np.linalg.norm(xy[-1]-xy[0]); path=np.sum(np.linalg.norm(np.diff(xy,axis=0),axis=1)); straight=net/max(path,1e-6)
    if spd<1.5 or net<2.5 or straight<0.75: continue
    sup=sum(1 for f,x,y in P if len(LDYN[int(f)]) and np.count_nonzero(np.hypot(LDYN[int(f)][:,0]-x,LDYN[int(f)][:,1]-y)<=2.0)>=2)
    if sup/len(P)>=0.4: conf.append(dict(P=P,spd=spd))
print(f"{len(conf)} confirmed movers",flush=True)
# pick cached frame with most confirmed movers present & in camera FOV
def fov(lf,ll): return lf>0.5 and abs(np.degrees(np.arctan2(ll,lf)))<28
best=None
for fi,p in enumerate(pairs):
    if not os.path.exists(f"{cache}/color_{int(rows[p]['camera_color_idx']):06d}.npy"): continue
    cnt=0
    for mv in conf:
        Pm=mv['P']; j=np.argmin(np.abs(Pm[:,0]-fi))
        if abs(Pm[j,0]-fi)>2: continue
        wl=(inv_pose[fi]@np.r_[Pm[j,1],Pm[j,2],0,1])[:2]
        if fov(*wl): cnt+=1
    if best is None or cnt>best[1]: best=(fi,cnt,p)
fi,_,FP=best; print(f"camera frame: pair {FP} (frame {fi})",flush=True)
Pl=loadfull(FP); img=np.load(f"{cache}/color_{int(rows[FP]['camera_color_idx']):06d}.npy")
if img.ndim==1: img=img.reshape(ci['height'],ci['width'],3)
fig,axs=plt.subplots(1,2,figsize=(19,7))
axs[0].imshow(img)
for mv in conf:
    Pm=mv['P']; j=np.argmin(np.abs(Pm[:,0]-fi))
    if abs(Pm[j,0]-fi)>2: continue
    wl=(inv_pose[fi]@np.r_[Pm[j,1],Pm[j,2],0,1])[:2]            # lidar frame @ FP
    d=np.hypot(Pl[:,0]-wl[0],Pl[:,1]-wl[1]); near=Pl[d<1.8]; z=np.median(near[:,2]) if len(near)>=3 else -0.5
    opt=(L2C@np.r_[wl,z,1])[:3]; Z=opt[2]
    if Z<=0.3: continue
    u=ci['fx']*opt[0]/Z+ci['cx']; v=ci['fy']*opt[1]/Z+ci['cy']
    axs[0].scatter([u],[v],s=300,facecolors='none',edgecolors='lime',linewidths=3)
    axs[0].text(u+8,v,f"{mv['spd']:.1f} m/s",color='lime',fontsize=12,fontweight='bold')
axs[0].set_xlim(0,ci['width']); axs[0].set_ylim(ci['height'],0); axs[0].axis('off')
axs[0].set_title(f"Radar-first movers projected on CAMERA (pair {FP})\ngreen circle = radar proposal (night, camera-weak)")
ml=np.linalg.norm(Pl[:,:2],axis=1)<25
axs[1].scatter(Pl[ml,1],Pl[ml,0],s=0.5,c='0.75')
for mv in conf:
    Pm=mv['P']; LL=np.array([(inv_pose[fi]@np.r_[x,y,0,1])[:2] for _,x,y in Pm])
    axs[1].plot(LL[:,1],LL[:,0],'-',color='lime',lw=2.5); axs[1].annotate('',xy=(LL[-1,1],LL[-1,0]),xytext=(LL[0,1],LL[0,0]),arrowprops=dict(arrowstyle='->',color='k',lw=2))
    axs[1].text(LL[-1,1],LL[-1,0]+0.5,f"{mv['spd']:.1f} m/s",fontsize=11,fontweight='bold')
axs[1].set_aspect('equal'); axs[1].set_xlim(15,-15); axs[1].set_ylim(0,25); axs[1].set_xlabel('lateral Y (m)'); axs[1].set_ylabel('forward X (m)')
axs[1].set_title("Lidar BEV + radar-first mover tracks (speed)")
plt.tight_layout(); plt.savefig(OUT,dpi=110,bbox_inches='tight'); print("saved",OUT,flush=True)
