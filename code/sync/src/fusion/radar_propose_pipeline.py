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
CSV=f"{RES}/camera_sync_pairs.csv"; OUT=f"{RES}/fusion_step12_radar_redo2.png"
P0,NP=13850,60; SPPF=45
cal=json.load(open(f"{RES}/calibration.json")); R2L=np.array(cal['radar_to_lidar_auto'])
rows={int(r['pair_idx']):r for r in csv.DictReader(open(CSV))}
pairs=[p for p in range(P0,P0+NP) if p in rows]; N=len(pairs)
t=np.array([float(rows[p]['radar_t']) for p in pairs]); t-=t[0]
radar=radarConfig(); radar.parse_radar(cfg_file=CFG); rres=radar.range_resolution
sv=radar.steering_vectors[1]; angs=np.linspace(-90,90,sv.shape[0]); proc=resolve_radar_processing('ti')
src=open_pcap_scan_source(PCAP,META,index=True); meta=sensor_info_from_source(src)

# ---- ODOMETRY (ICP) ----
def loadfull(p):
    s=_scan_from_scan_set(src[int(rows[p]['lidar_idx'])]); xyz=scan_to_xyz(meta,s).reshape(-1,3); rr=scan_range_flat(s).reshape(-1)
    return xyz[np.isfinite(xyz).all(1)&(rr>=500)]
def mkpcd(a):
    q=o3d.geometry.PointCloud(); q.points=o3d.utility.Vector3dVector(a); q=q.voxel_down_sample(0.25)
    q.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.75,max_nn=30)); return q
raw=[loadfull(p) for p in pairs]; poses=[np.eye(4)]; prev=mkpcd(raw[0])
for i in range(1,N):
    cur=mkpcd(raw[i]); reg=o3d.pipelines.registration.registration_icp(cur,prev,0.6,np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
    poses.append(poses[-1]@reg.transformation); prev=cur
tlen=float(np.sum([np.linalg.norm(poses[i][:3,3]-poses[i-1][:3,3]) for i in range(1,N)]))
print(f"ego moved {tlen:.1f} m over segment (why identity was wrong)",flush=True)
def toworld(xy,i,z=0.0):
    h=np.c_[xy[:,0],xy[:,1],np.full(len(xy),z),np.ones(len(xy))]; return (poses[i]@h.T).T[:,:2]

# ---- lidar DYNAMIC (WORLD frame, proper) ----
WL=[]
for i,a in enumerate(raw):
    gz=np.percentile(a[np.linalg.norm(a[:,:2],axis=1)<25][:,2],3)
    a=a[(np.linalg.norm(a[:,:2],axis=1)<25)&(a[:,2]>gz+0.3)&(a[:,2]<gz+2.2)]
    st=max(1,len(a)//6000); a=a[::st]; w=toworld(a[:,:2],i); WL.append(np.c_[w,np.full(len(w),i)])
WL=np.vstack(WL); k=np.floor(WL[:,:2]/0.3).astype(np.int64); vk,iv=np.unique(k,axis=0,return_inverse=True)
vf=defaultdict(set); [vf[int(v)].add(int(f)) for v,f in zip(iv,WL[:,2])]
occ=np.array([len(vf[i]) for i in range(len(vk))])[iv]; dynm=occ<=0.15*N
print(f"lidar dynamic now {100*dynm.sum()//len(WL)}% (was ~100% w/o odometry)",flush=True)
LDYN=[WL[(WL[:,2]==i)&dynm][:,:2] for i in range(N)]

# ---- radar detect -> WORLD ----
def detect(pid,i):
    ridx=int(rows[pid]['radar_idx']); adc,_=load_radar_adc_frame_at_packet_row(H5,sync_radar_idx_to_packet_row(ridx,SPPF),radar)
    rd=_adc_to_range_doppler(adc.astype(np.float64),proc); rdp=10*np.log10(np.sum(np.abs(rd)**2,axis=1)+1e-9); maxR=int(25/rres)
    det,_,_=cfar(2,10,1e-4,rdp[:maxR]); ri,di=np.where(det>0); rng=ri*rres; pw=rdp[ri,di]
    az=np.array([angs[np.argmax(np.abs(sv@rd[r,:,d])**2)] for r,d in zip(ri,di)]) if len(ri) else np.array([])
    kk=(rng>=2)&(rng<=25)&(pw>np.percentile(rdp,99.6)); rng,az=rng[kk],az[kk]
    azr=np.radians(-az); X=rng*np.cos(azr); Y=rng*np.sin(azr)
    lf=R2L[0,0]*X+R2L[0,1]*Y+R2L[0,3]; ll=R2L[1,0]*X+R2L[1,1]*Y+R2L[1,3]
    if not len(lf): return np.zeros((0,2))
    L=DBSCAN(eps=1.2,min_samples=1).fit_predict(np.c_[lf,ll]); out=np.array([[lf[L==c].mean(),ll[L==c].mean()] for c in set(L)])
    return toworld(out,i)                                       # -> WORLD frame
frames=[detect(p,i) for i,p in enumerate(pairs)]

# ---- track in WORLD (ego-compensated) ----
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

# ---- filter: WORLD-speed (ego-compensated) + straightness + dynamic-lidar ----
props=[]
for tr in tracks:
    P=np.array(tr['pts'])
    if len(P)<6: continue
    tt=t[P[:,0].astype(int)]; xy=P[:,1:3]
    vx=np.polyfit(tt,xy[:,0],1)[0]; vy=np.polyfit(tt,xy[:,1],1)[0]; spd=np.hypot(vx,vy)
    net=np.linalg.norm(xy[-1]-xy[0]); path=np.sum(np.linalg.norm(np.diff(xy,axis=0),axis=1)); straight=net/max(path,1e-6)
    if spd<1.5 or net<2.5: continue                             # world-speed: static (ego-comp) ~0 -> removed
    sup=0
    for f,x,y in P:
        D=LDYN[int(f)]
        if len(D) and np.count_nonzero(np.hypot(D[:,0]-x,D[:,1]-y)<=2.0)>=2: sup+=1
    props.append(dict(P=P,spd=spd,n=len(P),straight=straight,dyn=sup/len(P),
                      ok=(straight>=0.75 and sup/len(P)>=0.4)))
props.sort(key=lambda d:-d['spd']); conf=[p for p in props if p['ok']]
print(f"\nradar movers (world-speed>=1.5): {len(props)} -> CONFIRMED (straight>=0.75 & dyn-lidar>=40%): {len(conf)}",flush=True)
for p in props: print(f"  {p['spd']:.1f} m/s world, {p['n']}fr, straight={p['straight']:.2f}, dyn-lidar={p['dyn']*100:.0f}% -> {'CONFIRM' if p['ok'] else 'reject'}",flush=True)

SLw=WL[~dynm][:,:2]; DLw=WL[dynm][:,:2]
fig,axs=plt.subplots(1,2,figsize=(18,8))
axs[0].scatter(SLw[:,1],SLw[:,0],s=0.4,c='0.82',label='lidar static');
if len(DLw): axs[0].scatter(DLw[:,1],DLw[:,0],s=5,c='red',label='lidar DYNAMIC')
for p in props: axs[0].plot(p['P'][:,2],p['P'][:,1],'-o',ms=2,lw=1,color='orange',alpha=0.6)
axs[0].legend(); axs[0].set_title(f"World frame: {100*dynm.sum()//len(WL)}% lidar dynamic + {len(props)} radar movers")
axs[1].scatter(SLw[:,1],SLw[:,0],s=0.4,c='0.85')
if len(DLw): axs[1].scatter(DLw[:,1],DLw[:,0],s=5,c='salmon',alpha=0.5)
for p in props:
    P=p['P']; col='lime' if p['ok'] else 'gray'
    axs[1].plot(P[:,2],P[:,1],'-',lw=3 if p['ok'] else 1,color=col,alpha=0.95 if p['ok'] else 0.3)
    if p['ok']:
        axs[1].annotate('',xy=(P[-1,2],P[-1,1]),xytext=(P[0,2],P[0,1]),arrowprops=dict(arrowstyle='->',color='k',lw=2))
        axs[1].text(P[-1,2],P[-1,1]+0.5,f"{p['spd']:.1f} m/s",fontsize=11,fontweight='bold')
axs[1].plot([],[],'lime',lw=3,label=f'CONFIRMED mover ({len(conf)})'); axs[1].plot([],[],'gray',lw=1,label=f'rejected ({len(props)-len(conf)})')
axs[1].legend(); axs[1].set_title("Radar-proposed movers (ego-comp world speed + straight + dynamic-lidar)")
for ax in axs: ax.set_aspect('equal'); ax.set_xlabel('world Y (m)'); ax.set_ylabel('world X (m)'); ax.invert_xaxis()
plt.tight_layout(); plt.savefig(OUT,dpi=110,bbox_inches='tight'); print("saved",OUT,flush=True)
