import sys, csv
import numpy as np
import open3d as o3d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
import os; os.chdir(ROOT); sys.path.insert(0,'sync/src')
from lib.lidar_io import (open_pcap_scan_source, sensor_info_from_source,
                          scan_range_flat, scan_to_xyz, _scan_from_scan_set)

PCAP="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11/lidar/20260511_2137_35_OS-0-128_122519000479.pcap"
META="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11/lidar/20260511_2137_35_OS-0-128_122519000479.json"
CSV="sync/res/2026.05.11/21.37.07/camera_sync_pairs.csv"
OUT="sync/res/2026.05.11/21.37.07/fusion_step1_odometry.png"
P0,NP=21000,120          # consecutive pairs
VOX=0.25                 # ICP voxel size (m)

rows={}
for r in csv.DictReader(open(CSV)):
    try: rows[int(r['pair_idx'])]=int(r['lidar_idx'])
    except: pass
pairs=[p for p in range(P0,P0+NP) if p in rows]
lidx=[rows[p] for p in pairs]
print(f"{len(pairs)} consecutive pairs [{pairs[0]}..{pairs[-1]}], lidar_idx {lidx[0]}..{lidx[-1]}", flush=True)

src=open_pcap_scan_source(PCAP,META,index=True)
meta=sensor_info_from_source(src)

def load(i):
    scan=_scan_from_scan_set(src[int(i)])
    xyz=scan_to_xyz(meta,scan).reshape(-1,3); rng=scan_range_flat(scan).reshape(-1)
    v=np.isfinite(xyz).all(1)&(rng>=500)
    return xyz[v]

def make_pcd(a):
    p=o3d.geometry.PointCloud(); p.points=o3d.utility.Vector3dVector(a)
    p=p.voxel_down_sample(VOX)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*3,max_nn=30))
    return p

# load once: keep downsampled pcd (for ICP) + plotting subsample
pcds=[]; subs=[]
for j,li in enumerate(lidx):
    a=load(li)
    pcds.append(make_pcd(a))
    step=max(1,len(a)//3000); subs.append(a[::step])
    if j%30==0: print(f"  loaded {j}/{len(lidx)}", flush=True)

# scan-to-scan ICP odometry
poses=[np.eye(4)]; fit=[]
for i in range(1,len(pcds)):
    reg=o3d.pipelines.registration.registration_icp(
        pcds[i],pcds[i-1],0.6,np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    poses.append(poses[-1]@reg.transformation)   # cur->prev accumulated -> world(=frame0)
    fit.append(reg.fitness)

# accumulate world cloud
W=[]; FID=[]
for i,a in enumerate(subs):
    h=np.c_[a,np.ones(len(a))]; w=(poses[i]@h.T).T[:,:3]
    W.append(w); FID.append(np.full(len(w),i))
W=np.vstack(W); FID=np.concatenate(FID)
traj=np.array([p[:3,3] for p in poses])
tlen=float(np.sum(np.linalg.norm(np.diff(traj,axis=0),axis=1)))

fig,axs=plt.subplots(1,3,figsize=(22,7))
axs[0].scatter(W[:,1],W[:,0],s=0.2,c='0.55'); axs[0].set_title('Accumulated world cloud (BEV)\nsharp=good odometry')
sc=axs[1].scatter(W[:,1],W[:,0],s=0.2,c=FID,cmap='turbo'); axs[1].set_title('Colored by frame (time)\nmovers = colored trails')
plt.colorbar(sc,ax=axs[1],label='frame')
axs[2].plot(traj[:,1],traj[:,0],'-',lw=1); axs[2].scatter(traj[0,1],traj[0,0],c='g',label='start'); axs[2].scatter(traj[-1,1],traj[-1,0],c='r',label='end')
axs[2].set_title(f'Ego trajectory: {tlen:.1f} m over {len(pairs)} frames'); axs[2].legend()
for ax in axs: ax.set_aspect('equal'); ax.set_xlabel('world Y (m)'); ax.set_ylabel('world X (m)'); ax.invert_xaxis()
plt.tight_layout(); plt.savefig(OUT,dpi=110,bbox_inches='tight')
print(f"\nmedian ICP fitness: {np.median(fit):.3f}  (higher=better overlap)", flush=True)
print(f"ego trajectory length: {tlen:.1f} m", flush=True)
print(f"saved {OUT}", flush=True)
