import sys, csv, os
import numpy as np, open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT); sys.path.insert(0,'sync/src')
from lib.lidar_io import (open_pcap_scan_source, sensor_info_from_source,
                          scan_range_flat, scan_to_xyz, _scan_from_scan_set)
PCAP="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11/lidar/20260511_2137_35_OS-0-128_122519000479.pcap"
META="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11/lidar/20260511_2137_35_OS-0-128_122519000479.json"
CSV="sync/res/2026.05.11/21.37.07/camera_sync_pairs.csv"
OUT="sync/res/2026.05.11/21.37.07/fusion_step3_dynamic_v2.png"
P0,NP=14090,45; VOX=0.3; NEAR=20.0     # short segment, near range

rows={}
for r in csv.DictReader(open(CSV)):
    try: rows[int(r['pair_idx'])]=int(r['lidar_idx'])
    except: pass
pairs=[p for p in range(P0,P0+NP) if p in rows]; lidx=[rows[p] for p in pairs]; N=len(pairs)
print(f"{N} consecutive pairs [{pairs[0]}..{pairs[-1]}]", flush=True)
src=open_pcap_scan_source(PCAP,META,index=True); meta=sensor_info_from_source(src)
def load(i):
    s=_scan_from_scan_set(src[int(i)]); xyz=scan_to_xyz(meta,s).reshape(-1,3); rng=scan_range_flat(s).reshape(-1)
    return xyz[np.isfinite(xyz).all(1)&(rng>=500)]
def mkpcd(a):
    p=o3d.geometry.PointCloud(); p.points=o3d.utility.Vector3dVector(a); p=p.voxel_down_sample(0.25)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.75,max_nn=30)); return p

raw=[load(li) for li in lidx]
# odometry
poses=[np.eye(4)]; prev=mkpcd(raw[0])
for i in range(1,N):
    cur=mkpcd(raw[i])
    reg=o3d.pipelines.registration.registration_icp(cur,prev,0.6,np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
    poses.append(poses[-1]@reg.transformation); prev=cur

# ground level (from frame 0, near range)
a0=raw[0]; gz=np.percentile(a0[np.linalg.norm(a0[:,:2],axis=1)<NEAR][:,2],3)
print(f"ground z ~ {gz:.2f} m; keeping [{gz+0.3:.2f}, {gz+2.2:.2f}]", flush=True)

W=[]; F=[]
for i,a in enumerate(raw):
    a=a[(np.linalg.norm(a[:,:2],axis=1)<NEAR)&(a[:,2]>gz+0.3)&(a[:,2]<gz+2.2)]  # near + above-ground slab
    st=max(1,len(a)//5000); a=a[::st]
    h=np.c_[a,np.ones(len(a))]; W.append((poses[i]@h.T).T[:,:3]); F.append(np.full(len(a),i))
W=np.vstack(W); F=np.concatenate(F)

key=np.floor(W/VOX).astype(np.int64)
vkey,inv=np.unique(key,axis=0,return_inverse=True)
from collections import defaultdict
vf=defaultdict(set)
for vid,f in zip(inv,F): vf[int(vid)].add(int(f))
occv=np.array([len(vf[i]) for i in range(len(vkey))]); occ=occv[inv]
static=occ>=0.50*N; dyn=(occ>=1)&(occ<=0.15*N)
print(f"points {len(W)}  static {static.sum()}  dynamic-cand {dyn.sum()} ({100*dyn.sum()//len(W)}%)", flush=True)

Wd=W[dyn]; Fd=F[dyn]; movers=[]
if len(Wd)>20:
    lab=DBSCAN(eps=0.6,min_samples=6).fit_predict(Wd[:,:2])
    for c in set(lab):
        if c==-1: continue
        m=lab==c; n=int(m.sum())
        if not (20<=n<=1500): continue            # object-sized
        pts=Wd[m]; fr=Fd[m]
        cents={ff:pts[fr==ff,:2].mean(0) for ff in sorted(set(fr))}
        if len(cents)<5: continue
        C=np.array(list(cents.values())); span=float(np.linalg.norm(C.max(0)-C.min(0)))
        # per-frame footprint (object stays compact) and net displacement
        fp=np.median([np.ptp(pts[fr==ff,:2],axis=0).max() for ff in set(fr)])
        net=float(np.linalg.norm(C[-1]-C[0]))
        if span>=1.5 and fp<3.0:                    # moved + object-sized footprint
            movers.append(dict(mask=m,n=n,span=span,net=net,nfr=len(cents),cents=C,fp=fp))
movers.sort(key=lambda d:-d['net'])
print(f"mover clusters: {len(movers)}", flush=True)
for mv in movers[:6]:
    print(f"  {mv['n']} pts, footprint {mv['fp']:.1f}m, net displacement {mv['net']:.1f}m (span {mv['span']:.1f}m) over {mv['nfr']} frames", flush=True)

fig,axs=plt.subplots(1,3,figsize=(22,7.5))
axs[0].scatter(W[static,1],W[static,0],s=0.3,c='0.7'); axs[0].scatter(Wd[:,1],Wd[:,0],s=2,c='red')
axs[0].set_title(f'Static (gray) vs dynamic (red)\n{static.sum()} static / {dyn.sum()} dynamic')
sc=axs[1].scatter(Wd[:,1],Wd[:,0],s=3,c=Fd,cmap='turbo'); plt.colorbar(sc,ax=axs[1],label='frame')
axs[1].set_title('Dynamic pts by frame (mover = trail)')
axs[2].scatter(W[static,1],W[static,0],s=0.3,c='0.85')
if movers:
    mv=movers[0]; pts=Wd[mv['mask']]; fr=Fd[mv['mask']]
    axs[2].scatter(pts[:,1],pts[:,0],s=10,c=fr,cmap='turbo')
    C=mv['cents']; axs[2].plot(C[:,1],C[:,0],'k-o',ms=3,lw=1.2)
    cx,cy=C.mean(0); axs[2].set_xlim(cy+6,cy-6); axs[2].set_ylim(cx-6,cx+6)
    axs[2].set_title(f"Extracted mover: {mv['n']} pts, {mv['fp']:.1f}m footprint\nmoved {mv['net']:.1f}m -> radar would confirm")
else: axs[2].set_title("no clean mover")
for ax in axs: ax.set_aspect('equal'); ax.set_xlabel('world Y (m)'); ax.set_ylabel('world X (m)'); ax.invert_xaxis()
plt.tight_layout(); plt.savefig(OUT,dpi=110,bbox_inches='tight'); print("saved",OUT,flush=True)
