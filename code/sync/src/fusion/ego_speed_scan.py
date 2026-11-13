import sys, csv
import numpy as np, open3d as o3d, os
ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT); sys.path.insert(0,'sync/src')
from lib.lidar_io import (open_pcap_scan_source, sensor_info_from_source,
                          scan_range_flat, scan_to_xyz, _scan_from_scan_set)
PCAP="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11/lidar/20260511_2137_35_OS-0-128_122519000479.pcap"
META="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11/lidar/20260511_2137_35_OS-0-128_122519000479.json"
CSV="sync/res/2026.05.11/21.37.07/camera_sync_pairs.csv"
rows={}
for r in csv.DictReader(open(CSV)):
    try: rows[int(r['pair_idx'])]=int(r['lidar_idx'])
    except: pass
src=open_pcap_scan_source(PCAP,META,index=True); meta=sensor_info_from_source(src)
def pcd(i):
    s=_scan_from_scan_set(src[int(i)]); xyz=scan_to_xyz(meta,s).reshape(-1,3); rng=scan_range_flat(s).reshape(-1)
    a=xyz[np.isfinite(xyz).all(1)&(rng>=500)]
    p=o3d.geometry.PointCloud(); p.points=o3d.utility.Vector3dVector(a); p=p.voxel_down_sample(0.3)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.9,max_nn=30)); return p
print("start_pair  |  10-frame path (m)  =>  m/frame", flush=True)
best=None
for P0 in range(13500,29000,2500):
    pairs=[p for p in range(P0,P0+11) if p in rows]
    if len(pairs)<6: continue
    ps=[pcd(rows[p]) for p in pairs]; pose=np.eye(4); L=0; prev=ps[0]
    for i in range(1,len(ps)):
        reg=o3d.pipelines.registration.registration_icp(ps[i],prev,0.6,np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        L+=np.linalg.norm(reg.transformation[:3,3]); prev=ps[i]
    mpf=L/(len(ps)-1)
    print(f"  {P0:6d}    |   {L:5.1f}            =>  {mpf:.2f}", flush=True)
    if best is None or mpf>best[1]: best=(P0,mpf)
print(f"\nFASTEST segment start ~{best[0]} ({best[1]:.2f} m/frame)", flush=True)
