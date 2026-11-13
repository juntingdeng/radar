import sys, csv, os, json
import numpy as np
from types import SimpleNamespace
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT); sys.path.insert(0,'.'); sys.path.insert(0,'sync/src'); sys.path.insert(0,'sync/src/camera_projection')
from utils.parse_config import radarConfig
from utils.cfar import cfar
from lib.bev_render import (load_radar_adc_frame_at_packet_row, sync_radar_idx_to_packet_row,
                            _adc_to_range_doppler, resolve_radar_processing)
from lib.lidar_io import (open_pcap_scan_source, sensor_info_from_source, scan_range_flat, scan_to_xyz, _scan_from_scan_set)
from yolo_to_lidar import _radar_setup, _radar_panel, _peaks_from_bev
LOCAL="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11"; RES="sync/res/2026.05.11/21.37.07"
H5=f"{LOCAL}/radar/2026.05.11-21.37.07.h5"; CFG=f"{ROOT}/mmWaveStudio/mobile_collection/rng40_res0004_tx3_rx4.lua"
PCAP=f"{LOCAL}/lidar/20260511_2137_35_OS-0-128_122519000479.pcap"; META=f"{LOCAL}/lidar/20260511_2137_35_OS-0-128_122519000479.json"
CSV=f"{RES}/camera_sync_pairs.csv"; OUT=f"{RES}/fusion_diag_azimuth.png"; PAIR=13865; SPPF=45
cal=json.load(open(f"{RES}/calibration.json")); R2L=np.array(cal['radar_to_lidar_auto'])
row={int(r['pair_idx']):r for r in csv.DictReader(open(CSV))}[PAIR]
ridx,lidx=int(row['radar_idx']),int(row['lidar_idx'])
radar=radarConfig(); radar.parse_radar(cfg_file=CFG); rres=radar.range_resolution; sv=radar.steering_vectors[1]; angs=np.linspace(-90,90,sv.shape[0])

# lidar
src=open_pcap_scan_source(PCAP,META,index=True); meta=sensor_info_from_source(src)
s=_scan_from_scan_set(src[lidx]); xyz=scan_to_xyz(meta,s).reshape(-1,3); rr=scan_range_flat(s).reshape(-1)
Pl=xyz[np.isfinite(xyz).all(1)&(rr>=500)]; ml=np.linalg.norm(Pl[:,:2],axis=1)<25

# VALIDATED radar peaks (viz_labels chain, moving)
args=SimpleNamespace(radar_h5=H5,cfg_file=CFG,sync_csv=f"{RES}/sync_pairs.csv",sync_summary=f"{RES}/sync_summary.json",lidar_pcap=PCAP,lidar_metadata=META,radar_bev_res=0.1)
loader,R=_radar_setup(args,cal)
bev,la,fa=_radar_panel(loader,PAIR); vrf,vrl=_peaks_from_bev(bev,la,fa,R,99.0)   # lidar frame (fwd,lat)

# MY cfar detections, both azimuth conventions
adc,_=load_radar_adc_frame_at_packet_row(H5,sync_radar_idx_to_packet_row(ridx,SPPF),radar)
rd=_adc_to_range_doppler(adc.astype(np.float64),resolve_radar_processing('ti'))
rdp=10*np.log10(np.sum(np.abs(rd)**2,axis=1)+1e-9); maxR=int(25/rres)
det,_,_=cfar(2,10,1e-3,rdp[:maxR]); ri,di=np.where(det>0); rng=ri*rres; pw=rdp[ri,di]
az=np.array([angs[np.argmax(np.abs(sv@rd[r,:,d])**2)] for r,d in zip(ri,di)])
kk=(rng>=2)&(rng<=25)&(pw>np.percentile(rdp,99)); rng,az=rng[kk],az[kk]
def proj(az_use):
    azr=np.radians(az_use); X=rng*np.cos(azr); Y=rng*np.sin(azr)
    return R2L[0,0]*X+R2L[0,1]*Y+R2L[0,3], R2L[1,0]*X+R2L[1,1]*Y+R2L[1,3]
lf1,ll1=proj(az)       # as-is
lf2,ll2=proj(-az)      # flipped azimuth

fig,axs=plt.subplots(1,3,figsize=(22,7.5))
for ax in axs: ax.scatter(Pl[ml,1],Pl[ml,0],s=0.5,c='0.75')
axs[0].scatter(vrl,vrf,s=12,c='green',marker='x'); axs[0].set_title(f"VALIDATED radar peaks (viz_labels) on lidar (pair {PAIR})")
axs[1].scatter(ll1,lf1,s=40,facecolors='none',edgecolors='red'); axs[1].scatter(vrl,vrf,s=8,c='green',marker='x',alpha=0.4)
axs[1].set_title("MY CFAR az as-is (red) vs validated (green)")
axs[2].scatter(ll2,lf2,s=40,facecolors='none',edgecolors='blue'); axs[2].scatter(vrl,vrf,s=8,c='green',marker='x',alpha=0.4)
axs[2].set_title("MY CFAR az FLIPPED (blue) vs validated (green)")
for ax in axs: ax.set_aspect('equal'); ax.set_xlim(15,-15); ax.set_ylim(0,25); ax.set_xlabel('lateral Y (m)'); ax.set_ylabel('forward X (m)')
plt.tight_layout(); plt.savefig(OUT,dpi=110,bbox_inches='tight')
# quantify: which az matches validated peaks better
def nndist(lf,ll):
    if not len(vrf) or not len(lf): return np.nan
    return np.median([np.hypot(vrf-x,vrl-y).min() for x,y in zip(lf,ll)])
print(f"median NN dist to validated peaks: as-is={nndist(lf1,ll1):.1f}m  flipped={nndist(lf2,ll2):.1f}m",flush=True)
print(f"validated peaks: {len(vrf)}  my cfar: {len(rng)}",flush=True); print("saved",OUT,flush=True)
