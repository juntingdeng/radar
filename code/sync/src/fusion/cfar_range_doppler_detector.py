import sys, csv, os
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT); sys.path.insert(0,'.'); sys.path.insert(0,'sync/src')
from utils.parse_config import radarConfig
from utils.cfar import cfar
from lib.bev_render import (load_radar_adc_frame_at_packet_row, sync_radar_idx_to_packet_row,
                            _adc_to_range_doppler, resolve_radar_processing)
LOCAL="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11"; RES="sync/res/2026.05.11/21.37.07"
H5=f"{LOCAL}/radar/2026.05.11-21.37.07.h5"
CFG=f"{ROOT}/mmWaveStudio/mobile_collection/rng40_res0004_tx3_rx4.lua"
SYNC=f"{RES}/camera_sync_pairs.csv"; OUT=f"{RES}/fusion_step7_cfar_proto.png"
PAIR=13870; SYNC_PPF=45

radar=radarConfig(); radar.parse_radar(cfg_file=CFG)
rres=radar.range_resolution
# config's doppler_resolution is buggy (bandwidth Hz vs MHz) -> compute correctly:
lmbda=3e8/(radar.start_freq*1e9 + radar.bandwidth/2)      # center wavelength (m)
T_loop=radar.num_tx*radar.chirp_interval                  # time between doppler chirps (TDM)
print(f"range_res={rres:.4f} m  lambda={lmbda*1000:.2f} mm  T_loop={T_loop*1e6:.0f} us", flush=True)

ridx=None
for r in csv.DictReader(open(SYNC)):
    if int(r['pair_idx'])==PAIR: ridx=int(r['radar_idx']); break
prow=sync_radar_idx_to_packet_row(ridx, SYNC_PPF)
adc,aidx=load_radar_adc_frame_at_packet_row(H5, prow, radar)
print(f"pair {PAIR} -> radar_idx {ridx} -> adc frame {aidx}, adc shape {adc.shape}", flush=True)

proc=resolve_radar_processing('ti')     # clutter removal + windows + zero-Doppler notch (drops static)
rd=_adc_to_range_doppler(adc.astype(np.float64), proc)   # (n_range, n_ant, n_dop)
nR,nA,nD=rd.shape; print(f"range-doppler cube: {rd.shape}", flush=True)
dres=lmbda/(2*nD*T_loop)                                  # velocity resolution (m/s)
print(f"doppler_res={dres:.3f} m/s  v_max=+/-{nD//2*dres:.1f} m/s", flush=True)
rdp=np.sum(np.abs(rd)**2, axis=1)         # (n_range, n_dop) power
rdp_db=10*np.log10(rdp+1e-9)

# limit to usable range
maxR=int(25/rres)
det,_,_=cfar(2, 8, 1e-2, rdp_db[:maxR])   # 2D CA-CFAR
ri,di=np.where(det>0)
# velocity per doppler bin
vel=(di - nD//2)*dres
rng=ri*rres
# azimuth per detection (12-elem beamform)
sv=radar.steering_vectors[1]              # (n_angles, 12)
angs=np.linspace(-90,90,sv.shape[0])
az=np.array([angs[np.argmax(np.abs(sv@rd[r,:,d])**2)] for r,d in zip(ri,di)])
# keep MOVING, in-range, above floor
pw=rdp_db[ri,di]
keep=(np.abs(vel)>0.3)&(rng>=2)&(rng<=25)&(pw>np.percentile(rdp_db,99))
ri,di,vel,rng,az,pw=ri[keep],di[keep],vel[keep],rng[keep],az[keep],pw[keep]
# to BEV (radar frame): forward=range*cos(az), lateral=range*sin(az)
azr=np.radians(-az); X=rng*np.cos(azr); Y=rng*np.sin(azr)
print(f"\nCFAR detections (moving, in-range): {len(ri)}", flush=True)
# cluster nearby detections into targets
from sklearn.cluster import DBSCAN
tgt=[]
if len(X):
    L=DBSCAN(eps=1.5,min_samples=1).fit_predict(np.c_[X,Y])
    for c in set(L):
        m=L==c; tgt.append((X[m].mean(),Y[m].mean(),vel[m].mean(),m.sum()))
print(f"discrete radar targets: {len(tgt)}", flush=True)
for t in sorted(tgt,key=lambda z:-abs(z[2]))[:10]:
    print(f"  fwd {t[0]:5.1f} lat {t[1]:6.1f}  vel {t[2]:+.1f} m/s  ({t[3]} cells)", flush=True)

fig,axs=plt.subplots(1,2,figsize=(17,7))
ext=[-(nD//2)*dres,(nD//2)*dres,0,maxR*rres]
im=axs[0].imshow(rdp_db[:maxR],aspect='auto',origin='lower',extent=ext,cmap='viridis')
axs[0].scatter(vel,rng,s=40,facecolors='none',edgecolors='red',label='CFAR moving detections')
axs[0].set_xlabel('radial velocity (m/s)'); axs[0].set_ylabel('range (m)')
axs[0].set_title(f'Range-Doppler map + CFAR detections (pair {PAIR})\ndiscrete points, each with a VELOCITY'); axs[0].legend()
plt.colorbar(im,ax=axs[0],label='dB')
if len(tgt):
    T=np.array([(t[0],t[1],t[2]) for t in tgt])
    sc=axs[1].scatter(T[:,1],T[:,0],c=T[:,2],cmap='coolwarm',s=80,vmin=-3,vmax=3,edgecolors='k')
    plt.colorbar(sc,ax=axs[1],label='radial velocity (m/s)')
axs[1].set_aspect('equal'); axs[1].set_xlabel('lateral Y (m)'); axs[1].set_ylabel('forward X (m)'); axs[1].invert_xaxis()
axs[1].set_title(f'{len(tgt)} discrete radar targets in BEV\n(range+azimuth+velocity) vs the old arc flood')
plt.tight_layout(); plt.savefig(OUT,dpi=110,bbox_inches='tight'); print("saved",OUT,flush=True)
