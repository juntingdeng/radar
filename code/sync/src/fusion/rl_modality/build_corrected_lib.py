"""Rebuild the smoke layer library from the CORRECTLY-SYNCED smoke1_layers.npz (built with camera_sync_pairs.csv,
carries lidar_valid/radar_valid). Only lidar_valid & radar_valid frames -> same-instant camera+lidar+radar triplets.
Replaces the stale smoke_library_multisensor.npz (built from the old span_map manual pairing w the dropout mispair).
Output structure matches what models.py expects (lib[*] with alpha/airlight/backscatter/bev_smoke/bev_w/density/
cam_ok + shared clean_cam/clean_lid/clean_bev/A/la/fa/R). Radar stored as the measured near-field DELTA in bev_smoke
with clean_bev=0 so models.py's d=bev_smoke-clean_bev gives the delta directly."""
import os, numpy as np, cv2
SCR=next((p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)),os.path.expanduser("~/.cache/ti_setup_frames/rl_env"))  # persistent local first; session scratchpad fallback
V=np.load(f"{SCR}/smoke1_layers.npz")
el=V["el"]; lv=V["lidar_valid"]; rv=V["radar_valid"]; near=V["lidar_near"]
la,fa=V["la"],V["fa"]; f0,f1,l0,l1=[int(x) for x in V["crop"]]
mref=V["clean_bev_median_crop"].astype(np.float32)             # 10.15 clean radar median ref (for the delta)
cand=[k for k in range(len(el)) if 1770.4<=el[k]<=1776 and lv[k] and rv[k]]
sel=[cand[i] for i in np.linspace(0,len(cand)-1,16).astype(int)]   # 16 valid frames spanning the event
print("corrected library layers (el, cam_alpha, lidar_near):")
LIB=[]
for k in sel:
    bs=V["bs_pts"][int(V["bs_off"][k]):int(V["bs_off"][k+1])].astype(np.float16)
    sm=np.zeros((len(fa),len(la)),np.float32); sm[f0:f1,l0:l1]=V["bev_smoke_crop"][k].astype(np.float32)-mref   # DELTA
    w =np.zeros((len(fa),len(la)),np.float32); w[f0:f1,l0:l1]=V["bev_w_crop"][k].astype(np.float32)
    air=cv2.resize(V["airlight_ds4"][k].astype(np.float32),(640,480))
    LIB.append(dict(el=float(el[k]),alpha=(V["alpha"][k].astype(np.float32)/255),airlight=air,
                    backscatter=bs,bev_smoke=np.nan_to_num(sm),bev_w=np.nan_to_num(w),
                    density=int(near[k]),cam_ok=True))
    print(f"  el {el[k]:.1f}  cam_a {float(V['density'][k]):.2f}  near {int(near[k])}")
np.savez_compressed(f"{SCR}/smoke_library_corrected.npz",lib=np.array(LIB,dtype=object),
    clean_cam=V["B"],                                  # daytime clean background (for Blum airlight scaling)
    clean_lid=V["clean_lid"],clean_bev=np.zeros((len(fa),len(la)),np.float32),   # 0 -> models.py d = bev_smoke(delta)
    A=V["A"],la=la,fa=fa,R=V["R"])
print(f"saved {SCR}/smoke_library_corrected.npz ({len(LIB)} valid same-instant layers; stale 16-layer lib retired)")
