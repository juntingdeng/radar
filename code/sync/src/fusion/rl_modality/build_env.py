"""Build the RL modality-selection environment table from MEASURED per-frame quantities (no invented data).

495 real frames (smoke1 298 + fog1 197, clean -> dense smoke -> fog). Per frame:
  VISIBILITY (the reward's ingredients; full-measurement, computed offline):
    vis_cam   = far-band sharpness ratio vs the clean background (Laplacian variance, rows 120..300)
                -- real smoke BLURS + washes the scene (the discriminator lesson); clipped 0..1
    vis_lidar = far-field survival: lidar_far / clean base_far (stored counts), clipped 0..1
    vis_radar = strong-cell recall vs the window's median clean radar reference (stored crops, r<16m)
                normalized by the clean-frame median recall, clipped 0..1
  PROBES (the agent's cheap state; must NOT include the full measurements):
    thumbnail mean brightness / contrast (16x12 gray), near-band Laplacian, lidar NEAR count (cheap range-image
    stat), frame-of-day index. The far-band measurements and radar quantities are NOT in the state -- the agent
    must learn their value from experience (that is the selection problem).
Writes env_table.npz next to this script.
"""
import os, sys, json
import numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__))
REPO=os.path.abspath(os.path.join(HERE,"..","..","..",".."))
SCR=next((p for p in [os.path.expanduser("~/.cache/ti_setup_frames/rl_env"),"/private/tmp/claude-501/-Users-juntingdeng-Library-CloudStorage-GoogleDrive-juntingd-andrew-cmu-edu-My-Drive-CMU-Research-3DImage-sensor-TI-setup-test-code/9ad422ab-c5ed-4939-b084-4a92ef459884/scratchpad"] if os.path.isdir(p)),os.path.expanduser("~/.cache/ti_setup_frames/rl_env"))  # persistent local first; session scratchpad fallback
FARBAND=slice(120,300)   # image rows holding the far scene (trees + road across)

rows=[]
for video in ["smoke1","fog1"]:
    src=f"{SCR}/{video}_layers.npz"
    if not os.path.exists(src): src=f"{REPO}/sync/res/2026.10.15/{video}_layers.npz"
    V=np.load(src)
    B=V["B"].astype(np.uint8)
    gB=cv2.cvtColor(B,cv2.COLOR_RGB2GRAY)
    lapB=float(cv2.Laplacian(gB[FARBAND],cv2.CV_32F).var())
    # radar: per-frame strong-cell recall vs the stored median clean reference (same machinery as the proof)
    MREF=V["clean_bev_median_crop"].astype(np.float32)
    f0,f1,l0,l1=[int(x) for x in V["crop"]]
    la,fa=V["la"],V["fa"]; RL,RF=np.meshgrid(la,fa); RNGc=np.hypot(RF,RL)[f0:f1,l0:l1]
    valid=(RNGc>2)&(RNGc<16)
    strong=valid&(MREF>np.percentile(MREF[valid],99))
    sm=V["bev_smoke_crop"].astype(np.float32)
    rec=np.array([float(np.mean(sm[k][strong]>=MREF[strong]-3.0)) for k in range(len(sm))])
    base_far=float(V["base_far"]); lidar_far=V["lidar_far"].astype(float)
    lidar_near=V["lidar_near"].astype(float)
    vsrc=f"{SCR}/{video}.mp4"
    if not os.path.exists(vsrc): vsrc=f"{REPO}/sync/res/2026.10.15/{video}.mp4"
    # sequential decode (seek-based reads glitch): layer frames are every STRIDE-th video frame
    cap=cv2.VideoCapture(vsrc)
    frames={}
    want=set(int(x) for x in V["frame_idx"]); i=0
    while True:
        ok,fr=cap.read()
        if not ok: break
        if i in want: frames[i]=fr
        i+=1
    assert len(frames)==len(want),(len(frames),len(want))
    N=len(V["frame_idx"])
    rec_clean=[]
    tmp=[]
    for k in range(N):
        fr=frames[int(V["frame_idx"][k])]
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        lap=float(cv2.Laplacian(g[FARBAND],cv2.CV_32F).var())
        th=cv2.resize(g,(16,12)).astype(float)
        tmp.append(dict(video=video,k=k,el=float(V["el"][k]),
            cam_density=float(V["density"][k]),
            lidar_valid=bool(V["lidar_valid"][k]),radar_valid=bool(V["radar_valid"][k]),
            vis_cam=lap/lapB,
            vis_lidar=min(1.0,lidar_far[k]/base_far) if V["lidar_valid"][k] and lidar_far[k]>=0 else np.nan,
            rec_radar=rec[k] if V["radar_valid"][k] else np.nan,
            probe_bright=float(th.mean()),probe_contrast=float(th.std()),
            probe_near_lap=float(cv2.Laplacian(g[320:,:],cv2.CV_32F).var()),
            probe_lidar_near=float(lidar_near[k]) if lidar_near[k]>=0 else np.nan))
        if V["density"][k]<0.05 and V["lidar_valid"][k] and lidar_near[k]<3200 and V["radar_valid"][k]:
            rec_clean.append(rec[k])
    recn=float(np.median(rec_clean))
    for r in tmp:
        r["vis_radar"]=min(1.0,r["rec_radar"]/recn) if np.isfinite(r["rec_radar"]) else np.nan
        r["vis_cam"]=min(1.0,r["vis_cam"])
    rows+=tmp
    print(f"{video}: {N} frames, clean radar recall {recn:.3f}, "
          f"vis_cam {np.nanmin([r['vis_cam'] for r in tmp]):.2f}-{np.nanmax([r['vis_cam'] for r in tmp]):.2f}, "
          f"vis_lidar min {np.nanmin([r['vis_lidar'] for r in tmp]):.2f}",flush=True)

keys=["cam_density","vis_cam","vis_lidar","vis_radar","probe_bright","probe_contrast","probe_near_lap","probe_lidar_near","el"]
out={k:np.array([r.get(k,np.nan) for r in rows],float) for k in keys}
out["video"]=np.array([r["video"] for r in rows]); out["k"]=np.array([r["k"] for r in rows])
out["lidar_valid"]=np.array([r["lidar_valid"] for r in rows]); out["radar_valid"]=np.array([r["radar_valid"] for r in rows])
np.savez(f"{HERE}/env_table.npz",**out,
    note="per-frame measured visibility (reward ingredients) + cheap probes (state). vis_* in 0..1; "
         "NaN vis = modality had no valid same-moment measurement (frame excluded from training).")
print(f"saved {HERE}/env_table.npz ({len(rows)} frames)")
