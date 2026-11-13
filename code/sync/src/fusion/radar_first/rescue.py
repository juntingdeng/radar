"""Night mover rescue (merged). Subcommands: v4 (final) | raise | motion"""
import sys

# ======== src/fusion/radar_first/rescue_v4.py ========
# Night moving-car rescue with TEMPORAL STATIC-REJECTION.
# 
# Builds on rescue_v3 (ego-Doppler gate + raw-LiDAR snap). The remaining failure was
# static background that evades the ego-Doppler gate (spurious Doppler) and beats the car
# on density/power. Discriminator: a real moving car occupies a WORLD location only briefly;
# static clutter sits at the same world spot every frame. So:
#   - compute candidate moving-backed boxes for every frame in a window (ego-compensated via odometry)
#   - REJECT a target box if its world position is occupied by a candidate in many neighbor frames (static)
#   - keep transient (moving) boxes -> the car
# This is rejection via odometry, NOT the fragile per-cluster velocity tracking that failed.
#     python3 rescue_v4.py 14260 14270 14280 14290 14300
def v4_main():
    import os, sys, json
    import numpy as np
    from sklearn.cluster import DBSCAN
    from common import load_pairs, load_calib, LidarSource, RES, pose_apply
    from detector import RadarDetector
    from ego_doppler import fit_ego_velocity
    from odometry import odometry_scan2map
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import _ground_level
    TARGETS = [int(a) for a in sys.argv[1:]] or [14280]
    MARGIN = 4
    ROI = (0.0, 30.0, -16.0, 16.0); GPCT = 10.0; HBAND = (0.3, 2.2)
    ROI = (0.0, 30.0, -16.0, 16.0); GPCT = 10.0; HBAND = (0.3, 2.2)
    ROI = (0.0, 30.0, -16.0, 16.0); GPCT = 10.0; HBAND = (0.3, 2.2)
    MOVER_MARGIN = 0.8; RANGE_TOL = 1.5; AZ_TOL = 12.0; MIN_PTS = 12; MIN_EGO_INLIERS = 6
    MOVER_MARGIN = 0.8; RANGE_TOL = 1.5; AZ_TOL = 12.0; MIN_PTS = 12; MIN_EGO_INLIERS = 6
    MOVER_MARGIN = 0.8; RANGE_TOL = 1.5; AZ_TOL = 12.0; MIN_PTS = 12; MIN_EGO_INLIERS = 6
    MOVER_MARGIN = 0.8; RANGE_TOL = 1.5; AZ_TOL = 12.0; MIN_PTS = 12; MIN_EGO_INLIERS = 6
    MOVER_MARGIN = 0.8; RANGE_TOL = 1.5; AZ_TOL = 12.0; MIN_PTS = 12; MIN_EGO_INLIERS = 6
    PERSIST_R = 1.6            # world radius to call two candidates "the same spot"
    PERSIST_K = 3             # check only +/- this many neighbor frames (local window)
    PERSIST_MAX = 3           # occupied in > this many of the 2*K neighbors -> static -> reject
    pairs = load_pairs(); cal = load_calib()
    pairs = load_pairs(); cal = load_calib()
    ci = cal["color_intrinsics"]; L2C = np.linalg.inv(np.array(cal["camera_to_lidar_auto"]))
    ci = cal["color_intrinsics"]; L2C = np.linalg.inv(np.array(cal["camera_to_lidar_auto"]))
    R2L = np.array(cal["radar_to_lidar_auto"]); A = R2L[:2, :2]; t = R2L[:2, 3]; A_inv = np.linalg.inv(A)
    R2L = np.array(cal["radar_to_lidar_auto"]); A = R2L[:2, :2]; t = R2L[:2, 3]; A_inv = np.linalg.inv(A)
    R2L = np.array(cal["radar_to_lidar_auto"]); A = R2L[:2, :2]; t = R2L[:2, 3]; A_inv = np.linalg.inv(A)
    R2L = np.array(cal["radar_to_lidar_auto"]); A = R2L[:2, :2]; t = R2L[:2, 3]; A_inv = np.linalg.inv(A)
    det = RadarDetector(processing="legacy")
    W0, W1 = min(TARGETS) - MARGIN, max(TARGETS) + MARGIN
    ids = [p for p in range(W0, W1 + 1) if p in pairs]
    idx_of = {p: i for i, p in enumerate(ids)}
    src = LidarSource()
    scans = [src.xyz(pairs[p]["lidar_idx"]) for p in ids]
    poses, _ = odometry_scan2map(scans)
    def candidates(i):
        """Moving-radar-backed LiDAR clusters at frame index i -> boxes with world centroid."""
        P = scans[i]; T = ids[i]
        d, _ = det.detect(pairs[T]["radar_idx"])
        if len(d) < MIN_EGO_INLIERS: return []
        ab, inl = fit_ego_velocity(d["az"], d["vel"])
        if ab is None or inl.sum() < MIN_EGO_INLIERS: return []
        azr = np.radians(-d["az"]); pred = np.cos(azr)*ab[0] + np.sin(azr)*ab[1]
        d = d[(~inl) & (np.abs(d["vel"] - pred) > MOVER_MARGIN)]
        if not len(d): return []
        Xd, Yd = RadarDetector.to_xy(d); drng = np.hypot(Xd, Yd); dang = np.degrees(np.arctan2(Yd, Xd))
        x0, x1, y0, y1 = ROI
        reg = P[(P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)]
        gz = _ground_level(reg, GPCT)
        slab = reg[(reg[:, 2] > gz + HBAND[0]) & (reg[:, 2] < gz + HBAND[1])]
        rr = (A_inv @ (slab[:, :2] - t).T).T
        prng = np.hypot(rr[:, 0], rr[:, 1]); pang = np.degrees(np.arctan2(rr[:, 1], rr[:, 0]))
        backed = np.zeros(len(slab), bool)
        for dr, da in zip(drng, dang):
            backed |= (np.abs(prng - dr) < RANGE_TOL) & (np.abs(((pang - da + 180) % 360) - 180) < AZ_TOL)
        rp = slab[backed]; out = []
        if len(rp) >= MIN_PTS:
            lab = DBSCAN(eps=1.0, min_samples=6).fit_predict(rp[:, :2])
            for c in set(lab.tolist()):
                if c == -1: continue
                cl = rp[lab == c]
                if cl.shape[0] < MIN_PTS: continue
                cen = cl[:, :2].mean(0)
                if np.hypot(*cen) < 4.0: continue
                out.append(dict(pts=cl, cen=cen, cen_w=pose_apply(poses[i], cen[None, :])[0], n=int(cl.shape[0])))
        return out
    cand_per = [candidates(i) for i in range(len(ids))]
    def box_from(cl):
        xy = cl[:, :2]; cen = xy.mean(0); X = xy - cen
        _, _, vt = np.linalg.svd(X, full_matrices=False); yaw = float(np.arctan2(vt[0, 1], vt[0, 0]))
        proj = X @ vt.T; L = max(float(np.ptp(proj[:, 0])), 1.5); W = max(float(np.ptp(proj[:, 1])), 1.6)
        if L < 3.0: L = 4.3; yaw = float(np.arctan2(cen[1], cen[0]))
        z0 = float(cl[:, 2].min())
        return dict(cx=float(cen[0]), cy=float(cen[1]), yaw=yaw, L=min(L, 6.0), W=min(W, 2.2), z0=z0, z1=z0+1.5, n=int(cl.shape[0]))
    def corners(b):
        dx = np.array([b["L"]/2]*2 + [-b["L"]/2]*2); dy = np.array([b["W"]/2, -b["W"]/2, -b["W"]/2, b["W"]/2])
        cs, sn = np.cos(b["yaw"]), np.sin(b["yaw"]); fx = b["cx"]+dx*cs-dy*sn; fy = b["cy"]+dx*sn+dy*cs
        return np.c_[fx, fy, np.full(4, b["z0"])], np.c_[fx, fy, np.full(4, b["z1"])]
    def to_px(xyz):
        opt = (L2C @ np.r_[xyz, 1])[:3]
        return None if opt[2] <= 0.2 else [float(ci["fx"]*opt[0]/opt[2]+ci["cx"]), float(ci["fy"]*opt[1]/opt[2]+ci["cy"])]
    def persistence(cen_w, ti):
        """# of LOCAL neighbor frames (+/-PERSIST_K) with a candidate near this world spot.
        A moving car leaves the spot within a few frames (low); static clutter stays (high)."""
        n = 0
        for j in range(max(0, ti-PERSIST_K), min(len(ids), ti+PERSIST_K+1)):
            if j == ti: continue
            if any(np.hypot(c["cen_w"][0]-cen_w[0], c["cen_w"][1]-cen_w[1]) < PERSIST_R for c in cand_per[j]):
                n += 1
        return n
    allobj = {}
    for T in TARGETS:
        ti = idx_of[T]
        cands = cand_per[ti]
        kept = []
        for c in cands:
            per = persistence(c["cen_w"], ti)
            tag = "STATIC" if per > PERSIST_MAX else "moving"
            print(f"  pair {T} cand cen=({c['cen'][0]:5.1f},{c['cen'][1]:6.1f}) n{c['n']:3d} persist{per} -> {tag}", flush=True)
            if per <= PERSIST_MAX:
                kept.append(c)
        # NMS + density among moving survivors
        kept.sort(key=lambda c: -c["n"]); ded = []
        for c in kept:
            if any(np.hypot(c["cen"][0]-k["cen"][0], c["cen"][1]-k["cen"][1]) < 2.5 for k in ded): continue
            ded.append(c)
        if ded:
            nmax = max(c["n"] for c in ded); ded = [c for c in ded if c["n"] >= max(60, 0.4*nmax)]
        print(f"pair {T}: {len(cands)} cand -> {len(ded)} boxes after static-rejection+density", flush=True)
        lst = []
        for c in ded:
            b = box_from(c["pts"]); bo, tp = corners(b); px = [to_px(x) for x in list(bo)+list(tp)]; px = [p for p in px if p]
            bbox = None
            if len(px) >= 4:
                a = np.array(px); bbox = [float(a[:,0].min()), float(a[:,1].min()), float(a[:,0].max()), float(a[:,1].max())]
            box3d = dict(cx=b["cx"], cy=b["cy"], cz=(b["z0"]+b["z1"])/2, yaw=b["yaw"], length=b["L"], width=b["W"],
                         height=b["z1"]-b["z0"], reliable=True, size_plausible=True, length_fit=0.0, n_points=b["n"])
            lst.append(dict(label="car", source="radar_motion_rescue", score=0.6, color="lime", bbox_xyxy=bbox,
                            box3d=box3d, radar_confirmed=True,
                            forward=[b["cx"]-b["L"]/2, b["cx"]+b["L"]/2], lateral=[b["cy"]-b["W"]/2, b["cy"]+b["W"]/2]))
        allobj[str(T)] = lst
    ANN = f"{RES}/annotations/rescue_v4.json"
    json.dump(dict(objects=allobj), open(ANN, "w"), indent=1)
    print(f"wrote {ANN} ({sum(len(v) for v in allobj.values())} boxes / {len(allobj)} pairs)", flush=True)

# ======== src/fusion/radar_first/raise_boxes_rescue.py ========
# Night box-raising with radar RANGE-rescue + VELOCITY-rescue (verify on pair 14280).
# 
# Raise a car box for a LiDAR cluster if ANY holds:
#   A sizeok           -- clean L-shape (existing path)
#   B range-rescue     -- a strong radar arc at the cluster's RANGE (front-face / merged cars)
#   C velocity-rescue  -- cluster moves across frames AND a radar return near it has a
#                         velocity (alias-aware) matching the cluster's line-of-sight motion
# Radar is used only in its precise axes (range, velocity), never for azimuth localization.
def raise_main():
    import os, sys
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import load_pairs, load_calib, LidarSource, LOCAL, RES, H5, CFG, pose_apply
    from detector import RadarDetector
    from odometry import odometry_scan2map
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import (_cluster_bev, _drop_ground, _ground_level,
                               _radar_setup, _radar_panel, _peaks_from_bev,
                               _extract_frames, DEFAULT_COLOR_TOPIC)
    TARGET = 14280; HALF = 6; DV = 3            # window TARGET-HALF..TARGET+HALF; velocity from +/-DV
    TARGET = 14280; HALF = 6; DV = 3            # window TARGET-HALF..TARGET+HALF; velocity from +/-DV
    TARGET = 14280; HALF = 6; DV = 3            # window TARGET-HALF..TARGET+HALF; velocity from +/-DV
    ROI = (0.0, 30.0, -16.0, 16.0); GPCT, MINH = 10.0, 0.3
    ROI = (0.0, 30.0, -16.0, 16.0); GPCT, MINH = 10.0, 0.3
    MAJ = (2.0, 6.5); MIN = (1.2, 3.5); HGT = (0.5, 2.4)
    MAJ = (2.0, 6.5); MIN = (1.2, 3.5); HGT = (0.5, 2.4)
    MAJ = (2.0, 6.5); MIN = (1.2, 3.5); HGT = (0.5, 2.4)
    RANGE_TOL = 1.5; AZ_TOL = 20.0; VMAX = 2.44; VEL_TOL = 1.2; MOVE_MIN = 0.6
    RANGE_TOL = 1.5; AZ_TOL = 20.0; VMAX = 2.44; VEL_TOL = 1.2; MOVE_MIN = 0.6
    RANGE_TOL = 1.5; AZ_TOL = 20.0; VMAX = 2.44; VEL_TOL = 1.2; MOVE_MIN = 0.6
    RANGE_TOL = 1.5; AZ_TOL = 20.0; VMAX = 2.44; VEL_TOL = 1.2; MOVE_MIN = 0.6
    RANGE_TOL = 1.5; AZ_TOL = 20.0; VMAX = 2.44; VEL_TOL = 1.2; MOVE_MIN = 0.6
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAMERA_BAG = f"{LOCAL}/camera/20260511_213724.bag"; CAMERA_INDEX = f"{RES}/camera_index.npz"
    CAMERA_BAG = f"{LOCAL}/camera/20260511_213724.bag"; CAMERA_INDEX = f"{RES}/camera_index.npz"
    pairs = load_pairs(); cal = load_calib()
    pairs = load_pairs(); cal = load_calib()
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    R2L = np.array(cal["radar_to_lidar_auto"]); A = R2L[:2, :2]; t = R2L[:2, 3]
    R2L = np.array(cal["radar_to_lidar_auto"]); A = R2L[:2, :2]; t = R2L[:2, 3]
    R2L = np.array(cal["radar_to_lidar_auto"]); A = R2L[:2, :2]; t = R2L[:2, 3]
    ids = [p for p in range(TARGET - HALF, TARGET + HALF + 1) if p in pairs]
    ti = ids.index(TARGET)
    times = np.array([pairs[p]["radar_t"] for p in ids]); times -= times[0]
    times = np.array([pairs[p]["radar_t"] for p in ids]); times -= times[0]
    def clusters_of(P):
        x0, x1, y0, y1 = ROI
        reg = P[(P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)]
        gz = _ground_level(reg, GPCT); obj = _drop_ground(reg, GPCT, MINH)
        if len(obj) < 8: return []
        lab = _cluster_bev(obj[:, :2], ROI); out = []
        for lb in np.unique(lab):
            cl = obj[lab == lb]
            if cl.shape[0] < 12: continue
            L = float(np.ptp(cl[:, 0])); W = float(np.ptp(cl[:, 1])); H = float(np.ptp(cl[:, 2]))
            mx, mn = max(L, W), min(L, W)
            sizeok = (MAJ[0] <= mx <= MAJ[1] and MIN[0] <= mn <= MIN[1] and HGT[0] <= H <= HGT[1])
            out.append(dict(cen=cl[:, :2].mean(0), L=L, W=W, H=H, sizeok=sizeok, gz=gz, pts=cl))
        return out
    src = LidarSource()
    scans = [src.xyz(pairs[p]["lidar_idx"]) for p in ids]
    poses, _ = odometry_scan2map(scans)
    cl_per = [clusters_of(s) for s in scans]
    for i, cls in enumerate(cl_per):
        for c in cls:
            c["cen_w"] = pose_apply(poses[i], c["cen"][None, :])[0]
    def world_vel(cen_w, i):
        """LiDAR cluster velocity from +/-DV frames (nearest world centroid)."""
        def nearest(j):
            if j < 0 or j >= len(cl_per) or not cl_per[j]: return None
            cs = np.array([c["cen_w"] for c in cl_per[j]])
            d = np.hypot(cs[:, 0]-cen_w[0], cs[:, 1]-cen_w[1]); k = int(np.argmin(d))
            return cl_per[j][k]["cen_w"] if d[k] < 2.5 else None
        a, b = nearest(i-DV), nearest(i+DV)
        if a is None or b is None: return None
        dt = times[min(i+DV, len(times)-1)] - times[max(i-DV, 0)]
        return (b - a) / dt if dt > 0 else None
    det = RadarDetector(processing="legacy")
    d, _ = det.detect(pairs[TARGET]["radar_idx"])
    Xd, Yd = RadarDetector.to_xy(d)                                   # radar frame
    lf = R2L[0,0]*Xd + R2L[0,1]*Yd + R2L[0,3]; ll = R2L[1,0]*Xd + R2L[1,1]*Yd + R2L[1,3]  # lidar frame
    lf = R2L[0,0]*Xd + R2L[0,1]*Yd + R2L[0,3]; ll = R2L[1,0]*Xd + R2L[1,1]*Yd + R2L[1,3]  # lidar frame
    det_rng = np.hypot(lf, ll); det_ang = np.degrees(np.arctan2(ll, lf)); det_vel = d["vel"]
    det_rng = np.hypot(lf, ll); det_ang = np.degrees(np.arctan2(ll, lf)); det_vel = d["vel"]
    det_rng = np.hypot(lf, ll); det_ang = np.degrees(np.arctan2(ll, lf)); det_vel = d["vel"]
    args = SimpleNamespace(radar_h5=H5, cfg_file=CFG, sync_csv=f"{RES}/sync_pairs.csv",
                           sync_summary=f"{RES}/sync_summary.json", lidar_pcap=None, lidar_metadata=None, radar_bev_res=0.1)
    loader, Rmat = _radar_setup(args, cal, doppler_mode="all", processing="legacy")
    bev, la, fa = _radar_panel(loader, TARGET); pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.0)
    bev, la, fa = _radar_panel(loader, TARGET); pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.0)
    peak_rng = np.hypot(pf, pl) if len(pf) else np.array([])
    traj = np.array([p[:3, 3][:2] for p in poses]); dtv = np.r_[0.1, np.diff(times)]; dtv[dtv<=0]=0.1
    traj = np.array([p[:3, 3][:2] for p in poses]); dtv = np.r_[0.1, np.diff(times)]; dtv[dtv<=0]=0.1
    traj = np.array([p[:3, 3][:2] for p in poses]); dtv = np.r_[0.1, np.diff(times)]; dtv[dtv<=0]=0.1
    ego_v = np.zeros((len(poses), 2)); ego_v[1:] = np.diff(traj, axis=0)/dtv[1:, None]; ego_v[0]=ego_v[1]
    ego_v = np.zeros((len(poses), 2)); ego_v[1:] = np.diff(traj, axis=0)/dtv[1:, None]; ego_v[0]=ego_v[1]
    ego_v = np.zeros((len(poses), 2)); ego_v[1:] = np.diff(traj, axis=0)/dtv[1:, None]; ego_v[0]=ego_v[1]
    def vel_rescue(c):
        v = world_vel(c["cen_w"], ti)
        if v is None or np.hypot(*v) < MOVE_MIN: return False, None
        los = c["cen"] / (np.linalg.norm(c["cen"]) + 1e-6)          # lidar-frame LOS (ego~origin)
        # relative radial velocity expected (world vel rotated to lidar frame ~ same axes here)
        exp = float((v - ego_v[ti]) @ los)
        rng = float(np.hypot(*c["cen"])); ang = float(np.degrees(np.arctan2(c["cen"][1], c["cen"][0])))
        near = (np.abs(det_rng - rng) < RANGE_TOL) & (np.abs(((det_ang-ang+180)%360)-180) < AZ_TOL)
        if not near.any(): return False, v
        for mv in det_vel[near]:
            for k in range(-3, 4):
                for s in (1, -1):
                    if abs(exp - s*(mv + k*2*VMAX)) < VEL_TOL:
                        return True, v
        return False, v
    PRIOR = dict(L=4.3, W=1.9, H=1.5)
    raised = []
    for c in cl_per[ti]:
        rng = float(np.hypot(*c["cen"]))
        reason = None
        if c["sizeok"]:
            reason = "sizeok"
        elif len(peak_rng) and (np.abs(peak_rng - rng) < RANGE_TOL).any() and HGT[0] <= c["H"] <= 4.0:
            reason = "range"
        else:
            ok, _ = vel_rescue(c)
            if ok: reason = "velocity"
        if reason:
            raised.append(dict(c=c, reason=reason, rng=rng))
    print(f"pair {TARGET}: {len(cl_per[ti])} clusters -> raised {len(raised)} "
          f"(sizeok {sum(r['reason']=='sizeok' for r in raised)}, "
          f"range {sum(r['reason']=='range' for r in raised)}, "
          f"velocity {sum(r['reason']=='velocity' for r in raised)})", flush=True)
    for r in sorted(raised, key=lambda z: z["rng"]):
        c = r["c"]; print(f"  {r['reason']:8s} rng {r['rng']:5.1f}  cen=({c['cen'][0]:5.1f},{c['cen'][1]:6.1f})  "
                          f"L{c['L']:.1f} W{c['W']:.1f} H{c['H']:.1f}", flush=True)
    def box_from(c, reason):
        pts = c["pts"]; xy = pts[:, :2]; cc = xy.mean(0); Xc = xy - cc
        _, _, vt = np.linalg.svd(Xc, full_matrices=False); yaw = np.arctan2(vt[0,1], vt[0,0])
        if reason == "sizeok":
            L = max(float(np.ptp((Xc@vt.T)[:, 0])), 1.0); W = max(float(np.ptp((Xc@vt.T)[:, 1])), 1.4)
        else:                                   # partial view -> class prior, face the ego
            L, W = PRIOR["L"], PRIOR["W"]; yaw = np.arctan2(cc[1], cc[0])
        z0 = float(pts[:, 2].min())
        return dict(cx=cc[0], cy=cc[1], yaw=yaw, L=L, W=W, z0=z0, z1=z0 + PRIOR["H"])
    def corners(b):
        dx=np.array([b["L"]/2,b["L"]/2,-b["L"]/2,-b["L"]/2]); dy=np.array([b["W"]/2,-b["W"]/2,-b["W"]/2,b["W"]/2])
        cs,sn=np.cos(b["yaw"]),np.sin(b["yaw"]); fx=b["cx"]+dx*cs-dy*sn; fy=b["cy"]+dx*sn+dy*cs
        return np.c_[fx,fy,np.full(4,b["z0"])], np.c_[fx,fy,np.full(4,b["z1"])]
    def to_px(xyz):
        opt=(L2C@np.r_[xyz,1])[:3]
        return None if opt[2]<=0.2 else np.array([ci["fx"]*opt[0]/opt[2]+ci["cx"], ci["fy"]*opt[1]/opt[2]+ci["cy"]])
    COL = {"sizeok": "limegreen", "range": "orange", "velocity": "magenta"}
    import json
    objects = {str(TARGET): []}
    for r in raised:
        b = box_from(r["c"], r["reason"]); bo, tp = corners(b)
        px = [to_px(x) for x in list(bo) + list(tp)]; px = [p for p in px if p is not None]
        bbox = None
        if len(px) >= 4:
            arr = np.array(px); u0, v0 = arr.min(0); u1, v1 = arr.max(0)
            bbox = [float(u0), float(v0), float(u1), float(v1)]
        box3d = dict(cx=float(b["cx"]), cy=float(b["cy"]), cz=float((b["z0"]+b["z1"])/2),
                     yaw=float(b["yaw"]), length=float(b["L"]), width=float(b["W"]),
                     height=float(b["z1"]-b["z0"]), reliable=True, size_plausible=True,
                     length_fit=0.0, n_points=int(r["c"]["pts"].shape[0]))
        objects[str(TARGET)].append(dict(
            label="car", source=f"radar_{r['reason']}_rescue", score=0.5, color="lime",
            bbox_xyxy=bbox, box3d=box3d, radar_confirmed=(r["reason"] != "sizeok"),
            forward=[float(b["cx"]-b["L"]/2), float(b["cx"]+b["L"]/2)],
            lateral=[float(b["cy"]-b["W"]/2), float(b["cy"]+b["W"]/2)]))
    os.makedirs(f"{RES}/annotations", exist_ok=True)
    ANN = f"{RES}/annotations/rescue_14280.json"
    json.dump(dict(objects=objects), open(ANN, "w"), indent=1)
    print(f"wrote annotations {ANN} ({len(objects[str(TARGET)])} boxes)", flush=True)
    bag_ts = np.load(CAMERA_INDEX)["color_bag_ts_ns"]
    img = _extract_frames(CAMERA_BAG, DEFAULT_COLOR_TOPIC, [pairs[TARGET]["color_idx"]], Path(CACHE), "color", bag_ts).get(pairs[TARGET]["color_idx"])
    if img is not None and img.ndim == 1: img = img.reshape(ci["height"], ci["width"], 3)
    P = scans[ti]
    fig, axs = plt.subplots(1, 2, figsize=(19, 8))
    if img is not None: axs[0].imshow(img)
    for r in raised:
        b = box_from(r["c"], r["reason"]); bo, tp = corners(b)
        pb=[to_px(x) for x in bo]; pt=[to_px(x) for x in tp]
        if any(x is None for x in pb+pt): continue
        pb=np.array(pb); pt=np.array(pt); col=COL[r["reason"]]
        for aa in (pb,pt): axs[0].plot(np.r_[aa[:,0],aa[0,0]], np.r_[aa[:,1],aa[0,1]], col, lw=2)
        for k in range(4): axs[0].plot([pb[k,0],pt[k,0]],[pb[k,1],pt[k,1]], col, lw=2)
    axs[0].set_xlim(0,ci["width"]); axs[0].set_ylim(ci["height"],0); axs[0].axis("off")
    axs[0].set_xlim(0,ci["width"]); axs[0].set_ylim(ci["height"],0); axs[0].axis("off")
    axs[0].set_xlim(0,ci["width"]); axs[0].set_ylim(ci["height"],0); axs[0].axis("off")
    axs[0].set_title(f"CAMERA pair {TARGET} -- recovered boxes\ngreen=sizeok  orange=range-rescue  magenta=velocity-rescue")
    ml=np.linalg.norm(P[:,:2],axis=1)<30
    axs[1].scatter(P[ml,1],P[ml,0],s=0.4,c="0.85")
    if len(pf): axs[1].scatter(pl,pf,s=16,marker="x",c="purple",label="radar peak")
    for r in raised:
        b=box_from(r["c"],r["reason"]); bo,_=corners(b); col=COL[r["reason"]]
        axs[1].plot(np.r_[bo[:,1],bo[0,1]], np.r_[bo[:,0],bo[0,0]], col, lw=2)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15,-15); axs[1].set_ylim(0,28)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15,-15); axs[1].set_ylim(0,28)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15,-15); axs[1].set_ylim(0,28)
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
    axs[1].set_title(f"BEV -- {len(raised)} recovered boxes"); axs[1].legend(loc="upper right", fontsize=8)
    axs[1].set_title(f"BEV -- {len(raised)} recovered boxes"); axs[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout(); out=f"{RES}/fusion/recover_14280.png"; plt.savefig(out,dpi=120,bbox_inches="tight")
    plt.tight_layout(); out=f"{RES}/fusion/recover_14280.png"; plt.savefig(out,dpi=120,bbox_inches="tight")
    plt.tight_layout(); out=f"{RES}/fusion/recover_14280.png"; plt.savefig(out,dpi=120,bbox_inches="tight")
    print(f"saved {out}", flush=True)

# ======== src/fusion/radar_first/motion_rescue.py ========
# Motion-matched night rescue: capture moving cars, reject static background.
# 
# For each LiDAR cluster at a target frame:
#   - world velocity from cross-frame tracking (ego-compensated odometry)
#   - FILTER: |world velocity| < MOVE_MIN  -> static background (wall/pole/ground) -> drop
#   - CAPTURE: cluster near a MOVING radar return whose Doppler (alias-aware) matches the
#     cluster's line-of-sight velocity -> it's the real moving car -> raise box
# This resolves the ±azimuth arc ambiguity by MOTION, not flaky bearing.
#     python3 motion_rescue.py 14260 14270 14280 14290 14300
def motion_main():
    import os, sys, json
    import numpy as np
    from sklearn.cluster import DBSCAN
    from common import load_pairs, load_calib, LidarSource, RES, pose_apply
    from detector import RadarDetector
    from odometry import odometry_scan2map
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import _ground_level
    TARGETS = [int(a) for a in sys.argv[1:]] or [14280]
    MARGIN = 5; DV = 3
    MARGIN = 5; DV = 3
    ROI = (0.0, 30.0, -16.0, 16.0); GPCT = 10.0; HBAND = (0.3, 2.2)
    ROI = (0.0, 30.0, -16.0, 16.0); GPCT = 10.0; HBAND = (0.3, 2.2)
    ROI = (0.0, 30.0, -16.0, 16.0); GPCT = 10.0; HBAND = (0.3, 2.2)
    MOVE_MIN = 0.6              # world speed below this = static background -> filtered
    RANGE_TOL = 1.5; AZ_TOL = 16.0; VMAX = 2.44; VEL_TOL = 1.3
    RANGE_TOL = 1.5; AZ_TOL = 16.0; VMAX = 2.44; VEL_TOL = 1.3
    RANGE_TOL = 1.5; AZ_TOL = 16.0; VMAX = 2.44; VEL_TOL = 1.3
    RANGE_TOL = 1.5; AZ_TOL = 16.0; VMAX = 2.44; VEL_TOL = 1.3
    MIN_PTS = 12
    pairs = load_pairs(); cal = load_calib()
    pairs = load_pairs(); cal = load_calib()
    ci = cal["color_intrinsics"]; L2C = np.linalg.inv(np.array(cal["camera_to_lidar_auto"]))
    ci = cal["color_intrinsics"]; L2C = np.linalg.inv(np.array(cal["camera_to_lidar_auto"]))
    R2L = np.array(cal["radar_to_lidar_auto"])
    W0, W1 = min(TARGETS) - MARGIN, max(TARGETS) + MARGIN
    ids = [p for p in range(W0, W1 + 1) if p in pairs]
    times = np.array([pairs[p]["radar_t"] for p in ids]); times -= times[0]
    times = np.array([pairs[p]["radar_t"] for p in ids]); times -= times[0]
    idx_of = {p: i for i, p in enumerate(ids)}
    src = LidarSource()
    scans = [src.xyz(pairs[p]["lidar_idx"]) for p in ids]
    poses, _ = odometry_scan2map(scans)
    traj = np.array([p[:3, 3][:2] for p in poses]); dtv = np.r_[0.1, np.diff(times)]; dtv[dtv <= 0] = 0.1
    traj = np.array([p[:3, 3][:2] for p in poses]); dtv = np.r_[0.1, np.diff(times)]; dtv[dtv <= 0] = 0.1
    traj = np.array([p[:3, 3][:2] for p in poses]); dtv = np.r_[0.1, np.diff(times)]; dtv[dtv <= 0] = 0.1
    ego_v = np.zeros((len(poses), 2)); ego_v[1:] = np.diff(traj, axis=0) / dtv[1:, None]; ego_v[0] = ego_v[1]
    ego_v = np.zeros((len(poses), 2)); ego_v[1:] = np.diff(traj, axis=0) / dtv[1:, None]; ego_v[0] = ego_v[1]
    ego_v = np.zeros((len(poses), 2)); ego_v[1:] = np.diff(traj, axis=0) / dtv[1:, None]; ego_v[0] = ego_v[1]
    def clusters_of(P, pose):
        x0, x1, y0, y1 = ROI
        reg = P[(P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)]
        gz = _ground_level(reg, GPCT)
        slab = reg[(reg[:, 2] > gz + HBAND[0]) & (reg[:, 2] < gz + HBAND[1])]
        if len(slab) < MIN_PTS: return []
        lab = DBSCAN(eps=1.0, min_samples=6).fit_predict(slab[:, :2]); out = []
        for c in set(lab.tolist()):
            if c == -1: continue
            cl = slab[lab == c]
            if cl.shape[0] < MIN_PTS: continue
            cen = cl[:, :2].mean(0)
            out.append(dict(pts=cl, cen=cen, cen_w=pose_apply(pose, cen[None, :])[0]))
        return out
    cl_per = [clusters_of(scans[i], poses[i]) for i in range(len(ids))]
    def world_vel(cen_w, i):
        def nearest(j):
            if j < 0 or j >= len(cl_per) or not cl_per[j]: return None
            cs = np.array([c["cen_w"] for c in cl_per[j]])
            d = np.hypot(cs[:, 0]-cen_w[0], cs[:, 1]-cen_w[1]); k = int(np.argmin(d))
            return cl_per[j][k]["cen_w"] if d[k] < 2.5 else None
        a, b = nearest(i-DV), nearest(i+DV)
        if a is None or b is None: return None
        dt = times[min(i+DV, len(times)-1)] - times[max(i-DV, 0)]
        return (b - a) / dt if dt > 0 else None
    def alias_match(exp, measured):
        for k in range(-3, 4):
            for s in (1, -1):
                if abs(exp - s*(measured + k*2*VMAX)) < VEL_TOL:
                    return True
        return False
    det = RadarDetector(processing="ti")   # moving-target radar detections (range,az,vel)
    def box_from(cl):
        xy = cl[:, :2]; cen = xy.mean(0); X = xy - cen
        _, _, vt = np.linalg.svd(X, full_matrices=False); yaw = float(np.arctan2(vt[0, 1], vt[0, 0]))
        proj = X @ vt.T; L = max(float(np.ptp(proj[:, 0])), 1.5); W = max(float(np.ptp(proj[:, 1])), 1.6)
        if L < 3.0: L = 4.3; yaw = float(np.arctan2(cen[1], cen[0]))
        z0 = float(cl[:, 2].min())
        return dict(cx=float(cen[0]), cy=float(cen[1]), yaw=yaw, L=min(L, 6.0), W=min(W, 2.2), z0=z0, z1=z0+1.5)
    def corners(b):
        dx = np.array([b["L"]/2]*2 + [-b["L"]/2]*2); dy = np.array([b["W"]/2, -b["W"]/2, -b["W"]/2, b["W"]/2])
        cs, sn = np.cos(b["yaw"]), np.sin(b["yaw"]); fx = b["cx"]+dx*cs-dy*sn; fy = b["cy"]+dx*sn+dy*cs
        return np.c_[fx, fy, np.full(4, b["z0"])], np.c_[fx, fy, np.full(4, b["z1"])]
    def to_px(xyz):
        opt = (L2C @ np.r_[xyz, 1])[:3]
        return None if opt[2] <= 0.2 else [float(ci["fx"]*opt[0]/opt[2]+ci["cx"]), float(ci["fy"]*opt[1]/opt[2]+ci["cy"])]
    allobj = {}
    for T in TARGETS:
        ti = idx_of[T]
        d, _ = det.detect(pairs[T]["radar_idx"], min_abs_vel=0.3)
        if len(d):
            Xd, Yd = RadarDetector.to_xy(d)
            lf = R2L[0,0]*Xd+R2L[0,1]*Yd+R2L[0,3]; ll = R2L[1,0]*Xd+R2L[1,1]*Yd+R2L[1,3]
            drng = np.hypot(lf, ll); dang = np.degrees(np.arctan2(ll, lf)); dvel = d["vel"]
        else:
            drng = dang = dvel = np.array([])
        boxes = []; n_static = n_nobk = n_nomatch = 0
        for cl in cl_per[ti]:
            v = world_vel(cl["cen_w"], ti)
            if v is None or np.hypot(*v) < MOVE_MIN:
                n_static += 1; continue                      # static background filtered
            rng = float(np.hypot(*cl["cen"])); ang = float(np.degrees(np.arctan2(cl["cen"][1], cl["cen"][0])))
            near = (np.abs(drng - rng) < RANGE_TOL) & (np.abs(((dang-ang+180) % 360)-180) < AZ_TOL) if len(drng) else np.array([], bool)
            if not near.any():
                n_nobk += 1; continue                        # no moving-radar backing
            egp = poses[ti][:3, 3][:2]; u = cl["cen_w"] - egp; u = u/(np.linalg.norm(u)+1e-6)
            exp = float((v - ego_v[ti]) @ u)
            if not any(alias_match(exp, mv) for mv in dvel[near]):
                n_nomatch += 1; continue                     # Doppler doesn't match cluster motion
            b = box_from(cl["pts"]); b["speed"] = float(np.hypot(*v)); boxes.append(b)
        # light NMS
        boxes.sort(key=lambda b: -b["speed"]); kept = []
        for b in boxes:
            if any(np.hypot(b["cx"]-k["cx"], b["cy"]-k["cy"]) < 2.5 for k in kept): continue
            kept.append(b)
        print(f"pair {T}: {len(cl_per[ti])} clusters -> static-filtered {n_static}, no-radar {n_nobk}, "
              f"no-velmatch {n_nomatch} -> {len(kept)} boxes", flush=True)
        for b in kept:
            print(f"    rng {np.hypot(b['cx'],b['cy']):5.1f} cen=({b['cx']:5.1f},{b['cy']:6.1f}) spd {b['speed']:.1f} m/s", flush=True)
        lst = []
        for b in kept:
            bo, tp = corners(b); px = [to_px(x) for x in list(bo)+list(tp)]; px = [p for p in px if p]
            bbox = None
            if len(px) >= 4:
                a = np.array(px); bbox = [float(a[:,0].min()), float(a[:,1].min()), float(a[:,0].max()), float(a[:,1].max())]
            box3d = dict(cx=b["cx"], cy=b["cy"], cz=(b["z0"]+b["z1"])/2, yaw=b["yaw"], length=b["L"], width=b["W"],
                         height=b["z1"]-b["z0"], reliable=True, size_plausible=True, length_fit=0.0, n_points=0)
            lst.append(dict(label="car", source="radar_motion_rescue", score=0.6, color="lime",
                            bbox_xyxy=bbox, box3d=box3d, radar_confirmed=True,
                            forward=[b["cx"]-b["L"]/2, b["cx"]+b["L"]/2], lateral=[b["cy"]-b["W"]/2, b["cy"]+b["W"]/2]))
        allobj[str(T)] = lst
    ANN = f"{RES}/annotations/motion_rescue.json"
    json.dump(dict(objects=allobj), open(ANN, "w"), indent=1)
    print(f"wrote {ANN} ({sum(len(v) for v in allobj.values())} boxes / {len(allobj)} pairs)", flush=True)

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'v4': v4_main, 'raise': raise_main, 'motion': motion_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
