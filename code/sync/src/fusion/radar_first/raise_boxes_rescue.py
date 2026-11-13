"""Night box-raising with radar RANGE-rescue + VELOCITY-rescue (verify on pair 14280).

Raise a car box for a LiDAR cluster if ANY holds:
  A sizeok           -- clean L-shape (existing path)
  B range-rescue     -- a strong radar arc at the cluster's RANGE (front-face / merged cars)
  C velocity-rescue  -- cluster moves across frames AND a radar return near it has a
                        velocity (alias-aware) matching the cluster's line-of-sight motion
Radar is used only in its precise axes (range, velocity), never for azimuth localization.
"""
import os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_pairs, load_calib, LidarSource, LOCAL, RES, H5, CFG, pose_apply
from detector import RadarDetector
from odometry import odometry_scan2map

sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import (_cluster_bev, _drop_ground, _ground_level,
                           _radar_setup, _radar_panel, _peaks_from_bev,
                           _extract_frames, DEFAULT_COLOR_TOPIC)

TARGET = 14280; HALF = 6; DV = 3            # window TARGET-HALF..TARGET+HALF; velocity from +/-DV
ROI = (0.0, 30.0, -16.0, 16.0); GPCT, MINH = 10.0, 0.3
MAJ = (2.0, 6.5); MIN = (1.2, 3.5); HGT = (0.5, 2.4)
RANGE_TOL = 1.5; AZ_TOL = 20.0; VMAX = 2.44; VEL_TOL = 1.2; MOVE_MIN = 0.6
CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAMERA_BAG = f"{LOCAL}/camera/20260511_213724.bag"; CAMERA_INDEX = f"{RES}/camera_index.npz"

pairs = load_pairs(); cal = load_calib()
C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
R2L = np.array(cal["radar_to_lidar_auto"]); A = R2L[:2, :2]; t = R2L[:2, 3]
ids = [p for p in range(TARGET - HALF, TARGET + HALF + 1) if p in pairs]
ti = ids.index(TARGET)
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


# --- lidar over window + odometry ---
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


# --- radar at target frame ---
det = RadarDetector(processing="legacy")
d, _ = det.detect(pairs[TARGET]["radar_idx"])
Xd, Yd = RadarDetector.to_xy(d)                                   # radar frame
lf = R2L[0,0]*Xd + R2L[0,1]*Yd + R2L[0,3]; ll = R2L[1,0]*Xd + R2L[1,1]*Yd + R2L[1,3]  # lidar frame
det_rng = np.hypot(lf, ll); det_ang = np.degrees(np.arctan2(ll, lf)); det_vel = d["vel"]
args = SimpleNamespace(radar_h5=H5, cfg_file=CFG, sync_csv=f"{RES}/sync_pairs.csv",
                       sync_summary=f"{RES}/sync_summary.json", lidar_pcap=None, lidar_metadata=None, radar_bev_res=0.1)
loader, Rmat = _radar_setup(args, cal, doppler_mode="all", processing="legacy")
bev, la, fa = _radar_panel(loader, TARGET); pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.0)
peak_rng = np.hypot(pf, pl) if len(pf) else np.array([])

# ego world velocity at target
traj = np.array([p[:3, 3][:2] for p in poses]); dtv = np.r_[0.1, np.diff(times)]; dtv[dtv<=0]=0.1
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


# --- classify + raise ---
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

# --- box + projection ---
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

# --- write annotations JSON (yolo_to_lidar format) so viz_labels can render the 4-panel ---
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
axs[0].set_title(f"CAMERA pair {TARGET} -- recovered boxes\ngreen=sizeok  orange=range-rescue  magenta=velocity-rescue")

ml=np.linalg.norm(P[:,:2],axis=1)<30
axs[1].scatter(P[ml,1],P[ml,0],s=0.4,c="0.85")
if len(pf): axs[1].scatter(pl,pf,s=16,marker="x",c="purple",label="radar peak")
for r in raised:
    b=box_from(r["c"],r["reason"]); bo,_=corners(b); col=COL[r["reason"]]
    axs[1].plot(np.r_[bo[:,1],bo[0,1]], np.r_[bo[:,0],bo[0,0]], col, lw=2)
axs[1].set_aspect("equal"); axs[1].set_xlim(15,-15); axs[1].set_ylim(0,28)
axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
axs[1].set_title(f"BEV -- {len(raised)} recovered boxes"); axs[1].legend(loc="upper right", fontsize=8)
plt.tight_layout(); out=f"{RES}/fusion/recover_14280.png"; plt.savefig(out,dpi=120,bbox_inches="tight")
print(f"saved {out}", flush=True)
