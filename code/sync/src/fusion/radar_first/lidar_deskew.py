"""LiDAR motion compensation (deskew) for the GT-queue proposals/overlays.

The Ouster sweeps 360 deg over ~100 ms (per-column timestamps in scan.timestamp). When the EGO
moves, points captured early in the sweep are from a different ego pose than points captured late,
so the cloud smears relative to a camera snapshot. We correct each point to the instant the lidar
looked forward (~the camera capture time) using the per-column timestamp and the ego velocity.

IMPORTANT (learned the hard way 2026-10-08): the ego velocity MUST come from LiDAR ICP, NOT the
single-frame radar ego-Doppler fit -- on mover-heavy glare frames the radar fit blows up (gave
4.9 m/s LATERAL on 19877 where lidar ICP proves the ego is stationary at 0.01 m/s). Ego speeds are
precomputed by the lidar-ICP census into gt_queue/ego_motion.json: {frame: [vx,vy,speed,fit]} in
the ICP (before->after) convention, so ego velocity in the lidar frame (forward +X) = -(vx,vy).

Per the census, only ~2 frames (28770 @12 m/s, 36000 @6.9 m/s) have skew worth correcting (>0.5 m);
below ~1.5 m/s the correction is <0.15 m (within the 2 m labeling tolerance), so deskew is a no-op
there. Deskew does NOT fix moving-OBJECT offsets (a moving car's own motion between sweep and frame)
-- that is per-object and inherent.
"""
import json, os
import numpy as np

EGO_JSON = "sync/res/benchmark_v1/gt_queue/ego_motion.json"


def load_ego(path=EGO_JSON):
    """frame(int) -> np.array([vx,vy,speed,fit])  (ICP before->after convention)."""
    if not os.path.exists(path):
        return {}
    return {int(f): np.array(e, float) for f, e in json.load(open(path))["ego"].items() if e is not None}


def deskew_xyz(xyz, tpt, vego_lidar, az_ref=5.0):
    """Shift each point to the forward-look instant. xyz (N,3), tpt (N,) per-point ns timestamp,
    vego_lidar (2,) ego velocity in lidar frame (forward +X, m/s). Returns corrected (N,3)."""
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    fwd = (np.abs(az) < az_ref) & (xyz[:, 0] > 0)
    tref = np.median(tpt[fwd]) if fwd.any() else np.median(tpt)
    dt = (tpt - tref) * 1e-9                       # s, signed (points early in sweep: dt<0)
    out = xyz.copy()
    out[:, 0] += vego_lidar[0] * dt                # p + v_ego*(t_i - t_ref)
    out[:, 1] += vego_lidar[1] * dt
    return out


def radar_movers_lidar(rdet, radar_idx, R2L):
    """Radar movers in the LIDAR BEV frame: list of (range_m, bearing_deg, v_radial_ground).
    v_radial_ground = the mover's radial velocity w.r.t. GROUND (ego-Doppler removed)."""
    from detector import RadarDetector
    from ego_doppler import fit_ego_velocity
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        d, _ = rdet.detect(radar_idx)
    if not len(d):
        return []
    Xd, Yd = RadarDetector.to_xy(d)
    rf = R2L[0, 0]*Xd + R2L[0, 1]*Yd + R2L[0, 3]; rl = R2L[1, 0]*Xd + R2L[1, 1]*Yd + R2L[1, 3]
    rng = np.hypot(rf, rl); bear = np.degrees(np.arctan2(rl, rf))
    ab, inl = fit_ego_velocity(d["az"], d["vel"])
    if ab is None:
        return []
    pr = np.cos(np.radians(-d["az"]))*ab[0] + np.sin(np.radians(-d["az"]))*ab[1]
    resid = d["vel"] - pr                                    # ground-frame radial velocity
    mv = (~inl) & (np.abs(resid) > 0.8)
    return [(float(rng[i]), float(bear[i]), float(resid[i])) for i in np.where(mv)[0]]


IMU_CACHE = "sync/res/2026.05.11/21.37.07/imu_cache.npz"


def load_imu(path=IMU_CACHE):
    import os
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {"ts": d["ts"], "gyro": d["gyro"], "accel": d["accel"]}   # gyro deg/s, accel g, sensor frame


def ego_yaw_rate(imu, t0_ns, t1_ns, pad_ns=5_000_000):
    """Mean ego yaw rate (rad/s, about sensor +Z) over a scan window, from the Ouster IMU gyro."""
    if imu is None:
        return 0.0
    m = (imu["ts"] >= t0_ns - pad_ns) & (imu["ts"] <= t1_ns + pad_ns)
    if not m.any():
        return 0.0
    return float(np.radians(imu["gyro"][m, 2].mean()))


def _fit_ground(p):
    lo = p[p[:, 2] < np.median(p[:, 2])]; A = np.c_[lo[:, 0], lo[:, 1], np.ones(len(lo))]
    c, *_ = np.linalg.lstsq(A, lo[:, 2], rcond=None)
    for _ in range(4):
        r = p[:, 2] - (p[:, 0]*c[0] + p[:, 1]*c[1] + c[2]); inl = np.abs(r) < 0.3
        if inl.sum() < 50: break
        A = np.c_[p[inl, 0], p[inl, 1], np.ones(int(inl.sum()))]; c, *_ = np.linalg.lstsq(A, p[inl, 2], rcond=None)
    return c


def car_clusters(xyz, tpt, max_range=45.0):
    """Car-footprint lidar clusters: list of (centroid_xy(2,), ts_median, n_pts). Reuses the geometric
    detector (ground slab -> DBSCAN -> loose car footprint)."""
    from sklearn.cluster import DBSCAN
    m = np.hypot(xyz[:, 0], xyz[:, 1]) < max_range
    P, T = xyz[m], tpt[m]
    if len(P) < 30:
        return []
    coef = _fit_ground(P); ha = P[:, 2] - (P[:, 0]*coef[0] + P[:, 1]*coef[1] + coef[2])
    sl = (ha > 0.3) & (ha < 2.2); slab, slabt = P[sl], T[sl]
    out = []
    if len(slab) >= 6:
        lab = DBSCAN(eps=0.9, min_samples=5).fit_predict(slab[:, :2])
        for c in set(lab.tolist()) - {-1}:
            cl = slab[lab == c]
            if len(cl) < 5: continue
            X = cl[:, :2] - cl[:, :2].mean(0); _, _, vt = np.linalg.svd(X, full_matrices=False); pr = X @ vt.T
            mj, mn = float(np.ptp(pr[:, 0])), float(np.ptp(pr[:, 1]))
            if 1.0 <= mj <= 7.0 and 0.4 <= mn <= 3.5:
                out.append((cl[:, :2].mean(0), float(np.median(slabt[lab == c])), int(len(cl))))
    return out


def track_objects(src, scan_idx, ego_vel_lidar, min_range_mm=500.0, dt_lo=0.05, dt_hi=0.18,
                  match_m=2.5, search=15, max_speed=15.0):
    """Full 2D object velocity by tracking car clusters to a neighbour scan (~100ms away).
    Returns list of (cx, cy, vobj_ground_x, vobj_ground_y, radius) in frame `scan_idx`'s lidar frame.
    Naive centroid-diff gives v_REL (object vel in the ego frame); we add v_ego so motion_compensate's
    vrel = vobj - vego recovers v_rel. Ego ROTATION between frames is not compensated (IMU handles the
    in-sweep rotation; cross-frame turn adds minor error on the few turning+moving frames)."""
    xyz0, t0, _ = _load_raw(src, scan_idx, min_range_mm)
    cl0 = car_clusters(xyz0, t0)
    if not cl0:
        return []
    base_ts = np.median([c[1] for c in cl0])
    # pick one neighbour scan whose clusters are ~dt away (per-point sensor ts, robust to glitchy indexing)
    nb = None
    for k in list(range(scan_idx+1, scan_idx+search)) + list(range(scan_idx-1, scan_idx-search, -1)):
        try:
            x1, t1, _ = _load_raw(src, k, min_range_mm)
        except Exception:
            continue
        c1 = car_clusters(x1, t1)
        if not c1:
            continue
        dt = abs(np.median([c[1] for c in c1]) - base_ts) * 1e-9
        if dt_lo <= dt <= dt_hi:
            nb = c1; break
    if nb is None:
        return []
    vego = np.asarray(ego_vel_lidar)
    objs = []
    for (c0, tc0, n0) in cl0:
        best = None
        for (c1, tc1, n1) in nb:
            d = float(np.hypot(*(c1 - c0))); dtc = (tc1 - tc0) * 1e-9
            if abs(dtc) < 1e-3: continue
            if d < match_m and (best is None or d < best[0]):
                best = (d, (c1 - c0) / dtc)                 # apparent = v_rel (ego frame)
        if best is not None and np.hypot(*best[1]) <= max_speed:   # gate implausible (mis-associated) matches
            vobj_ground = best[1] + vego                    # -> motion_compensate recovers vrel = vobj-vego
            objs.append((float(c0[0]), float(c0[1]), float(vobj_ground[0]), float(vobj_ground[1]), 2.5))
    return objs


def motion_compensate(xyz, tpt, ego_vel_lidar, movers=None, obj_radius=2.5, az_ref=5.0,
                      tref_sensor=None, omega_z=0.0, objects=None):
    """GENERAL per-point motion compensation (handles ego + per-object; stationary -> v=0 no-op).

    Each point is captured at t_i (rotating sweep). Shift it to t_ref by its RELATIVE velocity:
        p_corr = p - (v_obj - v_ego) * (t_i - t_ref).
      - ego_vel_lidar (2,): ego velocity in the lidar frame (forward +X), m/s. Static field = -v_ego.
      - movers: list of (range_m, bearing_deg, v_radial_ground); points within obj_radius of a mover
        get v_obj = v_radial*[cos,sin] (radial approx; tangential needs cross-frame tracking).
      - tref_sensor: the reference time in the lidar SENSOR clock (ns). MUST be the CAMERA capture
        instant -- else moving objects are mis-aligned: the sweep hits a given azimuth tens of ms off
        the camera frame (sweep phase + sync gap ~63ms on 19877), and a mover moves that whole time.
        If None, falls back to the forward-column time (only valid when the scene is static)."""
    if tref_sensor is not None:
        tref = float(tref_sensor)
    else:
        az0 = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
        fwd = (np.abs(az0) < az_ref) & (xyz[:, 0] > 0)
        tref = np.median(tpt[fwd]) if fwd.any() else np.median(tpt)
    dt = (tpt - tref) * 1e-9                                 # s, signed
    vobj = np.zeros((len(xyz), 2))                           # per-point object ground velocity (default static)
    if movers:                                               # radar radial (coarse, approximate)
        for (R, th, vr) in movers:
            thr = np.radians(th); mx, my = R*np.cos(thr), R*np.sin(thr)
            near = np.hypot(xyz[:, 0]-mx, xyz[:, 1]-my) < obj_radius
            vobj[near, 0] = vr*np.cos(thr); vobj[near, 1] = vr*np.sin(thr)
    if objects:                                              # cross-frame tracked full 2D velocity (preferred)
        for (cxo, cyo, vx, vy, rad) in objects:
            near = np.hypot(xyz[:, 0]-cxo, xyz[:, 1]-cyo) < rad
            vobj[near, 0] = vx; vobj[near, 1] = vy
    vrel = vobj - np.asarray(ego_vel_lidar)[None, :]         # (N,2) relative velocity in ego frame
    out = xyz.copy()
    out[:, :2] -= vrel * dt[:, None]                         # p - vrel*(t_i - t_ref): translation term
    if omega_z:                                              # IMU rotational term: de-rotate the sweep
        th = omega_z * dt                                    # ego yaw from t_ref to t_i (rad), per point
        c, s = np.cos(th), np.sin(th)
        x, y = out[:, 0].copy(), out[:, 1].copy()
        out[:, 0] = c * x - s * y                            # R_z(theta_i) @ p  (brings point back to t_ref)
        out[:, 1] = s * x + c * y
    return out


def _load_raw(src, scan_idx, min_range_mm):
    from lib.lidar_io import scan_to_xyz, scan_range_flat, _scan_from_scan_set
    s = _scan_from_scan_set(src.src[int(scan_idx)])
    xyz = scan_to_xyz(src.meta, s).reshape(-1, 3)
    rng = scan_range_flat(s).reshape(-1)
    tcol = np.asarray(s.timestamp, dtype=np.int64)
    tpt = tcol[np.arange(xyz.shape[0]) % tcol.shape[0]].astype(np.float64)
    v = np.isfinite(xyz).all(1) & (rng >= min_range_mm) & (tpt > 0)
    tmin = float(tcol[tcol > 0].min()) if (tcol > 0).any() else 0.0   # sweep start (sensor ns)
    return xyz[v], tpt[v], tmin


def camera_tref_sensor(scan_idx, tmin, csv_path=None):
    """Reference time (lidar SENSOR clock, ns) corresponding to the CAMERA capture instant.
    Maps via the sync table assuming lidar_t(epoch) <-> sweep-start (tmin):
        tref_sensor = tmin + (camera_t - lidar_t) * 1e9 .  Returns None if the row is missing."""
    import csv as _csv
    from common import CSV
    for r in _csv.DictReader(open(csv_path or CSV)):
        if int(r["lidar_idx"]) == int(scan_idx):
            return tmin + (float(r["camera_t"]) - float(r["lidar_t"])) * 1e9
    return None


def load_xyz_deskewed(src, scan_idx, ego_row, min_range_mm=500.0, max_xy=45.0, min_speed=0.3):
    """Ego-only deskew (static-world assumption). Kept for callers that have no radar handle."""
    xyz, tpt, _ = _load_raw(src, scan_idx, min_range_mm)
    if ego_row is not None and ego_row[2] > min_speed:
        xyz = deskew_xyz(xyz, tpt, np.array([-ego_row[0], -ego_row[1]]))
    return xyz[np.hypot(xyz[:, 0], xyz[:, 1]) < max_xy]


def load_xyz_motioncomp(src, scan_idx, ego_row, rdet=None, radar_idx=None, R2L=None,
                        min_range_mm=500.0, max_xy=45.0):
    """GENERAL motion compensation for one scan: ego (static field) + per-object (radar movers),
    referenced to the CAMERA capture instant. Stationary ego and/or static scene -> ~no-op.
    NOTE: object velocity is radar-radial only -- close cars that aren't clean radar movers, or
    tangential (crossing) motion, need cross-frame cluster tracking (not wired in here)."""
    xyz, tpt, tmin = _load_raw(src, scan_idx, min_range_mm)
    vego = np.array([-ego_row[0], -ego_row[1]]) if ego_row is not None else np.zeros(2)
    movers = radar_movers_lidar(rdet, radar_idx, R2L) if (rdet is not None and radar_idx is not None) else None
    tref = camera_tref_sensor(scan_idx, tmin)
    global _IMU
    try:
        _IMU
    except NameError:
        _IMU = load_imu()
    wz = ego_yaw_rate(_IMU, tmin, float(tpt.max())) if _IMU is not None else 0.0   # IMU ego rotation
    objects = track_objects(src, scan_idx, vego)             # cross-frame full 2D object velocities
    xyz = motion_compensate(xyz, tpt, vego, movers, tref_sensor=tref, omega_z=wz, objects=objects)
    return xyz[np.hypot(xyz[:, 0], xyz[:, 1]) < max_xy]
