"""Phase 1 — yaw-accurate LiDAR odometry (scan-to-MAP ICP).

The old pipeline used scan-to-scan ICP: each frame registered only to the previous
one, so small per-frame yaw errors accumulate and static structure reads as motion.
Here we register each scan against an accumulated *local map* (a sliding window of
recent scans in the world frame), which corrects incremental drift.

Run as a script to compare scan-to-scan vs scan-to-map on a window:
    python3 sync/src/fusion/radar_first/odometry.py
Produces a BEV comparison (ghosting = drift) + scalar drift metrics.
"""
import numpy as np
import open3d as o3d

VOX = 0.25           # per-scan ICP downsample (m)
MAP_VOX = 0.30       # local-map voxel (m)
MAP_WINDOW = 18      # frames kept in the local map
ICP_DIST = 0.6       # ICP correspondence distance (m)
ICP_ITERS = 50


def _pcd(a, vox=VOX):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(a, dtype=np.float64))
    if vox:
        p = p.voxel_down_sample(vox)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 3, max_nn=30))
    return p


def _icp(src, tgt, init):
    return o3d.pipelines.registration.registration_icp(
        src, tgt, ICP_DIST, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_ITERS))


def odometry_scan2scan(scans):
    """Baseline: register frame i to frame i-1."""
    pcds = [_pcd(a) for a in scans]
    poses = [np.eye(4)]; fit = []
    for i in range(1, len(pcds)):
        reg = _icp(pcds[i], pcds[i - 1], np.eye(4))
        poses.append(poses[-1] @ reg.transformation)
        fit.append(reg.fitness)
    return poses, fit


MAX_STEP = 2.0       # reject a per-frame translation jump larger than this (m) -> drift/ICP blowup
WARMUP = 3           # seed the local map with scan-to-scan before trusting scan-to-map


def odometry_scan2map(scans):
    """Scan-to-map ICP with startup warm-up + per-frame jump gating.

    - WARMUP frames use scan-to-scan to build a stable initial map (scan-to-map against a
      1-frame map is ill-posed and was throwing ~240 m/s startup spikes).
    - Any registration whose translation jumps > MAX_STEP in one frame is rejected and
      replaced by the constant-velocity prediction, so a single bad ICP can't poison the trajectory.
    """
    pcds = [_pcd(a) for a in scans]
    poses = [np.eye(4)]; fit = []
    map_pts = [np.asarray(pcds[0].points)]
    map_fid = [0]
    rel_prev = np.eye(4)
    for i in range(1, len(pcds)):
        init = poses[-1] @ rel_prev                  # constant-velocity extrapolation
        if i <= WARMUP:
            reg = _icp(pcds[i], pcds[i - 1], np.eye(4))  # scan-to-scan while the map is thin
            pose = poses[-1] @ reg.transformation
            f = reg.fitness
        else:
            keep = [k for k, fr in enumerate(map_fid) if fr >= i - MAP_WINDOW]
            tgt = _pcd(np.vstack([map_pts[k] for k in keep]), vox=MAP_VOX)
            reg = _icp(pcds[i], tgt, init)           # T maps scan_i -> world
            pose = reg.transformation
            f = reg.fitness
        # jump gate: reject implausible per-frame translation -> fall back to prediction
        step = float(np.linalg.norm(pose[:3, 3] - poses[-1][:3, 3]))
        if step > MAX_STEP:
            pose = init; f = 0.0
        rel_prev = np.linalg.inv(poses[-1]) @ pose
        poses.append(pose); fit.append(f)
        h = np.c_[np.asarray(pcds[i].points), np.ones(len(pcds[i].points))]
        map_pts.append((pose @ h.T).T[:, :3]); map_fid.append(i)
        cut = [k for k, fr in enumerate(map_fid) if fr >= i - MAP_WINDOW + 1]
        map_pts = [map_pts[k] for k in cut]; map_fid = [map_fid[k] for k in cut]
    return poses, fit


def accumulate(scans, poses, sub=3000):
    """World cloud (subsampled) + per-point frame id, for metrics/plots."""
    W, F = [], []
    for i, a in enumerate(scans):
        step = max(1, len(a) // sub); s = a[::step]
        h = np.c_[s, np.ones(len(s))]
        W.append((poses[i] @ h.T).T[:, :3]); F.append(np.full(len(s), i))
    return np.vstack(W), np.concatenate(F)


def bev_occupancy_entropy(W, cell=0.1, zmin=0.3, zmax=3.0):
    """Drift proxy: normalized entropy of the BEV occupancy of static-height points.
    Sharp (low-drift) maps concentrate repeated structure in few cells -> lower entropy."""
    m = (W[:, 2] > zmin) & (W[:, 2] < zmax) & (np.linalg.norm(W[:, :2], axis=1) < 30)
    P = W[m][:, :2]
    if len(P) < 100:
        return float("nan")
    k = np.floor(P / cell).astype(np.int64)
    _, counts = np.unique(k, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))  # normalized 0..1


def _main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import LidarSource, load_pairs, OUTDIR
    P0, NP = 21000, 150
    pairs = load_pairs()
    idxs = [pairs[p]["lidar_idx"] for p in range(P0, P0 + NP) if p in pairs]
    print(f"window: {len(idxs)} consecutive scans [{idxs[0]}..{idxs[-1]}]", flush=True)
    src = LidarSource()
    scans = []
    for j, li in enumerate(idxs):
        scans.append(src.xyz(li))
        if j % 40 == 0: print(f"  loaded {j}/{len(idxs)}", flush=True)

    print("scan-to-scan ...", flush=True)
    ps_ss, fit_ss = odometry_scan2scan(scans)
    print("scan-to-map ...", flush=True)
    ps_sm, fit_sm = odometry_scan2map(scans)

    Wss, Fss = accumulate(scans, ps_ss)
    Wsm, Fsm = accumulate(scans, ps_sm)
    e_ss = bev_occupancy_entropy(Wss); e_sm = bev_occupancy_entropy(Wsm)
    len_ss = float(np.sum(np.linalg.norm(np.diff([p[:3, 3] for p in ps_ss], axis=0), axis=1)))
    len_sm = float(np.sum(np.linalg.norm(np.diff([p[:3, 3] for p in ps_sm], axis=0), axis=1)))

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    for ax, W, ttl, e in ((axs[0], Wss, "scan-to-scan (old)", e_ss),
                          (axs[1], Wsm, "scan-to-MAP (new)", e_sm)):
        m = np.linalg.norm(W[:, :2], axis=1) < 30
        ax.scatter(W[m, 1], W[m, 0], s=0.2, c="0.5")
        ax.set_aspect("equal"); ax.invert_xaxis()
        ax.set_title(f"{ttl}\naccumulated BEV · occupancy-entropy {e:.3f} (lower=crisper)")
        ax.set_xlabel("world Y (m)"); ax.set_ylabel("world X (m)")
    plt.tight_layout()
    out = f"{OUTDIR}/phase1_odometry_compare.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n--- Phase 1 odometry ---", flush=True)
    print(f"median ICP fitness   scan2scan={np.median(fit_ss):.3f}  scan2map={np.median(fit_sm):.3f}", flush=True)
    print(f"BEV occ. entropy     scan2scan={e_ss:.3f}  scan2map={e_sm:.3f}  (lower=less drift)", flush=True)
    print(f"trajectory length    scan2scan={len_ss:.1f} m  scan2map={len_sm:.1f} m", flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    _main()
