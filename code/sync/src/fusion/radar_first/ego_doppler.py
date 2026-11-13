"""Phase 2 — unaliased radar motion: ego-Doppler gate + range-rate.

Static detections obey v_radial = a*cos(az) + b*sin(az) (the radar's own motion
projected onto each line of sight). A RANSAC fit of that line:
  • estimates ego velocity from the radar alone (speed = sqrt(a^2+b^2)),
  • labels inliers = static clutter, outliers = candidate movers.
We cross-validate the radar ego-speed against the Phase-1 LiDAR odometry — if they
agree, both the detector and the odometry are trustworthy.

Single-frame Doppler is aliased (v_max ~ +/-2.4 m/s), so true mover speed must come
from cross-frame range-rate; this module also reports that for tracked movers.
"""
import numpy as np


def fit_ego_velocity(az_deg, vel, thr=0.4, iters=300, seed=0):
    """RANSAC line fit v = a*cos(azr) + b*sin(azr).  Returns (a,b), inlier_mask."""
    azr = np.radians(-np.asarray(az_deg))
    c, s = np.cos(azr), np.sin(azr)
    A = np.c_[c, s]; v = np.asarray(vel)
    n = len(v)
    if n < 3:
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    best = (None, -1, None)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        M = np.array([[c[i], s[i]], [c[j], s[j]]])
        if abs(np.linalg.det(M)) < 1e-6:
            continue
        ab = np.linalg.solve(M, v[[i, j]])
        inl = np.abs(v - A @ ab) < thr
        if inl.sum() > best[1]:
            best = (ab, inl.sum(), inl)
    ab, _, inl = best
    if ab is not None and inl.sum() >= 2:
        ab = np.linalg.lstsq(A[inl], v[inl], rcond=None)[0]
    return ab, inl


def _main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import LidarSource, load_pairs, OUTDIR
    from detector import RadarDetector
    from odometry import odometry_scan2map

    P0, NP = 21000, 100
    pairs = load_pairs()
    ids = [p for p in range(P0, P0 + NP) if p in pairs]
    t = np.array([pairs[p]["radar_t"] for p in ids]); t -= t[0]

    # --- radar: detect (legacy processing keeps static returns for the ego fit) ---
    det = RadarDetector(processing="legacy")
    frames, ego_rad, inl_frac = [], [], []
    for p in ids:
        d, dres = det.detect(pairs[p]["radar_idx"])
        frames.append(d)
        if len(d) >= 4:
            ab, inl = fit_ego_velocity(d["az"], d["vel"])
            ego_rad.append(np.hypot(*ab) if ab is not None else np.nan)
            inl_frac.append(inl.mean() if ab is not None else np.nan)
        else:
            ego_rad.append(np.nan); inl_frac.append(np.nan)
    ego_rad = np.array(ego_rad)
    print(f"radar detections/frame: median {np.median([len(f) for f in frames]):.0f}, "
          f"dres={dres:.3f} m/s, v_max=+/-{len(frames) and dres*0:.1f}", flush=True)

    # --- lidar odometry ego-speed over the same window (Phase 1) ---
    src = LidarSource()
    scans = [src.xyz(pairs[p]["lidar_idx"]) for p in ids]
    poses, _ = odometry_scan2map(scans)
    traj = np.array([pp[:3, 3] for pp in poses])
    dt = np.diff(t); dt[dt <= 0] = np.median(dt[dt > 0])
    ego_odo = np.r_[np.nan, np.linalg.norm(np.diff(traj, axis=0), axis=1) / dt]

    # --- compare ---
    m = np.isfinite(ego_rad) & np.isfinite(ego_odo)
    med_r, med_o = np.nanmedian(ego_rad), np.nanmedian(ego_odo)
    corr = np.corrcoef(ego_rad[m], ego_odo[m])[0, 1] if m.sum() > 3 else np.nan

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    # (a) one representative frame: vel vs azimuth with fitted static line
    k = int(np.nanargmax([len(f) for f in frames]))
    d = frames[k]; ab, inl = fit_ego_velocity(d["az"], d["vel"])
    azr = np.radians(-d["az"])
    order = np.argsort(azr)
    axs[0].scatter(np.degrees(azr[~inl]), d["vel"][~inl], c="crimson", s=36, label="mover (outlier)")
    axs[0].scatter(np.degrees(azr[inl]), d["vel"][inl], c="steelblue", s=36, label="static (inlier)")
    if ab is not None:
        axs[0].plot(np.degrees(azr[order]), (np.c_[np.cos(azr), np.sin(azr)] @ ab)[order],
                    "k--", label=f"ego fit (|v|={np.hypot(*ab):.2f} m/s)")
    axs[0].set_xlabel("azimuth (deg)"); axs[0].set_ylabel("radial velocity (m/s)")
    axs[0].set_title(f"Ego-Doppler fit · pair {ids[k]}\nstatic = line, movers = outliers"); axs[0].legend()

    # (b) ego speed: radar vs lidar odometry over the window
    axs[1].plot(t, ego_rad, "-o", ms=3, c="teal", label="radar ego-speed (RANSAC)")
    axs[1].plot(t, ego_odo, "-s", ms=3, c="darkorange", label="LiDAR odometry ego-speed")
    axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("ego speed (m/s)")
    axs[1].set_title(f"Radar vs LiDAR ego-speed\nmedian radar={med_r:.2f}  odo={med_o:.2f}  corr={corr:.2f}")
    axs[1].legend()
    plt.tight_layout()
    out = f"{OUTDIR}/phase2_ego_doppler.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n--- Phase 2 ego-Doppler ---", flush=True)
    print(f"median static-inlier fraction: {np.nanmedian(inl_frac):.2f}", flush=True)
    print(f"ego-speed  radar={med_r:.2f} m/s   lidar-odometry={med_o:.2f} m/s   corr={corr:.2f}", flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    _main()
