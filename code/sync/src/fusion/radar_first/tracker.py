"""Phase 3 — multi-target tracker for radar-first movers.

Pipeline per frame:
  detect (range-Doppler CFAR) -> ego-Doppler gate drops static clutter ->
  remaining mover detections are lifted to the WORLD frame (radar->LiDAR extrinsic,
  then LiDAR odometry pose) -> constant-velocity Kalman tracker with gating,
  greedy association, and M-of-N confirmation.

Why this fixes the old failure: static objects are removed BEFORE tracking (ego-Doppler
inliers) AND jitter can't survive M-of-N confirmation, so a traffic light no longer
becomes a "mover". True speed per track comes from cross-frame range-rate (unaliased),
not the aliased single-frame Doppler.
"""
import numpy as np

# constant-velocity Kalman in world XY: state [x, y, vx, vy]
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
GATE = 2.5          # association gate (m)
CONFIRM_HITS = 3    # hits needed to confirm a track
MAX_MISS = 3        # consecutive misses before deletion
R_MEAS = np.diag([0.8, 0.8]) ** 2     # radar position noise (m)
# Phase 4 gates (prune the clutter that flooded Phase 3)
MIN_EGO_INLIERS = 6   # don't trust the static/mover split if the ego fit is weak
MOVER_MARGIN = 0.8    # a mover must sit this far (m/s) off the ego-static line
MIN_HITS = 5          # minimum track length to be a real mover (kills fragments)
SUPPORT_THR = 0.4     # fraction of frames needing nearby LiDAR-dynamic corroboration
RR_TOL = 1.2          # |measured range-rate - geometry-predicted range-rate| must be below this


def _F(dt): return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
def _Q(dt, q=4.0):
    # discrete white-noise accel
    return q * np.array([[dt**4/4, 0, dt**3/2, 0], [0, dt**4/4, 0, dt**3/2],
                         [dt**3/2, 0, dt**2, 0], [0, dt**3/2, 0, dt**2]], float)


class Track:
    _n = 0
    def __init__(self, xy, frame, meta):
        self.x = np.array([xy[0], xy[1], 0, 0], float)
        self.P = np.diag([1, 1, 25, 25]).astype(float)
        self.hits = 1; self.misses = 0; self.confirmed = False
        self.id = Track._n; Track._n += 1
        self.hist = [(frame, xy[0], xy[1], *meta)]   # frame,x,y,range,vel,az
    def predict(self, dt):
        F = _F(dt); self.x = F @ self.x; self.P = F @ self.P @ F.T + _Q(dt)
    def update(self, xy, frame, meta):
        S = H @ self.P @ H.T + R_MEAS
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(xy) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1; self.misses = 0
        if self.hits >= CONFIRM_HITS: self.confirmed = True
        self.hist.append((frame, self.x[0], self.x[1], *meta))
    @property
    def speed(self): return float(np.hypot(self.x[2], self.x[3]))


def run_tracker(frames_world, times):
    """frames_world: list per frame of Nx5 arrays [x,y,range,vel,az] (world xy). times: radar_t."""
    tracks, done = [], []
    for fi, (dets, t) in enumerate(zip(frames_world, times)):
        dt = (t - times[fi - 1]) if fi > 0 else 0.1
        if dt <= 0: dt = 0.1
        for tr in tracks: tr.predict(dt)
        used = set()
        # greedy nearest association
        for tr in tracks:
            if not len(dets): break
            d = np.hypot(dets[:, 0] - tr.x[0], dets[:, 1] - tr.x[1])
            j = int(np.argmin(d))
            if d[j] <= GATE and j not in used:
                used.add(j); tr.update(dets[j, :2], fi, dets[j, 2:5])
            else:
                tr.misses += 1
        for j in range(len(dets)):
            if j not in used:
                tracks.append(Track(dets[j, :2], fi, dets[j, 2:5]))
        keep = []
        for tr in tracks:
            (done if tr.misses > MAX_MISS else keep).append(tr)
        tracks = keep
    done += tracks
    return done


def range_rate(tr, times):
    """Unaliased radial speed = slope of radar-frame range vs time over the track."""
    h = np.array(tr.hist)
    if len(h) < 4: return np.nan
    tt = times[h[:, 0].astype(int)]; rng = h[:, 3]
    return float(np.polyfit(tt, rng, 1)[0])


def predicted_range_rate(tr, poses, ego_vel):
    """Range-rate expected from geometry: (v_obj_world - v_ego_world) . line-of-sight.
    Should match the measured range-rate for a real rigid mover; clutter won't."""
    h = np.array(tr.hist)
    vobj = np.array([tr.x[2], tr.x[3]])
    preds = []
    for f, x, y in h[:, :3]:
        f = int(f)
        u = np.array([x, y]) - poses[f][:3, 3][:2]
        n = np.linalg.norm(u)
        if n < 1e-3: continue
        preds.append(float((vobj - ego_vel[f]) @ (u / n)))
    return float(np.mean(preds)) if preds else np.nan


def track_movers(ids, pairs):
    """Full radar-first mover pipeline on a window -> dict with the confirmed `movers`
    (+ poses, times, scans, R2L, funnel stats). Reused by _main and the projection tool."""
    from common import LidarSource, load_calib, pose_apply
    from detector import RadarDetector
    from odometry import odometry_scan2map
    from ego_doppler import fit_ego_velocity
    from lidar_dynamic import dynamic_points_per_frame, support_fraction

    times = np.array([pairs[p]["radar_t"] for p in ids]); times -= times[0]
    R2L = np.array(load_calib()["radar_to_lidar_auto"])
    src = LidarSource()
    scans = [src.xyz(pairs[p]["lidar_idx"]) for p in ids]
    poses, _ = odometry_scan2map(scans)
    traj = np.array([pp[:3, 3][:2] for pp in poses])
    dtv = np.r_[0.1, np.diff(times)]; dtv[dtv <= 0] = 0.1
    ego_vel = np.zeros((len(poses), 2))
    ego_vel[1:] = np.diff(traj, axis=0) / dtv[1:, None]; ego_vel[0] = ego_vel[1]

    det = RadarDetector(processing="legacy")
    frames_world, n_det, n_mov = [], [], []
    for fi, p in enumerate(ids):
        d, _ = det.detect(pairs[p]["radar_idx"]); n_det.append(len(d))
        mov = d[:0]
        if len(d) >= MIN_EGO_INLIERS:
            ab, inl = fit_ego_velocity(d["az"], d["vel"])
            if ab is not None and inl.sum() >= MIN_EGO_INLIERS:
                azr_all = np.radians(-d["az"])
                pred = np.cos(azr_all) * ab[0] + np.sin(azr_all) * ab[1]
                mov = d[(~inl) & (np.abs(d["vel"] - pred) > MOVER_MARGIN)]
        n_mov.append(len(mov))
        if not len(mov):
            frames_world.append(np.zeros((0, 5))); continue
        azr = np.radians(-mov["az"]); Xr = mov["range"] * np.cos(azr); Yr = mov["range"] * np.sin(azr)
        lf = R2L[0, 0]*Xr + R2L[0, 1]*Yr + R2L[0, 3]
        ll = R2L[1, 0]*Xr + R2L[1, 1]*Yr + R2L[1, 3]
        wx = pose_apply(poses[fi], np.c_[lf, ll])
        frames_world.append(np.c_[wx[:, 0], wx[:, 1], mov["range"], mov["vel"], mov["az"]])

    tracks = run_tracker(frames_world, times)
    conf = [t for t in tracks if t.confirmed and len(t.hist) >= CONFIRM_HITS]
    dyn_pf = dynamic_points_per_frame(scans, poses)
    movers = []
    for t in conf:
        h = np.array(t.hist); net = np.hypot(h[-1, 1]-h[0, 1], h[-1, 2]-h[0, 2])
        if len(t.hist) < MIN_HITS or t.speed < 0.5 or net < 1.0:
            continue
        t.support = support_fraction(t.hist, dyn_pf)
        if t.support < SUPPORT_THR:
            continue
        t.rr_meas = range_rate(t, times); t.rr_pred = predicted_range_rate(t, poses, ego_vel)
        if not np.isfinite(t.rr_meas) or abs(t.rr_meas - t.rr_pred) > RR_TOL:
            continue
        movers.append(t)
    return dict(movers=movers, poses=poses, times=times, scans=scans, R2L=R2L,
                tracks=tracks, conf=conf, n_det=n_det, n_mov=n_mov, ids=ids)


def _main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import load_pairs, OUTDIR
    from odometry import accumulate

    P0, NP = 13850, 100          # intersection-with-traffic window (has movers)
    pairs = load_pairs()
    ids = [p for p in range(P0, P0 + NP) if p in pairs]
    R = track_movers(ids, pairs)
    movers, poses, times, scans = R["movers"], R["poses"], R["times"], R["scans"]
    n_det, n_mov, tracks, conf = R["n_det"], R["n_mov"], R["tracks"], R["conf"]

    # ---- plot ----
    W, _ = accumulate(scans, poses)
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.5))
    m = np.linalg.norm(W[:, :2], axis=1) < 35
    axs[0].scatter(W[m, 1], W[m, 0], s=0.2, c="0.78")
    cmap = plt.cm.tab10
    for i, t in enumerate(movers):
        h = np.array(t.hist)
        axs[0].plot(h[:, 2], h[:, 1], "-", lw=2, color=cmap(i % 10))
        axs[0].annotate("", xy=(h[-1, 2], h[-1, 1]), xytext=(h[0, 2], h[0, 1]),
                        arrowprops=dict(arrowstyle="->", color=cmap(i % 10), lw=2))
        axs[0].text(h[-1, 2], h[-1, 1], f" T{t.id} {t.speed:.1f}m/s", fontsize=9, color=cmap(i % 10))
    axs[0].set_aspect("equal"); axs[0].invert_xaxis()
    axs[0].set_xlabel("world Y (m)"); axs[0].set_ylabel("world X (m)")
    axs[0].set_title(f"Confirmed radar-first movers in world frame\n"
                     f"{len(movers)} movers · static removed before tracking")

    # aliased single-frame Doppler vs unaliased range-rate, for the longest mover
    if movers:
        t = max(movers, key=lambda z: len(z.hist)); h = np.array(t.hist)
        tt = times[h[:, 0].astype(int)]
        axs[1].plot(tt, h[:, 4], "-o", ms=3, c="crimson", label="single-frame Doppler (aliased)")
        rr = range_rate(t, times)
        axs[1].axhline(rr, ls="--", c="teal", label=f"range-rate slope = {rr:+.2f} m/s (unaliased)")
        axs[1].axhline(0.305*8, ls=":", c="0.5"); axs[1].axhline(-0.305*8, ls=":", c="0.5")
        axs[1].set_xlabel("time (s)"); axs[1].set_ylabel("radial velocity (m/s)")
        axs[1].set_title(f"Track T{t.id}: aliased Doppler vs range-rate\n(dotted = +/- v_max)")
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, "no confirmed movers in window", ha="center")
    plt.tight_layout()
    out = f"{OUTDIR}/phase3_tracker.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")

    print(f"\n--- Phase 3 tracker ---", flush=True)
    print(f"detections/frame median {np.median(n_det):.0f}; mover-candidates/frame median {np.median(n_mov):.0f}", flush=True)
    print(f"tracks total {len(tracks)}; confirmed {len(conf)}; genuine movers {len(movers)}", flush=True)
    for t in sorted(movers, key=lambda z: -z.speed):
        print(f"  T{t.id}: {len(t.hist)} hits, speed {t.speed:.2f} m/s, "
              f"rr meas {t.rr_meas:+.2f} / pred {t.rr_pred:+.2f} m/s, lidar-support {t.support:.2f}", flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    _main()
