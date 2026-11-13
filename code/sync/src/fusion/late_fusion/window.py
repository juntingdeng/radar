"""Temporal-memory WINDOW pipeline -- one file, subcommands (merged from window_odometry /
window_detect / window_memory / window_compare / memory_codec).

  python window.py odometry <win> [viz]       lidar odometry: point-to-line ICP + deskew + keyframes
                                              + validation suite V1-V7 -> <win>/odometry.npz
  python window.py detect   <win>             3 frozen detectors per frame -> proposals_window.pkl
  python window.py memory   <win> [viz] [lidar1hz]   world-frame belief bank (radar-certified cam
                                              writes) + locked-metric report (+ M7 flipbook)
  python window.py compare  <win>             6 sensor suites through the identical bank
                                              (WC_VIZ=<suite> renders the M8 flipbook)
  python window.py codec    <win> [jscc|qam|awgn]    belief codec: compression table, bit-flip,
                                              fixed-16QAM ablation, analog AWGN channel

Windows live under ~/CMU/Research/mobile_data_local/multiview_v1/0508/win_*; each needs
sensors.npz (extraction) first, then odometry -> detect -> memory/compare/codec.
"""
import os, sys, json, csv, pickle, numpy as np, torch, torch.nn as nn
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.manual_seed(0); np.random.seed(0)
MV = os.path.expanduser("~/CMU/Research/mobile_data_local/multiview_v1/0508")

def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])

def radius_of(lb): return 1.0 if lb in ("person", "bicycle", "motorcycle") else 2.0

def ctr(b): return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

# ===================== ODOMETRY (was window_odometry.py) =====================

REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
LABDIR = os.path.join(REPO, "sync", "res", "benchmark_v1", "labels")
CAM_HALF_FOV = np.radians(34.0)

# ---------------- scan prep ----------------
def prep(pts, w=0.0, v=0.0, sign=1.0, cut=np.pi):
    """raw xyz (f16) -> deskewed, ground-removed, slab-cropped, 0.3 m-voxeled 2D points (f32).
    Deskew: spin phase from azimuth (cut = scan seam angle); point time = (phase-0.5)*0.1 s around
    the scan midpoint; un-rotate by w*dt, un-translate by v*dt (ego frame, forward=x)."""
    p = pts.astype(np.float32)
    r = np.hypot(p[:, 0], p[:, 1])
    m = (r < 32) & (r > 1.0)
    p = p[m]
    if w != 0.0 or v != 0.0:
        ph = np.mod(np.arctan2(p[:, 1], p[:, 0]) - cut, 2 * np.pi) / (2 * np.pi)
        dt = sign * (ph - 0.5) * 0.1
        c, s = np.cos(-w * dt), np.sin(-w * dt)
        x = c * p[:, 0] - s * p[:, 1] - v * dt
        p = np.column_stack([x, s * p[:, 0] + c * p[:, 1], p[:, 2]])
    zg = np.percentile(p[:, 2], 2)
    p = p[(p[:, 2] > zg + 0.5) & (p[:, 2] < zg + 3.0)][:, :2]
    key = np.round(p / 0.3).astype(np.int32)
    _, idx = np.unique(key, axis=0, return_index=True)
    p = p[idx]
    # planarity filter: keep LINEAR structure (walls, car sides), drop volumetric foliage --
    # diffuse canopy correspondences damp ICP translation (measured as along-path landmark streaks)
    if len(p) > 40:
        T = cKDTree(p)
        d, j = T.query(p, k=7)
        nb = p[j]                                   # (n,7,2)
        c = nb.mean(1, keepdims=True)
        X = nb - c
        cov = np.einsum("nki,nkj->nij", X, X) / 7
        ev = np.linalg.eigvalsh(cov)                # ascending
        lin = ev[:, 0] / np.maximum(ev[:, 1], 1e-9)  # small = line-like
        keep = (lin < 0.12) & (d[:, -1] < 1.2)       # tight neighbourhood, strongly linear
        if keep.sum() > 150: p = p[keep]
    return p

# ---------------- ICP core ----------------
_nrm_cache = {}
def _normals(B):
    """per-point 2D normals of the target cloud via 6-NN PCA (cached by array id)."""
    key = id(B)
    if key not in _nrm_cache:
        T = cKDTree(B)
        _, j = T.query(B, k=6)
        X = B[j] - B[j].mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", X, X)
        w, V = np.linalg.eigh(cov)
        _nrm_cache[key] = (T, V[:, :, 0])           # eigvec of SMALLEST eigenvalue = normal
        if len(_nrm_cache) > 800: _nrm_cache.pop(next(iter(_nrm_cache)))
    return _nrm_cache[key]

def icp(A, B, Rini=None, tini=None, iters=15, trim=0.8):
    """trimmed point-to-LINE 2D ICP mapping A into B's frame (point-to-point damps translation
    on this foliage-heavy plaza -- measured as along-path landmark streaks). Returns dict or None."""
    R = np.eye(2) if Rini is None else Rini.copy()
    t = np.zeros(2) if tini is None else tini.copy()
    TB, NB = _normals(B)
    inl = res = 0.0
    for it in range(iters):
        gate = 2.0 if it < 4 else 1.0 if it < 9 else 0.6
        Aw = A @ R.T + t
        d, j = TB.query(Aw, distance_upper_bound=gate)
        m = np.isfinite(d)
        if m.sum() < 60: return None
        k = max(60, int(trim * m.sum()))
        sel = np.where(m)[0][np.argsort(d[m])[:k]]            # trimmed: best-k residuals
        Pa, Pb, n = Aw[sel], B[j[sel]], NB[j[sel]]
        e = np.einsum("ni,ni->n", Pa - Pb, n)                 # signed point-to-line residuals
        # linearised small-angle solve: unknowns [dth, tx, ty]; J_i = [n_i x p_i_perp, n_i]
        Jth = Pa[:, 0] * n[:, 1] - Pa[:, 1] * n[:, 0]
        J = np.column_stack([Jth, n])
        H = J.T @ J + np.eye(3) * 1e-6
        dx = np.linalg.solve(H, -J.T @ e)
        R = rot(dx[0]) @ R; t = rot(dx[0]) @ t + dx[1:]
        inl = m.mean(); res = float(np.abs(e).mean())
    ev = np.linalg.eigvalsh(np.cov((B[j[sel]]).T))
    cond = float(np.sqrt(max(ev) / max(min(ev), 1e-6)))
    return dict(R=R, t=t, inl=float(inl), res=res, cond=cond)

def yaw_of(R): return np.arctan2(R[1, 0], R[0, 0])
# ---------------- labels (for validation) ----------------
def load_labels():
    lab = {}
    for fn in ("train", "val", "test_clear"):
        fp = os.path.join(LABDIR, fn + ".json")
        if os.path.exists(fp):
            for k, v in json.load(open(fp)).items():
                if k.startswith("0508"): lab.setdefault(k, []).extend(v)
    return lab

def gt_centres(lab, pid):
    sys.path.insert(0, HERE)
    import data_l2
    for d in (0, 1, -1):
        key = f"0508_{pid + d:06d}"
        if key in lab:
            us = data_l2._union(data_l2._gt_for(key, lab))
            return np.array([[(u["box"][0] + u["box"][2]) / 2, (u["box"][1] + u["box"][3]) / 2]
                             for u in us]), us
    return np.zeros((0, 2)), []

# ---------------- main build ----------------
def odo_build(win):
    WD = f"{MV}/{win}"
    z = np.load(f"{WD}/sensors.npz", allow_pickle=True)
    pids, L = z["pids"], z["lidar"]
    N = len(pids)
    print(f"{win}: {N} frames", flush=True)

    # ---- pass A: consecutive ICP, raw scans, CV init ----
    P0 = [prep(x) for x in L]
    dpsiA = np.zeros(N); dxyA = np.zeros((N, 2))
    Rp, tp = np.eye(2), np.zeros(2)
    for i in range(1, N):
        r = icp(P0[i], P0[i - 1], Rp, tp)
        if r is None: r = dict(R=Rp, t=tp, inl=0, res=9, cond=99)
        dpsiA[i] = yaw_of(r["R"]); dxyA[i] = r["t"]; Rp, tp = r["R"], r["t"]
    dtA = np.clip(np.diff(z["lidar_t"], prepend=z["lidar_t"][0] - 0.1), 0.1, 10.0)
    wA = np.convolve(dpsiA / dtA, np.ones(3) / 3, "same")          # rad/s over REAL dt (dropouts!)
    vA = np.convolve(np.hypot(*dxyA.T) / dtA, np.ones(3) / 3, "same")
    print(f"pass A: net yaw {np.degrees(dpsiA.sum()):+.0f} deg, path {np.hypot(*dxyA.T).sum():.1f} m", flush=True)

    # ---- deskew convention self-calibration on the highest-|w| NORMAL-cadence frames ----
    # (gap frames excluded: comparing a scan to one 1.25-2.8 s older is meaningless for deskew)
    hot = [i for i in np.argsort(-np.abs(wA))
           if i > 0 and dtA[i] <= 0.12 and dtA[i - 1] <= 0.12][:4]
    best, btag = None, None
    for sign in (1.0, -1.0):
        for cut in (np.pi, 0.0):
            vals = []
            for i in hot:
                A = prep(L[i], wA[i], vA[i], sign, cut)
                r = icp(A, P0[i - 1], rot(dpsiA[i]), dxyA[i])
                if r is not None: vals.append(r["res"])
            if len(vals) >= 2 and (best is None or np.mean(vals) < best):
                best, btag = float(np.mean(vals)), (sign, cut)
    sign, cut = btag
    print(f"deskew convention: sign={sign:+.0f} cut={cut:.2f} "
          f"(residual {best:.3f} m over {len(hot)} turn frames)", flush=True)

    # ---- pass B: deskewed scans, keyframe registration ----
    DS = [prep(L[i], wA[i], vA[i], sign, cut) if abs(wA[i]) > np.radians(3) or vA[i] > 0.3
          else P0[i] for i in range(N)]
    # real inter-scan time: pcap dropouts leave multi-second gaps that the sequential scan index
    # hides (measured: 2.84 s at one frame + truncated 7k-pt scans) -- gates must be VELOCITY gates
    dt = dtA
    VMAX, WMAX = 3.5, np.radians(80)                           # cart physical limits
    poses = np.zeros((N, 3))                                   # x, y, psi (frame 0 = identity)
    kf, kf_ids = 0, [0]
    Rk, tk = np.eye(2), np.zeros(2)                            # current scan in KEYFRAME frame
    inl = np.ones(N); res = np.zeros(N); cond = np.ones(N); bad = np.zeros(N, bool)
    def plaus(dR, dtr, i): return np.hypot(*dtr) < VMAX * dt[i] and abs(yaw_of(dR)) < WMAX * dt[i]
    force_kf = False
    for i in range(1, N):
        Rk0, tk0 = Rk, tk                                      # T_{kf<-i-1}, kept for recovery
        # prediction from pass-A RATES x this step's real duration
        Rpred = rot(wA[i] * dt[i])
        nv = np.hypot(*dxyA[i])
        tpred = (dxyA[i] / nv * vA[i] * dt[i]) if nv > 1e-6 else dxyA[i]
        used = None
        r = icp(DS[i], DS[kf], Rk @ Rpred, Rk @ tpred + tk)    # T_{kf<-i} prediction
        if r is not None and r["inl"] >= 0.55 and plaus(Rk0.T @ r["R"], Rk0.T @ (r["t"] - tk0), i):
            Rk, tk, used = r["R"], r["t"], r
        else:
            r2 = icp(DS[i], DS[i - 1], Rpred, tpred)           # ladder 1: consecutive frame
            if r2 is not None and r2["inl"] >= 0.5 and plaus(r2["R"], r2["t"], i):
                Rk, tk, used = Rk0 @ r2["R"], Rk0 @ r2["t"] + tk0, r2
            else:                                              # ladder 2: constant velocity x dt
                Rk, tk = Rk0 @ Rpred, Rk0 @ tpred + tk0
                bad[i] = True
            force_kf = True                                    # weak kf anchor -> re-anchor asap
        inl[i], res[i], cond[i] = (used["inl"], used["res"], used["cond"]) if used else (0, 0, 1)
        pk = poses[kf]
        pw = rot(pk[2]) @ tk + pk[:2]
        poses[i] = [pw[0], pw[1], pk[2] + yaw_of(Rk)]
        # promotion judged on the registration actually used (no stale-anchor deadlock)
        if used is not None and used["inl"] >= 0.5 and \
           (force_kf or np.hypot(*tk) > 2.5 or abs(yaw_of(Rk)) > np.radians(12)):
            kf = i; kf_ids.append(i); Rk, tk = np.eye(2), np.zeros(2); force_kf = False
    # SE(2)-interpolate across bad runs: CV poses there carry no measurement -- replace with the
    # geodesic between the bounding good frames (also distributes any re-anchor snap)
    gi = np.where(~bad)[0]
    for i in np.where(bad)[0]:
        a = gi[gi < i].max() if (gi < i).any() else 0
        b = gi[gi > i].min() if (gi > i).any() else N - 1
        if a == b: continue
        f = (z["lidar_t"][i] - z["lidar_t"][a]) / max(z["lidar_t"][b] - z["lidar_t"][a], 1e-6)
        dpsi = (poses[b, 2] - poses[a, 2] + np.pi) % (2 * np.pi) - np.pi
        poses[i] = [poses[a, 0] + f * (poses[b, 0] - poses[a, 0]),
                    poses[a, 1] + f * (poses[b, 1] - poses[a, 1]), poses[a, 2] + f * dpsi]
    if bad.any():
        print(f"pass B: {bad.sum()} frames bridged by interpolation at t={np.where(bad)[0]/10}",
              flush=True)
    w = np.gradient(np.unwrap(poses[:, 2])) / dt
    v = np.hypot(*np.gradient(poses[:, :2], axis=0).T) / dt
    print(f"pass B: net yaw {np.degrees(poses[-1,2]):+.0f} deg, path "
          f"{np.sum(np.hypot(*np.diff(poses[:,:2],axis=0).T)):.1f} m, {len(kf_ids)} keyframes", flush=True)

    # ---- validation 1: static-landmark world spread ----
    lab = load_labels()
    tracks = []                                                # [world_xy per obs]
    for i in range(N):
        C, _ = gt_centres(lab, int(pids[i]))
        if not len(C): continue
        Cw = C @ rot(poses[i, 2]).T + poses[i, :2]
        for c in Cw:
            tr = next((t_ for t_ in tracks if np.hypot(*(t_[-1] - c)) < 1.5), None)
            if tr is None: tracks.append([c])
            else: tr.append(c)
    # robust per-track statistic: label jitter (+-0.3 m) makes total RANGE grow with n-obs, so the
    # criterion is the SYSTEMATIC part -- first-third -> last-third mean drift (jitter averages out)
    drifts, stds = [], []
    for tr in tracks:
        if len(tr) < 60: continue
        T = np.array(tr)
        if np.hypot(*(T[-1] - T[0])) > 2.0: continue           # mover -> excluded from criterion
        k = len(T) // 3
        drifts.append(float(np.hypot(*(T[-k:].mean(0) - T[:k].mean(0)))))
        stds.append(float(np.hypot(*(T - T.mean(0)).T).std()))
    med = np.median(drifts) if drifts else np.nan
    print(f"validation: {len(drifts)} static landmark tracks, systematic world drift "
          f"median {med:.2f} m  max {max(drifts) if drifts else np.nan:.2f} m, "
          f"per-obs std median {np.median(stds) if stds else np.nan:.2f} m "
          f"{'PASS' if med < 0.5 else 'FAIL'} (<0.5 m; cross-turn consistency judged on V1)", flush=True)

    # ---- validation 2: label-fit vs ICP speed (1-s baseline, in ICP-posed frame) ----
    diffs = []
    for i in range(10, N, 10):
        C2, _ = gt_centres(lab, int(pids[i])); C1, _ = gt_centres(lab, int(pids[i - 10]))
        if len(C1) < 3 or len(C2) < 3: continue
        icp_d = np.hypot(*(poses[i, :2] - poses[i - 10, :2]))
        T = rot(poses[i - 10, 2]); T2 = rot(poses[i, 2])
        W1 = C1 @ T.T + poses[i - 10, :2]; W2 = C2 @ T2.T + poses[i, :2]
        d, j = cKDTree(W2).query(W1, distance_upper_bound=1.0)
        m = np.isfinite(d)
        diffs.append((icp_d, float(np.median(d[m])) if m.sum() >= 3 else np.nan))
    icp_v = np.array([a for a, _ in diffs])
    lres = np.array([b for _, b in diffs])
    print(f"validation: label residual in ICP world frame (1-s): median "
          f"{np.nanmedian(lres):.2f} m over {len(diffs)} checks (label jitter + odo error)", flush=True)

    np.savez(f"{WD}/odometry.npz", pids=pids, poses=poses, v=v, w=w, inl=inl, res=res,
             cond=cond, kf_ids=np.array(kf_ids), deskew_sign=sign, deskew_cut=cut,
             wA=wA, vA=vA, dt=dt, bad=bad, drift_median=med)
    print(f"saved {WD}/odometry.npz", flush=True)
    return WD, z, poses, DS, (wA, vA, sign, cut), (inl, res, cond, kf_ids), tracks

# ---------------- viz suite ----------------
def odo_viz(win):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    WD, z, poses, DS, (wA, vA, sign, cut), (inl, res, cond, kf_ids), tracks = odo_build(win)
    pids, L = z["pids"], z["lidar"]
    N = len(pids)
    VD = f"{WD}/odometry_viz"; os.makedirs(VD, exist_ok=True)

    def world(i): return DS[i] @ rot(poses[i, 2]).T + poses[i, :2]
    allw = [world(i) for i in range(0, N, 2)]

    # V1 accumulated map
    fig, ax = plt.subplots(figsize=(14, 14), dpi=170)
    cm = plt.cm.viridis
    for n, Wp in enumerate(allw):
        ax.scatter(Wp[:, 1], Wp[:, 0], s=0.25, color=cm(n / len(allw)), alpha=.35, lw=0)
    ax.plot(poses[:, 1], poses[:, 0], "r-", lw=1.5)
    ax.set_aspect("equal"); ax.invert_xaxis()
    ax.set_title(f"{win} V1 accumulated map (early=purple, late=yellow) -- crisp walls = PASS")
    ax.set_xlabel("lateral (m)"); ax.set_ylabel("forward of frame0 (m)")
    fig.savefig(f"{VD}/V1_map.png", bbox_inches="tight"); plt.close(fig)

    # V2 deskew before/after at sharpest turn
    j = int(np.argmax(np.abs(wA)))
    raw = prep(L[j]) @ rot(poses[j, 2]).T + poses[j, :2]
    dsk = world(j)
    ref = np.vstack([world(k) for k in range(max(0, j - 12), j - 2, 2)])
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150, sharex=True, sharey=True)
    for ax, pts, tag in ((axs[0], raw, "RAW"), (axs[1], dsk, "DESKEWED")):
        ax.scatter(ref[:, 1], ref[:, 0], s=0.3, c="#B0B8C0", lw=0)
        ax.scatter(pts[:, 1], pts[:, 0], s=0.6, c="#B05A1E", lw=0)
        ax.set_aspect("equal"); ax.invert_xaxis()
        ax.set_title(f"V2 {tag}  pid {pids[j]}  w={np.degrees(wA[j]):+.0f} deg/s")
    fig.savefig(f"{VD}/V2_deskew.png", bbox_inches="tight"); plt.close(fig)

    # V3 trajectory
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    M = np.vstack(allw[::2])
    ax.scatter(M[:, 1], M[:, 0], s=0.2, c="#C9D2DA", lw=0)
    sc = ax.scatter(poses[:, 1], poses[:, 0], c=np.hypot(*np.gradient(poses[:, :2], axis=0).T) * 10,
                    cmap="plasma", s=7)
    for i in range(0, N, 10):
        d = rot(poses[i, 2]) @ [1.1, 0]
        ax.annotate("", xy=(poses[i, 1] + d[1], poses[i, 0] + d[0]),
                    xytext=(poses[i, 1], poses[i, 0]),
                    arrowprops=dict(arrowstyle="->", color="#0E6E74", lw=1))
    ax.scatter(poses[kf_ids, 1], poses[kf_ids, 0], marker="s", s=40,
               facecolor="none", edgecolor="#B05A1E", lw=1.5, label=f"keyframes ({len(kf_ids)})")
    plt.colorbar(sc, ax=ax, label="speed m/s", shrink=.6)
    ax.set_aspect("equal"); ax.invert_xaxis(); ax.legend()
    ax.set_title(f"{win} V3 trajectory + heading (1 Hz) + keyframes")
    fig.savefig(f"{VD}/V3_trajectory.png", bbox_inches="tight"); plt.close(fig)

    # V4 landmark scatter
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    ax.scatter(M[:, 1], M[:, 0], s=0.2, c="#DDE4EA", lw=0)
    cmt = plt.cm.tab20
    ns = nm = 0
    for n, tr in enumerate(tracks):
        if len(tr) < 30: continue
        T = np.array(tr); mv = np.hypot(*(T[-1] - T[0])) > 2.0
        ax.scatter(T[:, 1], T[:, 0], s=3 if mv else 6, color=cmt(n % 20),
                   marker="x" if mv else "o", lw=0.8 if mv else 0)
        if not mv:
            sp = np.hypot(*(T.max(0) - T.min(0)))
            ax.annotate(f"{sp:.2f}", (T[:, 1].mean(), T[:, 0].mean()), fontsize=7, color="#1C2733")
            ns += 1
        else: nm += 1
    ax.set_aspect("equal"); ax.invert_xaxis()
    ax.set_title(f"V4 landmarks: {ns} static (dots, spread m annotated -- tight=PASS), {nm} movers (x)")
    fig.savefig(f"{VD}/V4_landmarks.png", bbox_inches="tight"); plt.close(fig)

    # V5 quality strip
    dtv = np.clip(np.diff(z["lidar_t"], prepend=z["lidar_t"][0] - 0.1), 0.1, 10.0)
    t = np.arange(N) / 10
    fig, axs = plt.subplots(5, 1, figsize=(13, 10), dpi=140, sharex=True)
    for ax, y, lab_ in zip(axs, [np.hypot(*np.gradient(poses[:, :2], axis=0).T) / dtv,
                                 np.degrees(np.gradient(np.unwrap(poses[:, 2])) / dtv), inl, res, cond],
                           ["v m/s", "omega deg/s", "ICP inlier", "residual m", "cond"]):
        ax.plot(t, y, lw=1, color="#0E6E74"); ax.set_ylabel(lab_); ax.grid(alpha=.3)
    for ax in axs:
        for k in kf_ids: ax.axvline(k / 10, color="#B05A1E", alpha=.25, lw=.8)
    axs[-1].set_xlabel("t (s)   [orange lines = keyframes]")
    axs[0].set_title(f"{win} V5 quality")
    fig.savefig(f"{VD}/V5_quality.png", bbox_inches="tight"); plt.close(fig)

    # V6 camera cross-check at 4 times
    sys.path.insert(0, HERE)
    import camera_detector as CD, cv2
    cb = CD.geo_calib("2026.05.08")
    Tcl = np.array(cb["camera_to_lidar"], float); Tlc = np.linalg.inv(Tcl); ci = cb["color_intrinsics"]
    cci = z["camera_color_idx"]
    Mall = np.vstack(allw)
    zg = np.percentile(np.concatenate([L[i].astype(np.float32)[:, 2] for i in range(0, N, 40)]), 2)
    for frac in (0.1, 0.35, 0.6, 0.9):
        i = int(frac * (N - 1))
        img = cv2.imread(f"{WD}/color/p{pids[i]:06d}.jpg")
        if img is None: continue
        Pe = (Mall - poses[i, :2]) @ rot(poses[i, 2])            # world -> ego_i
        P3 = np.column_stack([Pe, np.full(len(Pe), zg + 1.2), np.ones(len(Pe))])
        Pc = (Tlc @ P3.T).T[:, :3]
        m = Pc[:, 2] > 1.0
        u = (ci["fx"] * Pc[m, 0] / Pc[m, 2] + ci["cx"]).astype(int)
        vv = (ci["fy"] * Pc[m, 1] / Pc[m, 2] + ci["cy"]).astype(int)
        d = Pc[m, 2]
        ok = (u >= 0) & (u < img.shape[1]) & (vv >= 0) & (vv < img.shape[0]) & (d < 35)
        col = (plt.cm.plasma(1 - np.clip(d[ok] / 35, 0, 1))[:, :3] * 255).astype(np.uint8)
        for x_, y_, c_ in zip(u[ok], vv[ok], col):
            cv2.circle(img, (x_, y_), 1, (int(c_[2]), int(c_[1]), int(c_[0])), -1)
        cv2.imwrite(f"{VD}/V6_cam_{pids[i]}.png", img)

    # V7 flipbook -> mp4 via ffmpeg (cv2 cannot write mp4)
    import tempfile, subprocess, shutil
    td = tempfile.mkdtemp(prefix="odoflip_")
    Mds = Mall[::3]
    yawc = float(np.arctan2(Tcl[1, 2], Tcl[0, 2]))       # camera optical axis yaw in lidar frame
    for i in range(N):
        fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
        ax.scatter(Mds[:, 1], Mds[:, 0], s=0.15, c="#D6DDE3", lw=0)
        Wp = world(i)
        ax.scatter(Wp[:, 1], Wp[:, 0], s=0.6, c="#0E6E74", lw=0)
        a0 = poses[i, 2] + yawc
        for s_ in (-CAM_HALF_FOV, CAM_HALF_FOV):
            d = rot(a0 + s_) @ [25, 0]
            ax.plot([poses[i, 1], poses[i, 1] + d[1]], [poses[i, 0], poses[i, 0] + d[0]],
                    color="#B05A1E", lw=1.2, alpha=.8)
        ax.plot(poses[:i + 1, 1], poses[:i + 1, 0], "r-", lw=1)
        ax.set_aspect("equal"); ax.invert_xaxis()
        ax.set_xlim(Mds[:, 1].max() + 5, Mds[:, 1].min() - 5)
        ax.set_ylim(Mds[:, 0].min() - 5, Mds[:, 0].max() + 5)
        ax.set_title(f"pid {pids[i]}  t={i/10:.1f}s")
        fig.savefig(f"{td}/f{i:04d}.jpg"); plt.close(fig)
    out_local = f"{td}/V7_flipbook.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", "10", "-i", f"{td}/f%04d.jpg",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", out_local],
                   check=True, capture_output=True)
    shutil.copy(out_local, f"{VD}/V7_flipbook.mp4"); shutil.rmtree(td)
    print(f"viz saved to {VD}", flush=True)


# ===================== DETECT (was window_detect.py) =====================


def detect_main(win):
    import cv2, data_l2, pack_day
    import camera_detector as CD
    import precompute_proposals as PP
    WD = f"{MV}/{win}"
    z = np.load(f"{WD}/sensors.npz", allow_pickle=True)
    pids, L, RA = z["pids"], z["lidar"], z["radar_ra"]
    cci = z["camera_color_idx"]; lt = z["lidar_t"]
    N = len(pids)
    az, rf = pack_day._convention("2026.05.08")
    fa = np.arange(0, 30.0001, 0.1); la = np.arange(-30, 30.0001, 0.1)
    det_lid, det_rad = PP.load_detectors()
    nets = CD.load_nets()
    lab = load_labels()
    out = []
    for i in range(N):
        lid = L[i].astype(np.float32)
        if lid.shape[1] == 3:
            lid = np.concatenate([lid, np.ones((len(lid), 1), np.float32)], 1)
        bev = pack_day.ra_to_bev(RA[i].astype(np.float32), az, rf)
        rpts = data_l2.radar_points(bev, la, fa)
        props = dict(
            lidar=[(b, s) for b, s in PP.voxel_props(det_lid, lid)],
            radar=[(b, s) for b, s in PP.voxel_props(det_rad, rpts)])
        img = cv2.imread(f"{WD}/color/p{pids[i]:06d}.jpg")
        cam = []
        if img is not None:
            fr = dict(cam=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cam_idx=int(cci[i]), seq="2026.05.08")
            cam = [(p["bev_box"], p["score"], p["label"]) for p in CD.propose(fr, nets)]
        props["cam"] = cam
        _, us = gt_centres(lab, int(pids[i]))
        out.append(dict(pid=int(pids[i]), t=float(lt[i]), props=props,
                        union_gt=[dict(box=u["box"], label=u.get("label", "car")) for u in us]))
        if i % 20 == 0:
            print(f"  {i}/{N}  cam {len(cam)} lid {len(props['lidar'])} rad {len(props['radar'])} "
                  f"gt {len(out[-1]['union_gt'])}", flush=True)
    pickle.dump(out, open(f"{WD}/proposals_window.pkl", "wb"))
    tot = {m: sum(len(f["props"][m]) for f in out) for m in ("cam", "lidar", "radar")}
    print(f"saved {WD}/proposals_window.pkl  totals {tot}", flush=True)


# ===================== MEMORY (was window_memory.py) =====================

HALF_FOV = np.radians(34.0); RMAX = 30.0
TAU_S, TAU_M, EMIT, ASSOC = 30.0, 1.5, 0.25, 1.5
WRITE_RMAX = 12.0      # trust gate: cam localises 0.7-0.9 m inside 12 m but 3.7-7.6 m beyond 18 m
TRI_ON = False         # exit-sweep triangulation: TESTED AND REJECTED on win_8825-9175 -- the bearing
                       # diversity needed for the solve appears exactly while the 2D box clips at the
                       # image border (centre biased inward, bias grows through the sweep) -> 4 promoted
                       # entries, placement p50 4.8 m, ghosts +3/frame, recall unchanged. Keep for scenes
                       # where objects CROSS the wedge interior instead of exiting through the edge.
TRI_MIN_RAYS, TRI_MIN_SPREAD, TRI_MAX_RESID = 6, np.radians(8.0), 1.2
RADAR_XCHK = True      # radar cross-check: far cam det -> radar range along the cam bearing
                       # (cam bearing ~1 deg accurate + radar range ~cm accurate = certified write)
XCHK_AZ, XCHK_MINPTS = np.radians(4.0), 2
XCHK_PERSIST = 1       # TESTED AND REJECTED at 2: true objects give SPARSE certified sightings
                       # (filtered out) while ghost streams are SYSTEMATIC (cam FP at a wall corner,
                       # certified every frame by real radar energy) -- persistence selects wrongly.
XCHK_MINSCORE = 0.0    # score gate TESTED, NO EFFECT: ghost streams are HIGH-score cam FPs on radar-
                       # reflective structure (walls) -- a camera-side FP problem, not filterable here
NEG_DECAY = 0.97
FLOOR = float(__import__('os').environ.get('WC_FLOOR', 0.26))       # per-miss decay, TRUST-ZONE only: measured per-object cam recall can be 13%
                       # (obj3: 7 hits / 53 in-FOV frames) -- 0.85 killed correct entries between sightings
WARM = 15.0

def radius_of(lb): return 1.0 if lb in ("person", "bicycle", "motorcycle") else 2.0

def in_fov(pw, pose, yawc):
    """world point inside the camera wedge at `pose`?"""
    d = rot(pose[2]).T @ (pw - pose[:2])
    r = np.hypot(*d)
    return r <= RMAX and abs((np.arctan2(d[1], d[0]) - yawc + np.pi) % (2 * np.pi) - np.pi) <= HALF_FOV

def radar_range_at(rpts, az_c, r_mono):
    """radar range along camera bearing az_c, gated by the mono range band; None if no support."""
    if rpts is None or not len(rpts): return None
    az = np.arctan2(rpts[:, 1], rpts[:, 0]); rr = np.hypot(rpts[:, 0], rpts[:, 1])
    m = (np.abs((az - az_c + np.pi) % (2 * np.pi) - np.pi) <= XCHK_AZ) & \
        (np.abs(rr - r_mono) <= np.maximum(5.0, 0.5 * r_mono)) & (rr > 2.0)
    if m.sum() < XCHK_MINPTS: return None
    w = rpts[m, 2] - rpts[m, 2].min() + 1e-3
    o = np.argsort(rr[m]); cw = np.cumsum(w[o])
    return float(rr[m][o][np.searchsorted(cw, cw[-1] / 2)])     # power-weighted median range

def tri_solve(rays):
    """least-squares intersection of 2D rays [(o,d)]; robust re-solve after trimming worst 30%.
    -> (point, median residual, angular spread) or None if degenerate."""
    def solve(rs):
        A = np.zeros((2, 2)); b = np.zeros(2)
        for o, d in rs:
            Nm = np.eye(2) - np.outer(d, d)
            A += Nm; b += Nm @ o
        if abs(np.linalg.det(A)) < 1e-6: return None
        return np.linalg.solve(A, b)
    p0 = solve(rays)
    if p0 is None: return None
    res = [float(np.hypot(*((np.eye(2) - np.outer(d, d)) @ (p0 - o)))) for o, d in rays]
    keep = [r for _, r in sorted(zip(res, rays))][:max(4, int(0.7 * len(rays)))]
    p1 = solve(keep)
    if p1 is None: return None
    res1 = [float(np.hypot(*((np.eye(2) - np.outer(d, d)) @ (p1 - o)))) for o, d in keep]
    angs = [np.arctan2(d[1], d[0]) for _, d in keep]
    spread = max(angs) - min(angs)
    return p1, float(np.median(res1)), float(spread)

def memory_run(win, viz=False, source="cam"):
    WD = f"{MV}/{win}"
    P = pickle.load(open(f"{WD}/proposals_window.pkl", "rb"))
    oz = np.load(f"{WD}/odometry.npz")
    poses = oz["poses"]; pids = oz["pids"]
    assert all(P[i]["pid"] == pids[i] for i in range(len(P)))
    import camera_detector as CD
    cb = CD.geo_calib("2026.05.08")
    Tcl = np.array(cb["camera_to_lidar"], float)
    yawc = float(np.arctan2(Tcl[1, 2], Tcl[0, 2]))
    t0 = P[0]["t"]
    N = len(P)

    # coverage grid (1 m cells) for B_rec vs B_inf attribution
    allp = poses[:, :2]
    lo = allp.min(0) - RMAX - 2; hi = allp.max(0) + RMAX + 2
    gw = np.ceil((hi - lo)).astype(int)
    seen = np.full(gw, False)
    gy, gx = np.meshgrid(np.arange(gw[1]) + 0.5, np.arange(gw[0]) + 0.5)
    cells = np.stack([gx.ravel(), gy.ravel()], 1) + lo          # cell centres, world

    o_cam_ego = Tcl[:2, 3]                                      # camera origin in the lidar frame
    RPTS = None
    if source == "cam" and RADAR_XCHK:                          # radar points per frame (RA -> BEV -> CFAR)
        import pack_day, data_l2
        zs = np.load(f"{WD}/sensors.npz", allow_pickle=True)
        azc_, rfc_ = pack_day._convention("2026.05.08")
        fa_ = np.arange(0, 30.0001, 0.1); la_ = np.arange(-30, 30.0001, 0.1)
        RPTS = [data_l2.radar_points(pack_day.ra_to_bev(zs["radar_ra"][k].astype(np.float32),
                                                        azc_, rfc_), la_, fa_)
                for k in range(N)]
        print(f"radar cross-check armed: {np.mean([len(r) for r in RPTS]):.0f} radar pts/frame", flush=True)
    bank = []                                                   # memory entries
    xchk_stats = dict(n=0, derr=[], cand=0)
    xchk_pend = []                                              # (world pos, t) awaiting persistence
    rtracks = []                                                # exit-sweep ray tracks
    tri_stats = dict(promoted=0, derr=[])
    rows = dict(fov_gt=0, fov_hit=0, blind_gt=0, blind_mem=0, blind_lid=0, blind_cam=0,
                brec_gt=0, brec_mem=0, brec_lid=0, binf_gt=0, binf_mem=0, binf_lid=0, ghosts=0, qframes=0,
                all_gt=0, all_cam=0, all_cammem=0, all_lid=0)
    recov_log = []; dbg_blind = []; dbg_bank = []
    frames_viz = []
    past_src = []                     # every SOURCE detection's world pos (cam dets / lidar samples)
    past_wr = []                      # every ACCEPTED (trust-gated) write's world pos
    n1 = dict(seen=0, seen_hit=0, wr=0, wr_hit=0)   # blind instances with prior knowledge
    for i in range(N):
        t = P[i]["t"] - t0; pose = poses[i]
        Rw, pw = rot(pose[2]), pose[:2]
        # ---- coverage write ----
        d = (cells - pw) @ Rw                                   # ego coords of cells
        r = np.hypot(d[:, 0], d[:, 1])
        az = (np.arctan2(d[:, 1], d[:, 0]) - yawc + np.pi) % (2 * np.pi) - np.pi
        seen.ravel()[np.where((r <= RMAX) & (np.abs(az) <= HALF_FOV))[0]] = True
        # ---- live camera detections -> world ----
        camw = []
        for b, s, lb in P[i]["props"]["cam"]:
            c = Rw @ np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) + pw
            camw.append((c, s, lb))
        # ---- memory WRITE (uncertainty-weighted: EVERY det stored, far ones enter uncertain;
        #      repeated sightings fuse by inverse variance -> bearings from different poses
        #      triangulate the noisy range down; emission requires converged sigma) ----
        if source == "cam":
            # trust-gated writes: only store where the cam PLACES well (<=12 m: 0.7-0.9 m error).
            # (uncertainty-fusion variant tested and rejected: cam range errors are per-object BIASED,
            # inverse-variance fusion shrinks sigma without shrinking error, association contaminates)
            writes = []
            for (b, sc_, lb), (c, _, _) in zip(P[i]["props"]["cam"], camw):
                fc_, lc_ = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                r_mono = np.hypot(fc_, lc_)
                if r_mono <= WRITE_RMAX:
                    writes.append((c, sc_, lb, 0.8))            # near: cam's own placement trusted
                elif RPTS is not None:
                    az_c = np.arctan2(lc_, fc_)
                    r_hat = radar_range_at(RPTS[i], az_c, r_mono)
                    if r_hat is not None:                       # far: radar-certified range + cam bearing
                        ce = Rw @ np.array([r_hat * np.cos(az_c), r_hat * np.sin(az_c)]) + pw
                        xchk_stats["cand"] += 1
                        prev = next((q for q in xchk_pend
                                     if t - q[1] <= 0.35 and np.hypot(*(q[0] - ce)) <= 2.0), None) \
                            if XCHK_PERSIST > 1 else True
                        if sc_ < XCHK_MINSCORE: prev = None
                        if prev is not None:                    # accepted
                            writes.append((ce, sc_, lb, 1.0))
                            xchk_stats["n"] += 1
                            gds = [np.hypot(*(ce - (Rw @ np.array([(u["box"][0]+u["box"][2])/2,
                                   (u["box"][1]+u["box"][3])/2]) + pw))) for u in P[i]["union_gt"]]
                            if gds: xchk_stats["derr"].append(float(min(gds)))
                            if XCHK_PERSIST > 1: prev[0][:] = ce; prev[1] = t
                        else:
                            xchk_pend.append([ce.copy(), t])
            xchk_pend = [q for q in xchk_pend if t - q[1] <= 0.35]
        else:                                                   # lidar @ 1 Hz (sparse expensive sensor)
            writes = [] if i % 10 else                 [(Rw @ np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) + pw, sc, "car")
                 for b, sc in P[i]["props"]["lidar"]
                 if np.hypot((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) <= 28]
        past_src += [c for c, _, _ in camw] if source == "cam" else [c for c, _, _, _ in writes]
        past_wr += [c for c, _, _, _ in writes]
        for c, sc_, lb, sig in writes:
            e = next((e for e in bank if np.hypot(*(e["p"] - c)) < ASSOC), None)
            if e is None:
                bank.append(dict(p=c.copy(), p0=c.copy(), s=sc_, lb=lb, last=t,
                                 miss=0, disp=0.0, n=1))
            else:
                e["p"] = 0.7 * e["p"] + 0.3 * c
                e["s"] = max(e["s"], sc_); e["last"] = t; e["miss"] = 0; e["n"] += 1
                e["disp"] = max(e["disp"], float(np.hypot(*(e["p"] - e["p0"]))))
        # ---- exit-sweep triangulation: log a bearing RAY per det (range ignored), solve on spread ----
        if source == "cam" and TRI_ON:
            o_w = Rw @ o_cam_ego + pw
            for (b, sc_, lb), (cmono, _, _) in zip(P[i]["props"]["cam"], camw):
                az_e = np.arctan2((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)
                d_w = Rw @ np.array([np.cos(az_e), np.sin(az_e)])
                tr = None
                for rt in rtracks:
                    if rt["grp"] != (lb in (1, 2, 4)): continue
                    perp = np.hypot(*((np.eye(2) - np.outer(d_w, d_w)) @ (rt["p"] - o_w)))
                    if perp < 2.5 and np.hypot(*(rt["p"] - cmono)) < 12: tr = rt; break
                if tr is None:
                    rtracks.append(dict(p=cmono.copy(), rays=[(o_w, d_w)], grp=lb in (1, 2, 4),
                                        lb=lb, s=sc_, last=t))
                else:
                    tr["rays"].append((o_w, d_w)); tr["rays"] = tr["rays"][-40:]
                    tr["s"] = max(tr["s"], sc_); tr["last"] = t
                    if len(tr["rays"]) >= TRI_MIN_RAYS:
                        sol = tri_solve(tr["rays"])
                        if sol:
                            p_t, resid, spread = sol
                            tr["p"] = p_t
                            if spread >= TRI_MIN_SPREAD and resid <= TRI_MAX_RESID:
                                e = next((e for e in bank if np.hypot(*(e["p"] - p_t)) < 2.0), None)
                                if e is None:
                                    bank.append(dict(p=p_t.copy(), p0=p_t.copy(), s=tr["s"], lb=tr["lb"],
                                                     last=t, miss=0, disp=0.0, n=len(tr["rays"])))
                                    tri_stats["promoted"] += 1
                                    gds = [np.hypot(*(p_t - (Rw @ np.array([(u["box"][0]+u["box"][2])/2,
                                           (u["box"][1]+u["box"][3])/2]) + pw))) for u in P[i]["union_gt"]]
                                    if gds: tri_stats["derr"].append(float(min(gds)))
                                else:
                                    e["p"] = 0.6 * e["p"] + 0.4 * p_t
                                    e["s"] = max(e["s"], tr["s"]); e["last"] = t; e["n"] += 1
            rtracks = [rt for rt in rtracks if t - rt["last"] < 3.0]
        # ---- negative evidence (soft: score decay per contradicted frame) ----
        if source == "cam":
            for e in bank:
                d_ego = rot(pose[2]).T @ (e["p"] - pw)
                if np.hypot(*d_ego) <= WRITE_RMAX and in_fov(e["p"], pose, yawc) and \
                   not any(np.hypot(*(e["p"] - c)) < ASSOC for c, _, _ in camw):
                    e["s"] *= NEG_DECAY
                    if e["n"] >= 4: e["s"] = max(e["s"], FLOOR)  # multi-confirmed: never silenced by misses
            bank = [e for e in bank if e["s"] > 0.12]
        elif writes:                                            # lidar is 360: every sample re-tests all
            wc = [c for c, _, _ in writes]
            for e in bank:
                d_ego = rot(pose[2]).T @ (e["p"] - pw)
                if np.hypot(*d_ego) <= 28 and not any(np.hypot(*(e["p"] - c)) < ASSOC for c in wc):
                    e["s"] *= 0.6
            bank = [e for e in bank if e["s"] > 0.12]
        # ---- memory READ (blind zone only; emission needs CONVERGED position belief) ----
        mem = []
        for e in bank:
            tau = TAU_M if e["disp"] > 1.2 else TAU_S
            sc = e["s"] * np.exp(-(t - e["last"]) / tau)
            if sc >= EMIT and not in_fov(e["p"], pose, yawc):
                mem.append((e["p"], sc, e["lb"], t - e["last"]))
        if i % 25 == 0: dbg_bank.append((round(t, 1), len(bank), len(mem)))
        # ---- evaluation on query frames ----
        if t < WARM or not P[i]["union_gt"]:
            if viz: frames_viz.append((i, camw, mem, [], []))
            continue
        rows["qframes"] += 1
        lidw = [Rw @ np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) + pw
                for b, _ in P[i]["props"]["lidar"]]
        gts, recovered = [], []
        for u in P[i]["union_gt"]:
            gc = Rw @ np.array([(u["box"][0] + u["box"][2]) / 2, (u["box"][1] + u["box"][3]) / 2]) + pw
            dr = np.hypot(*(rot(pose[2]).T @ (gc - pw)))
            if dr > RMAX: continue
            rad = radius_of(u["label"])
            hit_cam = any(np.hypot(*(gc - c)) <= rad for c, _, _ in camw)
            hit_mem = any(np.hypot(*(gc - c)) <= rad for c, _, _, _ in mem)
            hit_lid = any(np.hypot(*(gc - c)) <= rad for c in lidw)
            fov = in_fov(gc, pose, yawc)
            rows["all_gt"] += 1
            rows["all_cam"] += hit_cam
            rows["all_cammem"] += (hit_cam or hit_mem)
            rows["all_lid"] += hit_lid
            gts.append((gc, fov, hit_cam, hit_mem))
            if fov:
                rows["fov_gt"] += 1; rows["fov_hit"] += hit_cam
            else:
                if any(np.hypot(*(gc - c)) <= 2.5 for c in past_src):
                    n1["seen"] += 1; n1["seen_hit"] += hit_mem
                if any(np.hypot(*(gc - c)) <= 2.5 for c in past_wr):
                    n1["wr"] += 1; n1["wr_hit"] += hit_mem
                dmem = min((np.hypot(*(gc - c)) for c, _, _, _ in mem), default=np.inf)
                dbank = min((np.hypot(*(gc - e["p"])) for e in bank), default=np.inf)
                dbg_blind.append((float(dmem), float(dbank)))
                rows["blind_gt"] += 1
                rows["blind_mem"] += hit_mem; rows["blind_lid"] += hit_lid
                rows["blind_cam"] += hit_cam
                ci = np.floor(gc - lo).astype(int)
                ever = seen[ci[0], ci[1]] if (0 <= ci[0] < gw[0] and 0 <= ci[1] < gw[1]) else False
                if ever:
                    rows["brec_gt"] += 1; rows["brec_mem"] += hit_mem; rows["brec_lid"] += hit_lid
                else:
                    rows["binf_gt"] += 1; rows["binf_mem"] += hit_mem; rows["binf_lid"] += hit_lid
                if hit_mem:
                    age = min(a for c, _, _, a in mem if np.hypot(*(gc - c)) <= rad)
                    recov_log.append((P[i]["pid"], u["label"], age))
                    recovered.append(gc)
        for c, _, _, _ in mem:
            if not any(np.hypot(*(c - g)) <= 2.5 for g, *_ in gts): rows["ghosts"] += 1
        if viz: frames_viz.append((i, camw, mem, [g for g, *_ in gts], recovered))

    # ---- report ----
    R = rows
    print(f"\n{win}: {R['qframes']} query frames (warm-up {WARM:.0f} s), GT<= {RMAX:.0f} m")
    print(f"  in-FOV GT   : {R['fov_gt']:4d}   live cam recall {R['fov_hit']/max(1,R['fov_gt']):.2f}")
    print(f"  BLIND GT    : {R['blind_gt']:4d}   cam live {R['blind_cam']/max(1,R['blind_gt']):.2f}  "
          f"MEMORY {R['blind_mem']/max(1,R['blind_gt']):.2f}")
    print(f"    CASE A fell-out-of-FOV (in accumulated cam FOV, out of current) -- THE TARGET CASE:")
    print(f"      {R['brec_gt']:4d} GT   memory {R['brec_mem']/max(1,R['brec_gt']):.2f}")
    print(f"    CASE B never-in-any-cam-FOV (outside accumulated wedges, lidar/radar-only world):")
    print(f"      {R['binf_gt']:4d} GT   memory {R['binf_mem']/max(1,R['binf_gt']):.2f}")
    print(f"  ALL GT      : {R['all_gt']:4d}   cam {R['all_cam']/max(1,R['all_gt']):.2f} -> "
          f"cam+MEM {R['all_cammem']/max(1,R['all_gt']):.2f}")
    print(f"  memory ghosts: {R['ghosts']/max(1,R['qframes']):.2f} per query frame")
    print(f"  RETRIEVABILITY (prior-knowledge denominators, blind GT only):")
    print(f"    n1 = blind GT the source EVER DETECTED near truth (<=2.5 m, any range, any time): "
          f"{n1['seen']:4d}  -> retrieved n2 {n1['seen_hit']:3d}   n2/n1 = {n1['seen_hit']/max(1,n1['seen']):.2f}")
    print(f"    n1t = blind GT with an ACCEPTED trust-gated write                              : "
          f"{n1['wr']:4d}  -> retrieved    {n1['wr_hit']:3d}   n2/n1t = {n1['wr_hit']/max(1,n1['wr']):.2f}")
    print(f"    vs blind N {R['blind_gt']}: {n1['seen_hit']/max(1,R['blind_gt']):.2f}   "
          f"vs ALL GT {R['all_gt']}: {n1['seen_hit']/max(1,R['all_gt']):.2f}")
    if source == "cam" and RADAR_XCHK:
        de = xchk_stats["derr"]
        print(f"  radar-certified far writes: {xchk_stats['n']} "
              f"(of {xchk_stats['cand']} candidates; persistence filtered "
              f"{xchk_stats['cand']-xchk_stats['n']})"
              + (f", placement error vs nearest GT p50 {np.median(de):.1f} m "
                 f"(p75 {np.percentile(de,75):.1f})" if de else ""))
    if source == "cam" and TRI_ON:
        de = tri_stats["derr"]
        print(f"  exit-sweep triangulation: {tri_stats['promoted']} entries promoted"
              + (f", placement error vs nearest GT p50 {np.median(de):.1f} m "
                 f"(p75 {np.percentile(de,75):.1f})" if de else ""))
    if dbg_blind:
        dm = np.array([a for a, _ in dbg_blind]); db = np.array([b for _, b in dbg_blind])
        print(f"  DEBUG blind-GT nearest EMISSION dist: p25 {np.percentile(dm,25):.1f} "
              f"p50 {np.percentile(dm,50):.1f} p75 {np.percentile(dm,75):.1f} m | "
              f"nearest BANK-entry dist: p25 {np.percentile(db,25):.1f} p50 {np.percentile(db,50):.1f} "
              f"p75 {np.percentile(db,75):.1f} m")
        print(f"  DEBUG bank size (t, entries, emitted): {dbg_bank}")
    if recov_log:
        ages = [a for _, _, a in recov_log]
        print(f"  recoveries: {len(recov_log)} (memory age p50 {np.median(ages):.1f} s, "
              f"max {max(ages):.1f} s)")
    json.dump(dict(rows=rows, n_recov=len(recov_log), source=source),
              open(f"{WD}/memory_result_{source}.json", "w"), indent=1)

    if viz:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile, subprocess, shutil
        z = np.load(f"{WD}/sensors.npz", allow_pickle=True)
        M = np.vstack([prep(z["lidar"][k]) @ rot(poses[k, 2]).T + poses[k, :2]
                       for k in range(0, N, 4)])[::2]
        VD = f"{WD}/odometry_viz"; os.makedirs(VD, exist_ok=True)      # all window viz in ONE folder
        td = tempfile.mkdtemp(prefix="memflip_")
        for n, (i, camw, mem, gts, recovered) in enumerate(frames_viz):
            fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=105)
            ax.scatter(M[:, 1], M[:, 0], s=0.15, c="#DDE2E8", lw=0)
            pose = poses[i]
            a0 = pose[2] + yawc
            for s_ in (-HALF_FOV, HALF_FOV):
                d = rot(a0 + s_) @ [RMAX, 0]
                ax.plot([pose[1], pose[1] + d[1]], [pose[0], pose[0] + d[0]],
                        color="orange", lw=1.2, alpha=.85)
            th = np.linspace(pose[2] - np.pi / 2, pose[2] + np.pi / 2, 60)   # radar FOV +-90deg/25m
            ax.plot(pose[1] + 25 * np.sin(th), pose[0] + 25 * np.cos(th),
                    color="blue", lw=.9, alpha=.45, ls="--")
            ax.plot(poses[:i + 1, 1], poses[:i + 1, 0], "-", color="#C44", lw=1)
            for g in gts: ax.scatter(g[1], g[0], marker="s", s=90, facecolor="none",
                                     edgecolor="red", lw=1.4)
            for c, s, lb in camw: ax.scatter(c[1], c[0], marker="o", s=34, color="orange")
            for c, s, lb, age in mem: ax.scatter(c[1], c[0], marker="D", s=30, color="green")
            for g in recovered: ax.scatter(g[1], g[0], marker="o", s=240, facecolor="none",
                                           edgecolor="green", lw=2.2)
            ax.set_aspect("equal"); ax.invert_xaxis()
            ax.set_xlim(M[:, 1].max() + 4, M[:, 1].min() - 4)
            ax.set_ylim(M[:, 0].min() - 4, M[:, 0].max() + 4)
            ax.set_title(f"pid {P[i]['pid']}  t={P[i]['t']-t0:5.1f}s   orange=cam(FOV+dets)  "
                         f"blue=radar FOV  green=memory/recovered  red=GT", fontsize=9)
            fig.savefig(f"{td}/f{n:04d}.jpg"); plt.close(fig)
        out = f"{td}/M_flipbook.mp4"
        subprocess.run(["ffmpeg", "-y", "-framerate", "10", "-i", f"{td}/f%04d.jpg",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", out],
                       check=True, capture_output=True)
        shutil.copy(out, f"{VD}/M7_flipbook_camrad-cert.mp4"); shutil.rmtree(td)
        print(f"viz saved to {VD}", flush=True)


# ===================== COMPARE (was window_compare.py) =====================

RMAX_EVAL = 30.0
EMIT_HOOK = None   # optional codec hook applied at EMISSION (the transmit point): f(e, pose, t, sc)
                   # -> (world_pos, label) possibly reconstructed through a codec, or None to drop
NEG = 0.97
FLOOR = float(__import__('os').environ.get('WC_FLOOR', 0.26))
WGATE = float(__import__('os').environ.get('WC_WGATE', 0.0))    # radar write score gate
TAU_U = float(__import__('os').environ.get('WC_TAU_U', 30.0))   # persistence for UNCONFIRMED (n<NCONF)
NCONF = int(__import__('os').environ.get('WC_NCONF', 1))        # confirmations for full tau
CAM_TRUST = 12.0
WEDGE = dict(cam=(np.radians(34.0), 30.0), radar=(np.radians(90.0), 25.0), lidar=(np.pi, 30.0))

def voc_ap(bag, npos):
    """VOC all-point AP from [(score, is_tp)] vs npos GT."""
    if not bag or npos == 0: return 0.0
    bag = sorted(bag, key=lambda x: -x[0])
    tp = np.cumsum([b for _, b in bag]); fp = np.cumsum([1 - b for _, b in bag])
    rc = tp / npos; pr = tp / np.maximum(tp + fp, 1e-9)
    pri = np.maximum.accumulate(pr[::-1])[::-1]
    ap, prev = 0.0, 0.0
    for k in range(len(rc)):
        if rc[k] > prev: ap += (rc[k] - prev) * pri[k]; prev = rc[k]
    return float(ap)

def compare_run(win, only=None):
    WD = f"{MV}/{win}"
    P = pickle.load(open(f"{WD}/proposals_window.pkl", "rb"))
    poses = np.load(f"{WD}/odometry.npz")["poses"]
    import camera_detector as CD, pack_day, data_l2
    cb = CD.geo_calib("2026.05.08"); Tcl = np.array(cb["camera_to_lidar"], float)
    YAW = dict(cam=float(np.arctan2(Tcl[1, 2], Tcl[0, 2])), radar=0.0, lidar=0.0)
    t0 = P[0]["t"]; N = len(P)
    zs = np.load(f"{WD}/sensors.npz", allow_pickle=True)
    azc, rfc = pack_day._convention("2026.05.08")
    fa = np.arange(0, 30.0001, 0.1); la = np.arange(-30, 30.0001, 0.1)
    RPTS = [data_l2.radar_points(pack_day.ra_to_bev(zs["radar_ra"][k].astype(np.float32), azc, rfc),
                                 la, fa) for k in range(N)]

    def in_wedge(pw, pose, sensor):
        half, rm = WEDGE[sensor]
        d = rot(pose[2]).T @ (pw - pose[:2]); r = np.hypot(*d)
        if r > rm: return False
        if half >= np.pi - 1e-6: return True
        return abs((np.arctan2(d[1], d[0]) - YAW[sensor] + np.pi) % (2 * np.pi) - np.pi) <= half

    # live = sensors whose detections count live AND define the suite's FOV;
    # certify = radar used ONLY to range-certify far cam writes (not live, not FOV)
    VIZ_CFG = os.environ.get("WC_VIZ", "")                      # set to a suite name to render it
    frames_viz = []
    CFG = {"cam":          dict(live=["cam"], certify=False),
           "radar":        dict(live=["radar"], certify=False),
           "lidar":        dict(live=["lidar"], certify=False),
           "camrad-live":  dict(live=["cam", "radar"], certify=False),
           "camrad-cert":  dict(live=["cam"], certify=True),
           "camrad-full":  dict(live=["cam", "radar"], certify=True, readgate=["cam"])}
    print(f"{win}: {N} frames | identical bank/decay/read for all suites\n")
    print(f"{'suite':<12} {'LIVE':>6} {'BLIND':>6} {'RETRIEVAL':>9} {'WORLD':>6} {'ghosts':>7}   "
          f"(n in-FOV / blind / known / all)")
    results = {}
    for cfg, cc in CFG.items():
        if only and cfg != only: continue
        sensors, certify = cc["live"], cc["certify"]
        readgate = cc.get("readgate", sensors)
        bank = []
        S = dict(fov=0, fovh=0, all=0, liv=0, lm=0, bl=0, blm=0, gh=0, qf=0, n1=0, n2=0,
                 tp=0, fpd=0); apbag = []
        past_known = []
        for i in range(N):
            t = P[i]["t"] - t0; pose = poses[i]
            Rw, pw = rot(pose[2]), pose[:2]
            suite_fov = lambda g: any(in_wedge(g, pose, sn) for sn in sensors)
            read_fov = lambda g: any(in_wedge(g, pose, sn) for sn in readgate)
            live, writes = [], []
            if "cam" in sensors:
                for b, sc_, lb in P[i]["props"]["cam"]:
                    c = Rw @ ctr(b) + pw; live.append((c, lb, sc_))
                    r_mono = np.hypot(*ctr(b))
                    if r_mono <= CAM_TRUST:
                        writes.append((c, sc_, lb))
                    elif "radar" in sensors or certify:
                        az_c = np.arctan2(ctr(b)[1], ctr(b)[0])
                        r_hat = radar_range_at(RPTS[i], az_c, r_mono)
                        if r_hat is not None:
                            writes.append((Rw @ (r_hat * np.array([np.cos(az_c), np.sin(az_c)])) + pw,
                                           sc_, lb))
            if "radar" in sensors:
                for b, sc_ in P[i]["props"]["radar"]:
                    c = Rw @ ctr(b) + pw; live.append((c, "car", sc_))
                    if sc_ >= WGATE: writes.append((c, sc_, "car"))
            if "lidar" in sensors:
                for b, sc_ in P[i]["props"]["lidar"]:
                    c = Rw @ ctr(b) + pw; live.append((c, "car", sc_)); writes.append((c, sc_, "car"))
            # ---- identical bank update ----
            for c, sc_, lb in writes:
                e = next((e for e in bank if np.hypot(*(e["p"] - c)) < ASSOC), None)
                if e is None:
                    bank.append(dict(p=c.copy(), p0=c.copy(), s=sc_, lb=lb, last=t, disp=0.0, n=1))
                else:
                    e["p"] = 0.7 * e["p"] + 0.3 * c
                    e["s"] = max(e["s"], sc_); e["last"] = t; e["n"] += 1
                    e["disp"] = max(e["disp"], float(np.hypot(*(e["p"] - e["p0"]))))
            for e in bank:                                      # identical negative evidence
                trust_here = suite_fov(e["p"]) and (
                    "radar" in sensors or "lidar" in sensors or
                    np.hypot(*(rot(pose[2]).T @ (e["p"] - pw))) <= CAM_TRUST)
                if trust_here and not any(np.hypot(*(e["p"] - c)) < ASSOC for c, _, _ in live):
                    e["s"] *= NEG
                    if e["n"] >= 4: e["s"] = max(e["s"], FLOOR)
            bank = [e for e in bank if e["s"] > 0.12]
            past_known += [c for c, _, _ in live] + [c for c, _, _ in writes]
            mem = []                                            # identical gated read
            for e in bank:
                tau = TAU_M if e["disp"] > 1.2 else (TAU_S if e["n"] >= NCONF else TAU_U)
                sc_ = e["s"] * np.exp(-(t - e["last"]) / tau)
                if sc_ >= EMIT and not any(in_wedge(e["p"], pose, sn) for sn in readgate):
                    if EMIT_HOOK is None:
                        mem.append((e["p"], e["lb"], sc_))
                    else:
                        out = EMIT_HOOK(e, pose, t, sc_)
                        if out is not None: mem.append((out[0], out[1], sc_))
            if cfg == VIZ_CFG:                                  # capture EVERY frame (incl. warm-up)
                ncam = len(P[i]["props"]["cam"]) if "cam" in sensors else 0
                camL = [c for c, _, _ in live[:ncam]]
                radL = [c for c, _, _ in live[ncam:]]
                gv = []
                for u in P[i]["union_gt"]:
                    gc = Rw @ ctr(u["box"]) + pw
                    if np.hypot(*(rot(pose[2]).T @ (gc - pw))) <= RMAX_EVAL: gv.append(gc)
                rec = [] if t < WARM else [g for g in gv if not in_wedge(g, pose, "cam")
                     and any(np.hypot(*(g - c)) <= 2.0 for c, _, _ in mem)]
                frames_viz.append((i, camL, radL, [c for c, _, _ in mem], gv, rec, t < WARM))
            if t < WARM or not P[i]["union_gt"]: continue
            S["qf"] += 1
            gts = []
            for u in P[i]["union_gt"]:
                gc = Rw @ ctr(u["box"]) + pw
                if np.hypot(*(rot(pose[2]).T @ (gc - pw))) > RMAX_EVAL: continue
                rad = radius_of(u["label"])
                h_l = any(np.hypot(*(gc - c)) <= rad for c, _, _ in live)
                h_m = any(np.hypot(*(gc - c)) <= rad for c, _, _ in mem)
                gts.append((gc, rad))
                S["all"] += 1; S["liv"] += h_l; S["lm"] += (h_l or h_m)
                if suite_fov(gc): S["fov"] += 1; S["fovh"] += h_l
                if not in_wedge(gc, pose, "cam"):               # STANDARD blind zone (cam wedge)
                    S["bl"] += 1; S["blm"] += h_m
                    if any(np.hypot(*(gc - c)) <= 2.5 for c in past_known):
                        S["n1"] += 1; S["n2"] += h_m
            for c, _, _ in mem:
                if not any(np.hypot(*(c - g)) <= 2.5 for g, _ in gts): S["gh"] += 1
            # ---- prediction-side scoring (precision / F1 / AP): live U memory, greedy by score ----
            preds = sorted([(p_sc, c) for c, _, p_sc in live] + [(p_sc, c) for c, _, p_sc in mem],
                           key=lambda x: -x[0])
            used = np.zeros(len(gts), bool)
            for p_sc, c in preds:
                bj, bd = -1, 1e9
                for j, (g, rad) in enumerate(gts):
                    if used[j]: continue
                    d = np.hypot(*(c - g))
                    if d <= rad and d < bd: bd, bj = d, j
                if bj >= 0:
                    used[bj] = True; S["tp"] += 1; apbag.append((p_sc, 1))
                else:
                    S["fpd"] += 1; apbag.append((p_sc, 0))

        S["prec"] = S["tp"] / max(1, S["tp"] + S["fpd"])
        rec_w = S["lm"] / max(1, S["all"])
        S["f1"] = 2 * S["prec"] * rec_w / max(1e-9, S["prec"] + rec_w)
        S["ap"] = voc_ap(apbag, S["all"])
        results[cfg] = S
        rtr = f"{S['n2']/S['n1']:>9.2f}" if S["n1"] else f"{'--':>9}"
        bld = f"{S['blm']/S['bl']:>6.2f}" if S["bl"] else f"{'--':>6}"
        print(f"{cfg:<12} {S['fovh']/max(1,S['fov']):>6.2f} {bld} {rtr} "
              f"{S['lm']/max(1,S['all']):>6.2f} {S['gh']/max(1,S['qf']):>7.2f} | "
              f"P {S['prec']:.2f} F1 {S['f1']:.2f} AP {S['ap']:.2f}   "
              f"({S['fov']}/{S['bl']}/{S['n1']}/{S['all']})")

    if VIZ_CFG and frames_viz:
        render(win, frames_viz, poses, YAW["cam"], VIZ_CFG)
    return results

def render(win, FV, poses, yawc, tag):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tempfile, subprocess, shutil
    WD = f"{MV}/{win}"
    zs = np.load(f"{WD}/sensors.npz", allow_pickle=True)
    pids = zs["pids"]
    N = len(poses)
    M = np.vstack([prep(zs["lidar"][k]) @ rot(poses[k, 2]).T + poses[k, :2]
                   for k in range(0, N, 4)])[::2]
    VD = f"{WD}/odometry_viz"; os.makedirs(VD, exist_ok=True)          # all window viz in ONE folder
    td = tempfile.mkdtemp(prefix="cmpflip_")
    best = max(FV, key=lambda f: len(f[5]))
    first_q = next(f for f in FV if not f[6])
    stills = {first_q[0]: "M8_S1_first_query", best[0]: "M8_S2_max_recovered", FV[-1][0]: "M8_S3_last"}
    for n, (i, camL, radL, mem, gts, rec, warm) in enumerate(FV):
        fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=105)
        ax.scatter(M[:, 1], M[:, 0], s=0.15, c="#DDE2E8", lw=0)
        pose = poses[i]; a0 = pose[2] + yawc
        for s_ in (-np.radians(34), np.radians(34)):
            d = rot(a0 + s_) @ [30, 0]
            ax.plot([pose[1], pose[1] + d[1]], [pose[0], pose[0] + d[0]], color="orange", lw=1.2)
        th = np.linspace(pose[2] - np.pi / 2, pose[2] + np.pi / 2, 50)
        ax.plot(pose[1] + 25 * np.sin(th), pose[0] + 25 * np.cos(th),
                color="blue", lw=.9, alpha=.45, ls="--")
        ax.plot(poses[:i + 1, 1], poses[:i + 1, 0], "-", color="#C44", lw=1)
        for g in gts: ax.scatter(g[1], g[0], marker="s", s=90, facecolor="none",
                                 edgecolor="red", lw=1.3)
        for c in radL: ax.scatter(c[1], c[0], marker="x", s=26, color="blue", lw=1.2)
        for c in camL: ax.scatter(c[1], c[0], marker="o", s=30, color="orange")
        for c in mem: ax.scatter(c[1], c[0], marker="D", s=26, color="green")
        for g in rec: ax.scatter(g[1], g[0], marker="o", s=240, facecolor="none",
                                 edgecolor="green", lw=2.2)
        ax.set_aspect("equal"); ax.invert_xaxis()
        ax.set_xlim(M[:, 1].max() + 4, M[:, 1].min() - 4)
        ax.set_ylim(M[:, 0].min() - 4, M[:, 0].max() + 4)
        ax.set_title(f"{tag}  pid {int(pids[i])}{'  [WARM-UP]' if warm else ''}  |  "
                     f"orange=cam o+FOV  blue=radar x+FOV  green=memory D/recovered O ({len(rec)})  "
                     f"red=GT", fontsize=8)
        fig.savefig(f"{td}/f{n:04d}.jpg")
        if i in stills: fig.savefig(f"{VD}/{stills[i]}.png", dpi=130)
        plt.close(fig)
    out = f"{td}/flipbook_{tag}.mp4"
    subprocess.run(["ffmpeg", "-y", "-framerate", "10", "-i", f"{td}/f%04d.jpg",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", out],
                   check=True, capture_output=True)
    shutil.copy(out, f"{VD}/M8_flipbook_{tag}.mp4"); shutil.rmtree(td)
    print(f"viz saved to {VD}", flush=True)


# ===================== CODEC (was memory_codec.py) =====================

DIM = 8
SPLIT_T = 0.6          # VAE trained on emissions from the first 60% of the window (by time)

def vec_of(e, pose, t, sc):
    """entry -> normalised 8-d sender-frame vector (what would be serialised)."""
    d = rot(pose[2]).T @ (e["p"] - pose[:2])
    veh = 0.0 if e["lb"] in ("person", "bicycle", "motorcycle") else 1.0
    return np.array([d[0] / 30, d[1] / 30, sc, (t - e["last"]) / 30,
                     veh, 1.0 - veh, min(e["n"], 10) / 10, min(e["disp"], 3.0) / 3], np.float32)

def devec(v, pose, lb_fallback="car"):
    d = np.array([v[0], v[1]]) * 30
    pw = rot(pose[2]) @ d + pose[:2]
    lb = "car" if v[4] >= v[5] else "person"
    return pw, lb

class VAE(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.mu = nn.Linear(64, zdim); self.lv = nn.Linear(64, zdim)
        self.dec = nn.Sequential(nn.Linear(zdim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, DIM))
        self.zdim = zdim

    def forward(self, x):
        h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv

    def roundtrip(self, v):
        """deterministic encode -> int8 quantise (scale 4sigma->127) -> decode."""
        with torch.no_grad():
            mu = self.mu(self.enc(torch.tensor(v[None])))
            q = torch.clamp(torch.round(mu / 4.0 * 127), -127, 127) * 4.0 / 127
            return self.dec(q).numpy()[0]

W_POS = 4.0            # position dominates downstream matching -> weighted reconstruction

def train_vae(X, zdim, steps=3000, beta=0.02):
    m = VAE(zdim); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    X = torch.tensor(X)
    w = torch.ones(DIM); w[0] = w[1] = W_POS
    for it in range(steps):
        xb = X[torch.randint(0, len(X), (256,))]
        xr, mu, lv = m(xb)
        rec = ((xr - xb) ** 2 * w).mean()
        kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean()
        loss = rec + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m

def codec_main(win):
    global EMIT_HOOK
    # ---- harvest: collect emission vectors (train = first 60% by time) ----
    bag_tr, bag_all = [], []
    t_end = [0.0]
    def collector(e, pose, t, sc):
        v = vec_of(e, pose, t, sc)
        bag_all.append((t, v)); t_end[0] = max(t_end[0], t)
        return (e["p"], e["lb"])                                # pass-through
    EMIT_HOOK = collector
    print("== harvest (camrad-full, pass-through) ==")
    base = compare_run(win, only="camrad-full")["camrad-full"]
    t_split = t_end[0] * SPLIT_T
    Xtr = np.stack([v for t, v in bag_all if t <= t_split])
    print(f"harvested {len(bag_all)} emissions, {len(Xtr)} train (t <= {t_split:.1f}s)\n")

    # ---- codecs ----
    def explicit_int8(e, pose, t, sc):
        v = vec_of(e, pose, t, sc)
        q = np.clip(np.round(v * 127), -127, 127) / 127         # 8 x int8
        return devec(q, pose)
    codecs = {"explicit-fp32": None, "explicit-int8": explicit_int8}
    for zd in (8, 4, 2):
        m = train_vae(Xtr, zd)
        err = []
        for t, v in bag_all:
            if t > t_split:
                r = m.roundtrip(v)
                err.append(30 * np.hypot(r[0] - v[0], r[1] - v[1]))
        print(f"latent-{zd}: held-out position reconstruction p50 {np.median(err):.2f} m "
              f"p90 {np.percentile(err, 90):.2f} m")
        codecs[f"latent-{zd}"] = (lambda mm: lambda e, pose, t, sc:
                                  devec(mm.roundtrip(vec_of(e, pose, t, sc)), pose))(m)

    # ---- eval each codec through the full pipeline ----
    BYTES = {"explicit-fp32": 32, "explicit-int8": 8, "latent-8": 8, "latent-4": 4, "latent-2": 2}
    print(f"\n{'codec':<14} {'B/entry':>7} {'LIVE':>6} {'BLIND':>6} {'RETRIEVAL':>9} {'WORLD':>6} {'ghosts':>7}")
    for name, hook in codecs.items():
        EMIT_HOOK = hook
        S = compare_run(win, only="camrad-full")["camrad-full"]
        rtr = S["n2"] / S["n1"] if S["n1"] else 0
        print(f"{name:<14} {BYTES[name]:>7} {S['fovh']/max(1,S['fov']):>6.2f} "
              f"{S['blm']/max(1,S['bl']):>6.2f} {rtr:>9.2f} {S['lm']/max(1,S['all']):>6.2f} "
              f"{S['gh']/max(1,S['qf']):>7.2f}")
    EMIT_HOOK = None




# ===================== JSCC prototype: noise-tolerant learned code vs explicit+FEC =====================
# Channel = BSC (bit flips WITHIN delivered packets). Budget-matched at 32 bits/entry:
#   exp2B-naked   16 b  x,y uint8, no protection      (teleports on MSB flips)
#   exp2B+Hamming 32 b  x,y uint8 + Hamming(7,4)      (corrects 1 flip / 7-bit block, then cliffs)
#   latent4-plain 32 b  yesterday's VAE, noise-naive
#   latent4-JSCC  32 b  trained WITH bit-flips on the code (p ~ U(0, 0.12), straight-through)

def _u8(x):   return np.clip(np.round((np.clip(x, -1, 1) + 1) * 127.5), 0, 255).astype(np.uint8)
def _un8(b):  return b.astype(np.float32) / 127.5 - 1.0

def bsc(bits, pflip, rng): return bits ^ (rng.random(bits.shape) < pflip)

H_G = np.array([[1,1,0,1],[1,0,1,1],[1,0,0,0],[0,1,1,1],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.uint8)
H_H = np.array([[1,0,1,0,1,0,1],[0,1,1,0,0,1,1],[0,0,0,1,1,1,1]], np.uint8)

def hamming_enc(nib):  return (H_G @ nib) % 2
def hamming_dec(cw):
    syn = (H_H @ cw) % 2
    pos = syn[0] + 2 * syn[1] + 4 * syn[2]
    if pos: cw = cw.copy(); cw[pos - 1] ^= 1
    return cw[[2, 4, 5, 6]]

def tx_exp(v, pflip, rng, fec):
    """x,y uint8 -> bits (+Hamming if fec) -> BSC -> decode -> v' (other fields: veh default)."""
    by = _u8(v[:2])
    bits = np.unpackbits(by)
    if fec:
        nibs = bits.reshape(4, 4)
        code = np.concatenate([hamming_enc(n) for n in nibs])
        code = bsc(code, pflip, rng)
        bits = np.concatenate([hamming_dec(code[i*7:(i+1)*7]) for i in range(4)])
    else:
        bits = bsc(bits, pflip, rng)
    r = np.zeros(DIM, np.float32)
    r[:2] = _un8(np.packbits(bits)); r[4] = 1.0
    return r

class JSCC(nn.Module):
    """deterministic AE, 4-d code, uint8 on the wire, trained with the BSC inside the loop (STE)."""
    def __init__(self, zdim=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, zdim), nn.Tanh())
        self.dec = nn.Sequential(nn.Linear(zdim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, DIM))
        self.zdim = zdim

    def channel(self, z, pflip, rng):
        by = _u8(z.detach().numpy().ravel())
        bits = bsc(np.unpackbits(by), pflip, rng)
        zq = torch.tensor(_un8(np.packbits(bits)).reshape(z.shape))
        return z + (zq - z).detach()                            # straight-through

def train_jscc(X, steps=4000, pmax=0.12, zdim=4, with_noise=True):
    m = JSCC(zdim); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    rng = np.random.default_rng(0)
    X = torch.tensor(X); w = torch.ones(DIM); w[0] = w[1] = W_POS
    for it in range(steps):
        xb = X[torch.randint(0, len(X), (256,))]
        z = m.enc(xb)
        z = m.channel(z, rng.uniform(0, pmax) if with_noise else 0.0, rng)
        loss = ((m.dec(z) - xb) ** 2 * w).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m

def tx_latent(m, v, pflip, rng):
    with torch.no_grad():
        z = m.enc(torch.tensor(v[None]))
        by = _u8(z.numpy().ravel())
        bits = bsc(np.unpackbits(by), pflip, rng)
        zq = torch.tensor(_un8(np.packbits(bits))[None])
        return m.dec(zq).numpy()[0]

def jscc_main(win):
    global EMIT_HOOK
    bag = []
    def collector(e, pose, t, sc):
        bag.append(vec_of(e, pose, t, sc)); return (e["p"], e["lb"])
    EMIT_HOOK = collector
    print("== harvest ==")
    compare_run(win, only="camrad-full")
    X = np.stack(bag)
    print(f"{len(X)} emission vectors\n== train (plain + JSCC) ==")
    m_plain = train_jscc(X, with_noise=False)
    m_jscc = train_jscc(X, with_noise=True)
    CODECS = {
        "exp2B-naked(16b)":  lambda v, p, rng: tx_exp(v, p, rng, fec=False),
        "exp2B+Ham(32b)":    lambda v, p, rng: tx_exp(v, p, rng, fec=True),
        "latent4-plain(32b)": lambda v, p, rng: tx_latent(m_plain, v, p, rng),
        "latent4-JSCC(32b)": lambda v, p, rng: tx_latent(m_jscc, v, p, rng),
    }
    print(f"\n{'codec':<19} {'pflip':>6} {'BLIND':>6} {'RETRIEVAL':>9} {'WORLD':>6} {'ghosts':>7}")
    for name, tx in CODECS.items():
        for pflip in (0.0, 0.02, 0.05, 0.10, 0.15):
            rng = np.random.default_rng(7)
            def hook(e, pose, t, sc, tx=tx, pflip=pflip, rng=rng):
                return devec(tx(vec_of(e, pose, t, sc), pflip, rng), pose)
            EMIT_HOOK = hook
            S = compare_run(win, only="camrad-full")["camrad-full"]
            rtr = S["n2"] / S["n1"] if S["n1"] else 0
            print(f"{name:<19} {pflip:>6.2f} {S['blm']/max(1,S['bl']):>6.2f} {rtr:>9.2f} "
                  f"{S['lm']/max(1,S['all']):>6.2f} {S['gh']/max(1,S['qf']):>7.2f} | "
                  f"P {S['prec']:.2f} F1 {S['f1']:.2f} AP {S['ap']:.2f}", flush=True)
    EMIT_HOOK = None

# ===================== AWGN analog channel: DeepJSCC vs digital (16-QAM) at matched channel uses =====================
# The code vector IS the transmitted waveform: z in R^4 (tanh), power-normalised per entry
# (E|symbol|^2 = 1), y = z + AWGN(sigma), decoder recovers from y directly. Digital baselines modulate
# the explicit bits onto the SAME channel: 16-QAM (4 bits/symbol, unit avg power, hard demod).
#   analog-JSCC      4 channel uses   trained with SNR ~ U(0, 20) dB in the loop
#   16QAM-naked      4 channel uses   16-bit x,y payload
#   16QAM+Hamming    8 channel uses   28-bit protected payload (corrects 1 flip / block)

QAM = np.array([-3, -1, 1, 3], np.float32) / np.sqrt(5)         # 2-bit levels, UNIT power per real symbol
                                                                # (matches the analog code's E|sym|^2=1)

def qam_mod(bits):
    sym = []
    for i in range(0, len(bits), 4):
        b = bits[i:i + 4]
        sym += [QAM[b[0] * 2 + b[1]], QAM[b[2] * 2 + b[3]]]     # I, Q as two reals
    return np.array(sym, np.float32)

def qam_demod(sig):
    bits = []
    for v in sig:
        k = int(np.argmin(np.abs(QAM - v)))
        bits += [k // 2, k % 2]
    return np.array(bits, np.uint8)

def tx_exp_awgn(v, sigma, rng, fec):
    by = _u8(v[:2]); bits = np.unpackbits(by)
    if fec:
        bits = np.concatenate([hamming_enc(bits.reshape(4, 4)[i]) for i in range(4)])
        bits = np.concatenate([bits, np.zeros(4, np.uint8)])    # pad 28 -> 32 bits = 8 QAM symbols x2
    sig = qam_mod(bits)
    y = sig + rng.normal(0, sigma, sig.shape).astype(np.float32)
    rb = qam_demod(y)
    if fec:
        rb = np.concatenate([hamming_dec(rb[i * 7:(i + 1) * 7]) for i in range(4)])
    r = np.zeros(DIM, np.float32)
    r[:2] = _un8(np.packbits(rb[:16])); r[4] = 1.0
    return r

class AJSCC(nn.Module):
    """analog DeepJSCC: 4 real symbols, per-entry power normalisation, AWGN in the training loop."""
    def __init__(self, k=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, k))
        self.dec = nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, DIM))
        self.k = k

    def norm(self, z):                                          # E|symbol|^2 = 1 per entry
        return z * np.sqrt(self.k) / (z.norm(dim=-1, keepdim=True) + 1e-8)

def train_ajscc(X, steps=4000, k=4, with_noise=True):
    m = AJSCC(k); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    X = torch.tensor(X); w = torch.ones(DIM); w[0] = w[1] = W_POS
    for it in range(steps):
        xb = X[torch.randint(0, len(X), (256,))]
        z = m.norm(m.enc(xb))
        snr = np.random.uniform(-12, 15) if with_noise else 40.0   # 40 dB ~ clean (naive)
        y = z + torch.randn_like(z) * float(10 ** (-snr / 20))
        loss = ((m.dec(y) - xb) ** 2 * w).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m

def tx_digjscc_qam(m, v, sigma, rng):
    """REAL digital chain for the learned code: z -> 32 bits -> 8 QAM symbols -> AWGN -> demod -> decode.
    (m = the BSC-trained digital JSCC; demod errors ARE the bit flips it was trained for.)"""
    with torch.no_grad():
        z = m.enc(torch.tensor(v[None]))
        bits = np.unpackbits(_u8(z.numpy().ravel()))
        y = qam_mod(bits) + rng.normal(0, sigma, 16).astype(np.float32)
        zq = torch.tensor(_un8(np.packbits(qam_demod(y)))[None])
        return m.dec(zq).numpy()[0]

def tx_ajscc(m, v, sigma, rng):
    with torch.no_grad():
        z = m.norm(m.enc(torch.tensor(v[None])))
        y = z + torch.tensor(rng.normal(0, sigma, (1, m.k)).astype(np.float32))
        return m.dec(y).numpy()[0]

def awgn_main(win):
    global EMIT_HOOK
    bag = []
    def collector(e, pose, t, sc):
        bag.append(vec_of(e, pose, t, sc)); return (e["p"], e["lb"])
    EMIT_HOOK = collector
    print("== harvest ==")
    compare_run(win, only="camrad-full")
    X = np.stack(bag)
    print(f"{len(X)} emission vectors\n== train analog JSCC ==")
    m = train_ajscc(X)
    m_naive = train_ajscc(X, with_noise=False)
    m_dig = train_jscc(X, with_noise=True)                     # flip-trained digital latent
    CODECS = {"analog-JSCC(4use)": lambda v, s_, rng: tx_ajscc(m, v, s_, rng),
              "analog-naive(4use)": lambda v, s_, rng: tx_ajscc(m_naive, v, s_, rng),
              "digJSCC-QAM(8use)": lambda v, s_, rng: tx_digjscc_qam(m_dig, v, s_, rng),
              "16QAM-naked(4use)": lambda v, s_, rng: tx_exp_awgn(v, s_, rng, fec=False),
              "16QAM+Ham(8use)":   lambda v, s_, rng: tx_exp_awgn(v, s_, rng, fec=True)}
    print(f"\n{'codec':<18} {'SNR dB':>6} {'BLIND':>6} {'RETRIEVAL':>9} {'WORLD':>6} {'ghosts':>7}")
    for name, tx in CODECS.items():
        for snr in (10, 5, 0, -5, -10):
            sigma = float(10 ** (-snr / 20))
            rng = np.random.default_rng(7)
            def hook(e, pose, t, sc, tx=tx, sigma=sigma, rng=rng):
                return devec(tx(vec_of(e, pose, t, sc), sigma, rng), pose)
            EMIT_HOOK = hook
            S = compare_run(win, only="camrad-full")["camrad-full"]
            rtr = S["n2"] / S["n1"] if S["n1"] else 0
            print(f"{name:<18} {snr:>6} {S['blm']/max(1,S['bl']):>6.2f} {rtr:>9.2f} "
                  f"{S['lm']/max(1,S['all']):>6.2f} {S['gh']/max(1,S['qf']):>7.2f} | "
                  f"P {S['prec']:.2f} F1 {S['f1']:.2f} AP {S['ap']:.2f}", flush=True)
    EMIT_HOOK = None

def qam_main(win):
    global EMIT_HOOK
    """CLEAN ABLATION at fixed modulation: everything rides 16-QAM through the same AWGN chain;
    only the CODING varies (explicit / explicit+FEC / learned). Verifies the encoder saves QAM."""
    bag = []
    def collector(e, pose, t, sc):
        bag.append(vec_of(e, pose, t, sc)); return (e["p"], e["lb"])
    EMIT_HOOK = collector
    print("== harvest ==")
    compare_run(win, only="camrad-full")
    X = np.stack(bag)
    print(f"{len(X)} emission vectors\n== train flip-trained digital latent ==")
    m_dig = train_jscc(X, with_noise=True)
    m_dig_naive = train_jscc(X, with_noise=False)
    CODECS = {"16QAM explicit naked (4use)": lambda v, s_, rng: tx_exp_awgn(v, s_, rng, fec=False),
              "16QAM explicit+Hamming (8use)": lambda v, s_, rng: tx_exp_awgn(v, s_, rng, fec=True),
              "16QAM learned noise-naive (8use)": lambda v, s_, rng: tx_digjscc_qam(m_dig_naive, v, s_, rng),
              "16QAM learned noise-trained (8use)": lambda v, s_, rng: tx_digjscc_qam(m_dig, v, s_, rng)}
    print(f"\n{'codec':<34} {'SNR dB':>6} {'BLIND':>6} {'RETRIEVAL':>9} {'WORLD':>6} {'ghosts':>7}")
    for name, tx in CODECS.items():
        for snr in (10, 5, 0, -5, -10):
            sigma = float(10 ** (-snr / 20))
            rng = np.random.default_rng(7)
            def hook(e, pose, t, sc, tx=tx, sigma=sigma, rng=rng):
                return devec(tx(vec_of(e, pose, t, sc), sigma, rng), pose)
            EMIT_HOOK = hook
            S = compare_run(win, only="camrad-full")["camrad-full"]
            rtr = S["n2"] / S["n1"] if S["n1"] else 0
            print(f"{name:<34} {snr:>6} {S['blm']/max(1,S['bl']):>6.2f} {rtr:>9.2f} "
                  f"{S['lm']/max(1,S['all']):>6.2f} {S['gh']/max(1,S['qf']):>7.2f} | "
                  f"P {S['prec']:.2f} F1 {S['f1']:.2f} AP {S['ap']:.2f}", flush=True)
    EMIT_HOOK = None


# ===================== CLI =====================
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("odometry", "detect", "memory", "compare", "codec"):
        print(__doc__); sys.exit(1)
    cmd = sys.argv[1]
    win = sys.argv[2] if len(sys.argv) > 2 else "win_8825-9175"
    rest = sys.argv[3:]
    if cmd == "odometry":
        odo_viz(win) if "viz" in rest else odo_build(win)
    elif cmd == "detect":
        detect_main(win)
    elif cmd == "memory":
        memory_run(win, viz="viz" in rest, source="lidar1hz" if "lidar1hz" in rest else "cam")
    elif cmd == "compare":
        compare_run(win)
    elif cmd == "codec":
        if "jscc" in rest: jscc_main(win)
        elif "qam" in rest: qam_main(win)
        elif "awgn" in rest: awgn_main(win)
        else: codec_main(win)
