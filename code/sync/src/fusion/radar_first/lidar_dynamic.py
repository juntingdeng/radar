"""Phase 4 helper — world-frame LiDAR dynamic-point extraction.

Voxel temporal persistence: with clean odometry, points in voxels occupied across
many frames are static; voxels lit only transiently are dynamic (movers/sweep edges).
Returns per-frame world-XY of dynamic points, used to corroborate radar movers.
Requires good poses (Phase 1) — with identity poses everything reads dynamic.
"""
import numpy as np
from collections import defaultdict


def dynamic_points_per_frame(scans, poses, near=25.0, vox=0.3,
                             slab=(0.3, 2.2), static_frac=0.50, dyn_frac=0.15, sub=5000):
    """-> list (len N) of Mx2 world-XY dynamic points per frame."""
    N = len(scans)
    a0 = scans[0]
    gz = np.percentile(a0[np.linalg.norm(a0[:, :2], axis=1) < near][:, 2], 3)
    W, F = [], []
    for i, a in enumerate(scans):
        m = (np.linalg.norm(a[:, :2], axis=1) < near) & (a[:, 2] > gz + slab[0]) & (a[:, 2] < gz + slab[1])
        a = a[m]
        if len(a) > sub:
            a = a[:: max(1, len(a) // sub)]
        h = np.c_[a, np.ones(len(a))]
        W.append((poses[i] @ h.T).T[:, :3]); F.append(np.full(len(a), i))
    W = np.vstack(W); F = np.concatenate(F)
    key = np.floor(W / vox).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    vf = defaultdict(set)
    for vid, f in zip(inv, F):
        vf[int(vid)].add(int(f))
    occ = np.array([len(vf[i]) for i in range(inv.max() + 1)])[inv]
    dyn = (occ >= 1) & (occ <= dyn_frac * N)
    Wd, Fd = W[dyn][:, :2], F[dyn]
    return [Wd[Fd == i] for i in range(N)]


def support_fraction(track_hist, dyn_per_frame, radius=2.5):
    """Fraction of a track's frames that have a LiDAR-dynamic point within `radius`."""
    h = np.array(track_hist)
    hit = 0
    for f, x, y in h[:, :3]:
        dp = dyn_per_frame[int(f)]
        if len(dp) and np.min(np.hypot(dp[:, 0] - x, dp[:, 1] - y)) <= radius:
            hit += 1
    return hit / len(h)
