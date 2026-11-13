"""Rigid-transform solvers for sensor extrinsic calibration.

Given hand-picked point correspondences between two sensor frames, estimate the
rigid transform that maps *source* points onto *target* points:

    target = R @ source + t          (rotation R, translation t; no scaling)

- ``umeyama_3d`` — full 3D rigid transform (3 DoF rotation + 3 DoF translation)
  from >= 3 non-collinear 3D<->3D correspondences. Used for camera -> lidar,
  where camera 3D comes from depth back-projection and lidar 3D from raw XYZ.
- ``umeyama_2d`` — planar rigid transform (yaw + XY translation) from >= 2
  correspondences. Used for radar -> lidar, where radar has no reliable
  elevation, so calibration is done in the top-down (forward, lateral) plane.

Both are the classic Umeyama / Kabsch least-squares solution with the scale
term fixed to 1 (rigid, not similarity). A reflection guard keeps ``det(R)=+1``.

Run ``python rigid_solve.py --self_test`` for a headless correctness check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RigidFit:
    """Result of a rigid-transform fit."""

    transform: np.ndarray  # (D+1, D+1) homogeneous matrix
    rotation: np.ndarray  # (D, D)
    translation: np.ndarray  # (D,)
    residuals: np.ndarray  # per-correspondence Euclidean error (target units)
    rms: float  # root-mean-square residual
    max_error: float  # worst single residual
    n_points: int

    def summary(self) -> str:
        return (
            f"{self.n_points} correspondences | "
            f"RMS={self.rms * 1e3:.1f} mm, max={self.max_error * 1e3:.1f} mm"
        )


def _kabsch(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R, t) with target ~= R @ source + t (least squares, no scale)."""
    src_c = source.mean(axis=0)
    tgt_c = target.mean(axis=0)
    src = source - src_c
    tgt = target - tgt_c

    cov = src.T @ tgt  # (D, D)
    u, _s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    # Reflection guard: force a proper rotation (det = +1).
    correction = np.eye(cov.shape[0])
    correction[-1, -1] = d
    rot = vt.T @ correction @ u.T
    trans = tgt_c - rot @ src_c
    return rot, trans


def _fit(source: np.ndarray, target: np.ndarray, dim: int) -> RigidFit:
    source = np.asarray(source, dtype=np.float64).reshape(-1, dim)
    target = np.asarray(target, dtype=np.float64).reshape(-1, dim)
    if source.shape != target.shape:
        raise ValueError(
            f"source {source.shape} and target {target.shape} must match, dim={dim}"
        )
    n = source.shape[0]
    min_pts = dim  # 3 for 3D, 2 for 2D (2 gives an exact planar fit)
    if n < min_pts:
        raise ValueError(f"Need >= {min_pts} correspondences for {dim}D fit, got {n}.")

    rot, trans = _kabsch(source, target)
    mapped = source @ rot.T + trans
    residuals = np.linalg.norm(mapped - target, axis=1)

    transform = np.eye(dim + 1, dtype=np.float64)
    transform[:dim, :dim] = rot
    transform[:dim, dim] = trans
    return RigidFit(
        transform=transform,
        rotation=rot,
        translation=trans,
        residuals=residuals,
        rms=float(np.sqrt(np.mean(residuals**2))) if n else float("nan"),
        max_error=float(residuals.max()) if n else float("nan"),
        n_points=n,
    )


def umeyama_3d(source: np.ndarray, target: np.ndarray) -> RigidFit:
    """Fit a 3D rigid transform: target ~= R(3x3) @ source + t(3)."""
    return _fit(source, target, 3)


def umeyama_2d(source: np.ndarray, target: np.ndarray) -> RigidFit:
    """Fit a planar rigid transform: target ~= R(2x2) @ source + t(2)."""
    return _fit(source, target, 2)


def fit_yaw_deg(fit: RigidFit) -> float:
    """Planar yaw (deg) of a 2D RigidFit's rotation."""
    return float(np.degrees(np.arctan2(fit.rotation[1, 0], fit.rotation[0, 0])))


def yaw_stability_2d(solver, source, target, *, n_boot: int = 2000, seed: int = 0):
    """Bootstrap + leave-one-out spread of the fitted yaw (deg).

    ``solver`` is any ``(src, tgt) -> RigidFit`` (e.g. ``umeyama_2d``). Returns a dict
    with the point estimate and its empirical spread so a caller can judge whether the
    yaw is actually constrained by the data (tight spread) or not (wide/railed). Yaw is
    the azimuth-derived quantity, so for sparse mmWave correspondences its uncertainty is
    what decides whether a solve is trustworthy.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    n = source.shape[0]
    full = fit_yaw_deg(solver(source, target))
    loo = np.array(
        [fit_yaw_deg(solver(np.delete(source, i, 0), np.delete(target, i, 0)))
         for i in range(n)]
    ) if n >= 3 else np.array([full])
    rgen = np.random.default_rng(seed)
    boot = []
    if n >= 3:
        for _ in range(int(n_boot)):
            idx = rgen.integers(0, n, n)
            try:
                boot.append(fit_yaw_deg(solver(source[idx], target[idx])))
            except Exception:  # noqa: BLE001 - degenerate resample, skip
                pass
    boot = np.array(boot) if boot else np.array([full])
    return {
        "yaw_deg": full,
        "boot_std": float(np.std(boot)),
        "boot_p16": float(np.percentile(boot, 16)),
        "boot_p84": float(np.percentile(boot, 84)),
        "loo": loo,
        "loo_std": float(np.std(loo)),
        "loo_min": float(loo.min()),
        "loo_max": float(loo.max()),
    }


def weighted_umeyama_2d(
    source: np.ndarray,
    target: np.ndarray,
    *,
    sigma_ratio: float = 10.0,
    loss: str = "soft_l1",
) -> RigidFit:
    """Planar fit that trusts the radar SOURCE point's range >> its azimuth.

    A mmWave radar measures range precisely but azimuth (cross-range) coarsely, so a
    correspondence's ``source`` point has small radial uncertainty and large tangential
    uncertainty. This solves (yaw, tx, ty) minimizing the residual with the tangential
    (azimuth) component down-weighted by ``sigma_ratio`` relative to the radial (range)
    component -- so azimuth noise no longer poisons the fit the way plain Umeyama does.

    ``loss`` is a scipy.least_squares robust loss ('soft_l1' by default) for outlier
    tolerance. Residuals/RMS in the result are the plain isotropic errors (meters), for
    comparability with ``umeyama_2d``.
    """
    from scipy.optimize import least_squares

    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape:
        raise ValueError(f"source {source.shape} and target {target.shape} must match")
    n = source.shape[0]
    if n < 2:
        raise ValueError(f"Need >= 2 correspondences, got {n}.")

    seed = umeyama_2d(source, target)
    yaw0 = float(np.arctan2(seed.rotation[1, 0], seed.rotation[0, 0]))
    rng = np.linalg.norm(source, axis=1)
    rng[rng < 1e-6] = 1e-6
    u = source / rng[:, None]                       # radial unit (radar frame)
    v = np.column_stack([-u[:, 1], u[:, 0]])        # tangential (azimuth) unit
    sigma_t = max(float(sigma_ratio), 1e-6)

    def residual(params):
        yaw, tx, ty = params
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        e = source @ rot.T + np.array([tx, ty]) - target      # (N, 2) target frame
        e_rad = np.sum(e * (u @ rot.T), axis=1)               # along source range dir
        e_tan = np.sum(e * (v @ rot.T), axis=1)               # along source azimuth dir
        return np.concatenate([e_rad, e_tan / sigma_t])

    sol = least_squares(
        residual, np.array([yaw0, seed.translation[0], seed.translation[1]]), loss=loss
    )
    yaw, tx, ty = sol.x
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    trans = np.array([tx, ty])
    residuals = np.linalg.norm(source @ rot.T + trans - target, axis=1)
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = rot
    transform[:2, 2] = trans
    return RigidFit(
        transform=transform,
        rotation=rot,
        translation=trans,
        residuals=residuals,
        rms=float(np.sqrt(np.mean(residuals**2))),
        max_error=float(residuals.max()),
        n_points=n,
    )


def _weighted_translation(source, target, u, v, yaw, w_rad, w_tan):
    """Closed-form range-weighted translation for a FIXED yaw. Returns (t, cost)."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    rad = u @ rot.T
    tan = v @ rot.T
    sp = source @ rot.T
    d = target - sp
    A = w_rad * (rad.T @ rad) + w_tan * (tan.T @ tan)
    bb = (w_rad * (rad * np.sum(rad * d, axis=1)[:, None]).sum(0)
          + w_tan * (tan * np.sum(tan * d, axis=1)[:, None]).sum(0))
    t = np.linalg.solve(A, bb)
    e = sp + t - target
    cost = float(np.sum(w_rad * np.sum(e * rad, axis=1) ** 2
                        + w_tan * np.sum(e * tan, axis=1) ** 2))
    return t, cost


def yaw_bruteforce_2d(
    source: np.ndarray,
    target: np.ndarray,
    *,
    yaw_lo_deg: float,
    yaw_hi_deg: float,
    step_deg: float = 0.25,
    sigma_ratio: float = 10.0,
    return_curve: bool = False,
):
    """Bounded yaw brute-force with a closed-form range-weighted translation per yaw.

    mmWave yaw is the azimuth-derived quantity, so a free fit can run to implausible
    angles. Here yaw is swept over a physically-plausible ``[yaw_lo_deg, yaw_hi_deg]``;
    for each candidate the optimal translation is solved in closed form (range trusted
    >> azimuth via ``sigma_ratio``), and the lowest-cost yaw wins. The cost-vs-yaw curve
    (returned with ``return_curve``) shows how sharply yaw is actually constrained.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    n = source.shape[0]
    rng = np.linalg.norm(source, axis=1)
    rng[rng < 1e-6] = 1e-6
    u = source / rng[:, None]
    v = np.column_stack([-u[:, 1], u[:, 0]])
    w_rad, w_tan = 1.0, 1.0 / max(float(sigma_ratio), 1e-6) ** 2

    yaws = np.deg2rad(np.arange(yaw_lo_deg, yaw_hi_deg + 1e-9, step_deg))
    costs = np.empty(yaws.size)
    ts = np.empty((yaws.size, 2))
    for k, yaw in enumerate(yaws):
        ts[k], costs[k] = _weighted_translation(source, target, u, v, yaw, w_rad, w_tan)
    k = int(np.argmin(costs))
    yaw, t = float(yaws[k]), ts[k]

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    residuals = np.linalg.norm(source @ rot.T + t - target, axis=1)
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = rot
    transform[:2, 2] = t
    fit = RigidFit(
        transform=transform, rotation=rot, translation=t, residuals=residuals,
        rms=float(np.sqrt(np.mean(residuals**2))), max_error=float(residuals.max()),
        n_points=n,
    )
    if return_curve:
        return fit, np.degrees(yaws), costs
    return fit


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Nominal camera-optical (X right, Y down, Z fwd) -> lidar sensor (X fwd, Y left, Z up).
IDEAL_CAM_TO_LIDAR = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


def level_constrained_fit(
    source: np.ndarray,
    target: np.ndarray,
    *,
    ideal_rotation: np.ndarray = IDEAL_CAM_TO_LIDAR,
    max_pitch_deg: float = 25.0,
    max_yaw_deg: float = 45.0,
) -> RigidFit:
    """Rigid fit with ROLL (rotation about the lidar forward axis) locked to zero.

    When both sensors are physically level, the camera->lidar rotation is
    ``Rz(yaw) @ Ry(pitch) @ ideal`` — there is no roll about forward. A full
    Umeyama solve, given correspondences clustered near one height plane, will
    happily spend a spurious ~10-15 deg roll to shave a little RMS off noise.
    Locking roll removes that failure mode; only pitch, yaw and translation are
    free. Solved by a coarse-to-fine grid over (yaw, pitch), optimal translation
    per candidate.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError(f"source {source.shape} and target {target.shape} must match")
    n = source.shape[0]
    if n < 2:
        raise ValueError(f"Need >= 2 correspondences, got {n}.")

    def eval_rt(yaw: float, pitch: float):
        rot = _rot_z(yaw) @ _rot_y(pitch) @ ideal_rotation
        trans = (target - source @ rot.T).mean(axis=0)
        res = np.linalg.norm(source @ rot.T + trans - target, axis=1)
        return rot, trans, res, float(np.sqrt(np.mean(res**2)))

    best = None
    # coarse grid then two refinement passes around the current best
    yaw_lo, yaw_hi = -np.radians(max_yaw_deg), np.radians(max_yaw_deg)
    pit_lo, pit_hi = -np.radians(max_pitch_deg), np.radians(max_pitch_deg)
    step = np.radians(1.0)
    for _ in range(3):
        yaws = np.arange(yaw_lo, yaw_hi + step, step)
        pits = np.arange(pit_lo, pit_hi + step, step)
        for y in yaws:
            for p in pits:
                rot, trans, res, rms = eval_rt(y, p)
                if best is None or rms < best[0]:
                    best = (rms, y, p, rot, trans, res)
        # zoom in around the best for the next pass
        _, by, bp, *_ = best
        yaw_lo, yaw_hi = by - step, by + step
        pit_lo, pit_hi = bp - step, bp + step
        step /= 20.0

    rms, _y, _p, rot, trans, res = best
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return RigidFit(
        transform=transform,
        rotation=rot,
        translation=trans,
        residuals=res,
        rms=float(rms),
        max_error=float(res.max()),
        n_points=n,
    )


def fit_with_rejection(solve_fn, source, target, *, max_residual: float, min_points: int):
    """Iteratively drop the single worst correspondence while its residual exceeds
    ``max_residual`` (meters), refitting each time. Returns (fit, keep_mask).

    Robust to a few gross mis-correspondences (e.g. a near camera object paired
    with a far lidar object) without letting them dominate the least-squares fit.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    keep = np.ones(source.shape[0], dtype=bool)
    fit = solve_fn(source, target)
    while int(keep.sum()) > min_points:
        f = solve_fn(source[keep], target[keep])
        fit = f
        worst = int(np.argmax(f.residuals))
        if f.residuals[worst] <= max_residual:
            break
        kept_idx = np.where(keep)[0]
        keep[kept_idx[worst]] = False  # drop the single worst kept point
    return fit, keep


def planar_fit_to_4x4(fit: RigidFit, z_offset: float = 0.0) -> np.ndarray:
    """Embed a 2D planar fit (forward, lateral) into a 4x4 sensor transform.

    The planar solve maps source (X=forward, Y=lateral) into target's plane.
    Z (up) is not observed by radar, so it is carried through with a fixed,
    user-supplied ``z_offset`` (meters) and identity rotation about the plane.
    """
    if fit.rotation.shape != (2, 2):
        raise ValueError("planar_fit_to_4x4 expects a 2D fit.")
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0], mat[0, 1] = fit.rotation[0, 0], fit.rotation[0, 1]
    mat[1, 0], mat[1, 1] = fit.rotation[1, 0], fit.rotation[1, 1]
    mat[0, 3] = fit.translation[0]
    mat[1, 3] = fit.translation[1]
    mat[2, 3] = float(z_offset)
    return mat


def _self_test() -> int:
    rng = np.random.default_rng(0)
    failures = 0

    # --- 3D: recover a known transform from noise-free correspondences ---
    theta = 0.7
    axis = np.array([0.2, -0.5, 1.0])
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    kmat = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    r_true = np.eye(3) + np.sin(theta) * kmat + (1 - np.cos(theta)) * (kmat @ kmat)
    t_true = np.array([1.5, -2.0, 0.3])
    src = rng.uniform(-5, 5, size=(12, 3))
    tgt = src @ r_true.T + t_true
    fit = umeyama_3d(src, tgt)
    if not np.allclose(fit.rotation, r_true, atol=1e-9):
        print("FAIL 3D rotation"); failures += 1
    if not np.allclose(fit.translation, t_true, atol=1e-9):
        print("FAIL 3D translation"); failures += 1
    if fit.rms > 1e-9:
        print(f"FAIL 3D rms {fit.rms}"); failures += 1

    # --- 3D with noise: residuals reported, transform still close ---
    tgt_noisy = tgt + rng.normal(0, 0.01, size=tgt.shape)
    fit_n = umeyama_3d(src, tgt_noisy)
    if abs(np.linalg.det(fit_n.rotation) - 1.0) > 1e-9:
        print("FAIL 3D noisy det != 1"); failures += 1
    if fit_n.rms <= 0 or fit_n.rms > 0.05:
        print(f"FAIL 3D noisy rms {fit_n.rms}"); failures += 1

    # --- 2D: recover yaw + translation ---
    a = 0.35
    r2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    t2 = np.array([0.4, -1.1])
    s2 = rng.uniform(-10, 10, size=(6, 2))
    d2 = s2 @ r2.T + t2
    fit2 = umeyama_2d(s2, d2)
    if not np.allclose(fit2.rotation, r2, atol=1e-9):
        print("FAIL 2D rotation"); failures += 1
    if not np.allclose(fit2.translation, t2, atol=1e-9):
        print("FAIL 2D translation"); failures += 1
    m44 = planar_fit_to_4x4(fit2, z_offset=0.12)
    if abs(m44[2, 3] - 0.12) > 1e-12 or abs(np.linalg.det(m44) - 1.0) > 1e-9:
        print("FAIL planar_fit_to_4x4"); failures += 1

    # --- reflection guard: mirrored data must not yield det=-1 ---
    mirror = src.copy()
    mirror[:, 0] *= -1
    fitm = umeyama_3d(src, mirror)
    if abs(np.linalg.det(fitm.rotation) - 1.0) > 1e-9:
        print("FAIL reflection guard det != +1"); failures += 1

    if failures == 0:
        print("rigid_solve self-test: ALL PASS")
    else:
        print(f"rigid_solve self-test: {failures} FAILURE(S)")
    return failures


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self_test", action="store_true", help="Run headless correctness checks.")
    args = p.parse_args()
    if args.self_test:
        raise SystemExit(_self_test())
    p.print_help()
