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
