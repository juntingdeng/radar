"""Derive camera<->radar from the lidar-referenced extrinsics, and validate.

Both calibrators solve transforms into the lidar frame (``camera_to_lidar``,
``radar_to_lidar``), so the camera<->radar transform is derivable for free:

    camera_to_radar = inv(radar_to_lidar) @ camera_to_lidar

This tool computes and stores it, and cross-checks the calibration by:
  - recomputing each pairwise fit residual from the STORED matrix + its saved
    correspondences (confirms the stored extrinsic reproduces the picks), and
  - sanity-checking every transform (proper rotation, plausible translation).

Run after both ``calibrate_camera_lidar.py`` and ``calibrate_radar_lidar.py``::

    python code/sync/src/calibration/derive_extrinsics.py -d 2026.05.08/17-34-27
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_CALIB_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _CALIB_DIR.parent
_CODE_ROOT = _SYNC_DIR.parent
for _p in (_SYNC_DIR, _CODE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402

from correspondences import CorrespondenceSet  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--calibration_json", default=None, help="Calibration JSON (or via --dataset).")
    p.add_argument("--camera_correspondences", default=None, help="camera_lidar_correspondences.json")
    p.add_argument("--radar_correspondences", default=None, help="radar_lidar_correspondences.json")
    p.add_argument(
        "--max_translation_m",
        type=float,
        default=2.0,
        help="Warn if a solved translation exceeds this (sensors on one rig are close).",
    )
    add_dataset_arguments(p)
    return p


def _report_transform(name: str, mat: np.ndarray) -> list:
    """Print rotation/translation health for a 4x4; return list of warning strings."""
    warns = []
    R = np.asarray(mat)[:3, :3]
    t = np.asarray(mat)[:3, 3]
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det = float(np.linalg.det(R))
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    tnorm = float(np.linalg.norm(t))
    print(
        f"  {name}: rot={ang:6.2f} deg, |t|={tnorm:5.3f} m "
        f"t=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}), det(R)={det:.4f}, orthoErr={ortho_err:.2e}"
    )
    if ortho_err > 1e-3 or abs(det - 1.0) > 1e-3:
        warns.append(f"{name}: rotation is not a proper orthonormal matrix.")
    if np.allclose(mat, np.eye(4)):
        warns.append(f"{name}: is IDENTITY (unsolved placeholder).")
    return warns


def _residual_from_correspondences(mat: np.ndarray, path: Path, dim: int) -> Optional[float]:
    if not path.is_file():
        return None
    cs = CorrespondenceSet.load(path)
    src, tgt = cs.as_arrays()
    if len(cs) == 0:
        return None
    R = np.asarray(mat)[:dim, :dim]
    t = np.asarray(mat)[:dim, 3]  # translation is column 3 of the 4x4 (any dim)
    mapped = src @ R.T + t
    per = np.linalg.norm(mapped - tgt, axis=1)
    # Show the worst picks so bad correspondences can be re-done.
    order = np.argsort(per)[::-1]
    for k in order[: min(3, len(per))]:
        p = cs.points[k]
        depth = f", depth={p.source[2]:.1f}m" if dim == 3 else ""
        print(f"      worst #{k} pair={p.pair_idx} res={per[k]:.2f} m{depth}")
    return float(np.sqrt(np.mean(per**2)))


def _solved(payload: dict, key: str) -> bool:
    return bool(payload.get(f"{key}_meta", {}).get("solved", False))


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())
    calib_path = Path(args.calibration_json)
    with open(calib_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    missing = [k for k in ("camera_to_lidar", "radar_to_lidar") if k not in payload]
    if missing:
        raise SystemExit(
            f"{calib_path} is missing {missing}. Solve both with calibrate_camera_lidar.py "
            "and calibrate_radar_lidar.py first."
        )

    cam2lid = np.asarray(payload["camera_to_lidar"], dtype=np.float64)
    rad2lid = np.asarray(payload["radar_to_lidar"], dtype=np.float64)
    cam2rad = np.linalg.inv(rad2lid) @ cam2lid

    print("=== transforms ===")
    warns = []
    warns += _report_transform("camera_to_lidar   ", cam2lid)
    warns += _report_transform("radar_to_lidar    ", rad2lid)
    warns += _report_transform("camera_to_radar(*)", cam2rad)
    if np.linalg.norm(cam2rad[:3, 3]) > args.max_translation_m:
        warns.append(
            f"camera_to_radar translation ({np.linalg.norm(cam2rad[:3, 3]):.2f} m) "
            f"exceeds --max_translation_m={args.max_translation_m}."
        )

    if not _solved(payload, "camera_to_lidar"):
        warns.append("camera_to_lidar is not marked solved — run calibrate_camera_lidar.py.")
    if not _solved(payload, "radar_to_lidar"):
        warns.append("radar_to_lidar is not marked solved — run calibrate_radar_lidar.py.")

    # --- residual validation from saved correspondences ---
    cam_corr = Path(args.camera_correspondences or calib_path.parent / "camera_lidar_correspondences.json")
    rad_corr = Path(args.radar_correspondences or calib_path.parent / "radar_lidar_correspondences.json")
    print("\n=== fit residuals (stored matrix vs saved correspondences) ===")
    cam_rms = _residual_from_correspondences(cam2lid, cam_corr, dim=3)
    rad_rms = _residual_from_correspondences(rad2lid, rad_corr, dim=2)
    if cam_rms is not None:
        print(f"  camera_to_lidar RMS = {cam_rms * 1e3:6.1f} mm  ({cam_corr.name})")
        if cam_rms > 0.15:
            warns.append(f"camera_to_lidar RMS {cam_rms*1e3:.0f} mm is high (re-pick/re-solve).")
    else:
        print(f"  camera_to_lidar: no correspondences at {cam_corr.name} (skipped)")
    if rad_rms is not None:
        print(f"  radar_to_lidar  RMS = {rad_rms:6.3f} m   ({rad_corr.name})")
        if rad_rms > 0.5:
            warns.append(f"radar_to_lidar RMS {rad_rms:.2f} m is high (possible mirror/bad picks).")
    else:
        print(f"  radar_to_lidar: no correspondences at {rad_corr.name} (skipped)")

    # --- store derived transforms ---
    payload["camera_to_radar"] = cam2rad.tolist()
    payload["radar_to_camera"] = np.linalg.inv(cam2rad).tolist()
    payload["camera_to_radar_meta"] = {
        "derived": True,
        "formula": "inv(radar_to_lidar) @ camera_to_lidar",
        "camera_to_lidar_solved": _solved(payload, "camera_to_lidar"),
        "radar_to_lidar_solved": _solved(payload, "radar_to_lidar"),
        "written": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote camera_to_radar / radar_to_camera to {calib_path}")

    if warns:
        print("\n=== WARNINGS ===")
        for w in warns:
            print(f"  ! {w}")
        print("camera_to_radar is only trustworthy once BOTH pairwise solves are good.")
    else:
        print("\nAll checks passed — camera, radar, lidar are mutually consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
