"""Refine the camera->lidar extrinsic by CLOUD REGISTRATION (ICP) -- no point picking.

Manual correspondence picking is limited by how precisely a human can click the SAME
physical point in the depth image and the lidar (even in the range-image view, ~cm-level
scatter over a handful of picks leaves RMS in the 100s of mm). This tool removes picking
entirely: it back-projects the D435 depth to a dense 3D cloud, drops it into the lidar
frame using the EXISTING hand-picked extrinsic as a seed, and runs point-to-plane ICP
against the Ouster cloud over one or more near pairs. ICP uses tens of thousands of points
instead of ~14 clicks, so it averages out pick noise and snaps to the true surface fit.

Requirements / caveats
----------------------
* Needs a reasonable SEED (the hand-picked ``camera_to_lidar``): ICP only polishes, it
  can't find the basin from scratch. Your range-view fit (|t|~30cm, pitch/yaw~1.6 deg)
  is a good seed.
* Needs OVERLAP + VERTICAL STRUCTURE. Use near pairs (< ~5 m, D435 depth limit) that
  contain a wall/doorway/vehicle -- a bare ground plane lets ICP slide (planar degeneracy).
* The result is bounded by how well the D435 depth and Ouster actually agree; expect a
  few-cm floor, not zero.

By default this is a DRY RUN: it prints the before/after health numbers (inlier RMSE,
|t|, roll/pitch/yaw) and writes nothing. Pass ``--write`` to store the refined transform
as ``camera_to_lidar_auto`` (the hand-picked ``camera_to_lidar`` seed is preserved; and
``get_extrinsic`` prefers the ``_auto`` refinement at read time, matching the radar path).

Run::

    python code/sync/src/calibration/register_camera_lidar.py -d 2026.05.08/17-34-27 \
        --pairs 5000 12720 --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_CALIB_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _CALIB_DIR.parent
_CODE_ROOT = _SYNC_DIR.parent
for _p in (_SYNC_DIR, _CODE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.camera_io import DEFAULT_DEPTH_TOPIC  # noqa: E402
from lib.dataset_config import (  # noqa: E402
    add_dataset_arguments,
    apply_dataset_config,
)
from camera_projection.geometry import load_calibration  # noqa: E402
from extrinsic_io import get_extrinsic, write_extrinsic  # noqa: E402
from rigid_solve import level_constrained_fit, umeyama_3d  # noqa: E402
from correspondences import CorrespondenceSet  # noqa: E402

# Reuse the exact frame/pair loaders the picking tool uses, so the ICP source cloud
# is built identically to the manual pipeline (same intrinsics, same depth backproject).
from calibrate_camera_lidar import (  # noqa: E402
    _batch_read_frames,
    _load_pair_rows,
    _roll_pitch_yaw_vs_ideal,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arguments(p)
    p.add_argument("--pairs", type=int, nargs="+", required=True,
                   help="Sync pair indices with near shared structure (wall/vehicle < ~5 m).")
    p.add_argument("--lidar_pcap", default=None)
    p.add_argument("--lidar_metadata", default=None)
    p.add_argument("--camera_bag", default=None)
    p.add_argument("--camera_sync_csv", default=None)
    p.add_argument("--calibration_json", default=None)
    p.add_argument("--depth_scale", type=float, default=0.001, help="Metres per depth unit (D435: 0.001).")
    p.add_argument("--seed", default="auto", choices=["auto", "correspondences", "calibration"],
                   help="ICP seed source. 'auto' (default): solve the correspondences file if present "
                        "(the freshest hand-picked fit), else use calibration.json. 'correspondences': "
                        "force solving the picks. 'calibration': use the stored camera_to_lidar.")
    p.add_argument("--correspondences", default=None,
                   help="Path to camera_lidar.json correspondences (default: <calib dir>/correspondences/).")
    p.add_argument("--free_solve", action="store_true",
                   help="When seeding from correspondences, use a free 3-DoF fit (default locks roll).")
    p.add_argument("--min_depth_m", type=float, default=0.3)
    p.add_argument("--max_depth_m", type=float, default=5.0,
                   help="Crop depth cloud past this (D435 depth is unreliable beyond ~5 m).")
    p.add_argument("--overlap_margin_m", type=float, default=0.5,
                   help="Expand the camera-cloud bbox by this before cropping the lidar to it.")
    p.add_argument("--voxel_m", type=float, default=0.05, help="Voxel-downsample both clouds to this size.")
    p.add_argument("--drop_ground", action="store_true",
                   help="Remove near-ground points from both clouds before ICP (use if a flat "
                        "floor dominates and ICP slides; keep OFF when the ground helps pin pitch/z).")
    p.add_argument("--ground_z_m", type=float, default=0.15,
                   help="With --drop_ground: drop points within this height of the lowest returns.")
    p.add_argument("--icp_dists_m", type=float, nargs="+", default=[0.6, 0.3, 0.15],
                   help="Coarse->fine point-to-plane ICP correspondence distances (m).")
    p.add_argument("--max_rot_deg", type=float, default=15.0,
                   help="Refuse to write if ICP moved rotation more than this from the seed "
                        "(a large jump means ICP diverged on poor overlap).")
    p.add_argument("--overlay", action="store_true",
                   help="Render per-pair seed-vs-refined overlays (lidar projected into the depth "
                        "image, colored by range) for visual validation before writing.")
    p.add_argument("--overlay_dir", default=None,
                   help="Where to write overlay PNGs (default: <calib dir>/validation/icp_check/).")
    p.add_argument("--overlay_max_range_m", type=float, default=10.0,
                   help="Only project lidar within this range in the overlay.")
    p.add_argument("--write", action="store_true",
                   help="Write the refined transform as camera_to_lidar_auto (default: dry run).")
    p.add_argument("--force", action="store_true",
                   help="Write even if the health guards (|t| plausible, RMSE improved, rotation "
                        "within --max_rot_deg) fail. Use only if you have inspected the result.")
    return p


def _depth_to_cloud(depth_img: np.ndarray, calib, min_z: float, max_z: float) -> np.ndarray:
    """Back-project a depth image to a camera-optical-frame cloud (N,3), same as picking."""
    z = depth_img.astype(np.float64) * calib.depth_scale
    v, u = np.mgrid[0:depth_img.shape[0], 0:depth_img.shape[1]]
    valid = (z >= min_z) & (z <= max_z) & np.isfinite(z)
    z = z[valid]
    x = (u[valid] - calib.cx) / calib.fx * z
    y = (v[valid] - calib.cy) / calib.fy * z
    return np.column_stack([x, y, z])


def _transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def _project_lidar(ldr: np.ndarray, E: np.ndarray, calib, w: int, h: int):
    """Project lidar points into the depth camera via extrinsic E (cam_optical->lidar).

    Returns (u, v, range_m) for points in front of and inside the image. Mirrors
    validate_calibration: p_cam = inv(E) @ p_lidar, then pinhole projection.
    """
    lid2cam = np.linalg.inv(E)
    pc = ldr @ lid2cam[:3, :3].T + lid2cam[:3, 3]
    z = pc[:, 2]
    front = z > 0.1
    pc, z, rng = pc[front], z[front], np.linalg.norm(ldr[front], axis=1)
    u = calib.fx * pc[:, 0] / z + calib.cx
    v = calib.fy * pc[:, 1] / z + calib.cy
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[inb], v[inb], rng[inb]


def _render_overlays(overlay_data, seed, new, calib, out_dir, max_range_m):
    """Side-by-side (seed | refined) lidar-into-depth overlays, one PNG per pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for pid, depth_img, ldr in overlay_data:
        h, w = depth_img.shape
        depth_m = depth_img.astype(np.float32) * calib.depth_scale
        disp = np.where((depth_m > 0.3) & (depth_m < max_range_m), depth_m, np.nan)
        ldr_c = ldr[np.linalg.norm(ldr[:, :2], axis=1) <= max_range_m]
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        for ax, E, name in ((axs[0], seed, "SEED (picks)"), (axs[1], new, "ICP-refined")):
            u, v, rng = _project_lidar(ldr_c, E, calib, w, h)
            ax.imshow(disp, cmap="gray")
            sc = ax.scatter(u, v, s=4, c=rng, cmap="turbo", alpha=0.8, vmin=0, vmax=max_range_m)
            ax.set_title(f"pair {pid}: {name} — lidar(color) on D435 depth(gray); aligned=good [{len(u)} pts]")
            ax.axis("off")
        fig.colorbar(sc, ax=axs, label="lidar range (m)", fraction=0.02)
        out = out_dir / f"icp_overlay_pair{pid}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
        print(f"  wrote {out}")
    return written


def _drop_ground(pts: np.ndarray, ground_z: float) -> np.ndarray:
    if pts.size == 0:
        return pts
    floor = np.percentile(pts[:, 2], 2.0)  # robust "lowest" in lidar frame (Z up)
    return pts[pts[:, 2] > floor + ground_z]


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())
    import open3d as o3d

    calib = load_calibration(args.calibration_json)

    # --- choose the ICP seed --------------------------------------------------
    # The interactive tool's on-quit solve is PRINT-ONLY, so the freshest hand-picked
    # fit lives in the correspondences file, not calibration.json. Seed from the picks
    # when we can (solving them here), so ICP starts in the right basin.
    corr_path = Path(
        args.correspondences
        or (Path(args.calibration_json).parent / "correspondences" / "camera_lidar.json")
    )
    seed = None
    seed_key = ""
    if args.seed in ("auto", "correspondences") and corr_path.is_file():
        cs = CorrespondenceSet.load(corr_path)
        if len(cs) >= 3:
            src_c, tgt_c = cs.as_arrays()
            fit = (umeyama_3d if args.free_solve else level_constrained_fit)(src_c, tgt_c)
            seed = fit.transform
            seed_key = f"correspondences ({len(cs)} pts, RMS={fit.rms*1e3:.0f}mm)"
        elif args.seed == "correspondences":
            print(f"Correspondence file has only {len(cs)} points (<3); cannot solve seed.")
            return 2
    if seed is None:
        if args.seed == "correspondences":
            print(f"No usable correspondences at {corr_path}.")
            return 2
        import json
        calib_dict = json.load(open(args.calibration_json, "r", encoding="utf-8"))
        seed, base_key = get_extrinsic(calib_dict, "camera_to_lidar")
        seed_key = f"calibration.json:{base_key}"
    print(f"ICP seed: {seed_key}  |t|={np.linalg.norm(seed[:3, 3]) * 100:.1f} cm")

    rows = _load_pair_rows(Path(args.camera_sync_csv))
    plan = []
    for pid in args.pairs:
        if pid not in rows:
            print(f"  pair {pid} not in sync CSV; skipping.")
            continue
        r = rows[pid]
        plan.append((pid, int(r["camera_depth_idx"]), int(r["lidar_idx"])))
    if not plan:
        print("No valid pairs.")
        return 2

    depth_map = _batch_read_frames(
        args.camera_bag, DEFAULT_DEPTH_TOPIC, [d for _, d, _ in plan], label="depth"
    )

    from lib.bev_render import LidarScanReader
    lidar = LidarScanReader(args.lidar_pcap, args.lidar_metadata)

    src_all: List[np.ndarray] = []  # camera cloud in lidar frame (seed-transformed)
    tgt_all: List[np.ndarray] = []  # lidar cloud, cropped to the overlap
    overlay_data = []               # (pid, depth_img, full lidar cloud) for validation render
    for pid, d_idx, l_idx in plan:
        depth_img = depth_map.get(d_idx)
        if depth_img is None:
            print(f"  pair {pid}: depth frame {d_idx} missing; skipping.")
            continue
        cam = _depth_to_cloud(depth_img, calib, args.min_depth_m, args.max_depth_m)
        cam_l = _transform(cam, seed)  # into lidar frame via seed
        if cam_l.shape[0] < 100:
            print(f"  pair {pid}: only {cam_l.shape[0]} depth points; skipping.")
            continue
        ldr = lidar.points_from_scan_idx(l_idx, indexed=True)
        if args.overlay:
            overlay_data.append((pid, depth_img, ldr.copy()))
        # Crop lidar to the camera cloud's bounding box (+ margin) = the overlap region,
        # so ICP never matches camera points to lidar returns behind/beside the camera FOV.
        lo = cam_l.min(axis=0) - args.overlap_margin_m
        hi = cam_l.max(axis=0) + args.overlap_margin_m
        m = np.all((ldr >= lo) & (ldr <= hi), axis=1)
        ldr = ldr[m]
        if args.drop_ground:
            cam_l = _drop_ground(cam_l, args.ground_z_m)
            ldr = _drop_ground(ldr, args.ground_z_m)
        print(f"  pair {pid}: camera {cam_l.shape[0]} pts, lidar-overlap {ldr.shape[0]} pts")
        src_all.append(cam_l)
        tgt_all.append(ldr)

    if not src_all:
        print("No usable pairs after loading.")
        return 2

    src = np.vstack(src_all)
    tgt = np.vstack(tgt_all)

    def to_pcd(a):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(a)
        return p.voxel_down_sample(args.voxel_m)

    src_pcd = to_pcd(src)
    tgt_pcd = to_pcd(tgt)
    tgt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_m * 3, max_nn=30))
    print(f"Downsampled: source {len(src_pcd.points)} pts, target {len(tgt_pcd.points)} pts "
          f"(voxel {args.voxel_m * 100:.0f} cm)")

    # Baseline (seed) quality at the finest threshold: identity delta, since src is already
    # seed-transformed into the lidar frame.
    fine = float(args.icp_dists_m[-1])
    seed_eval = o3d.pipelines.registration.evaluate_registration(src_pcd, tgt_pcd, fine, np.eye(4))

    # Coarse->fine point-to-plane ICP. delta refines src (already in lidar frame) onto tgt.
    delta = np.eye(4)
    for dist in args.icp_dists_m:
        reg = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd, float(dist), delta,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        delta = reg.transformation
    ref_eval = o3d.pipelines.registration.evaluate_registration(src_pcd, tgt_pcd, fine, delta)

    new = delta @ seed  # refined camera_optical -> lidar

    # --- report ---
    def rpy(T):
        return _roll_pitch_yaw_vs_ideal(T[:3, :3])

    sr, sp, sy = rpy(seed)
    nr, np_, ny = rpy(new)
    d_rot_deg = np.degrees(
        np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))
    )
    t_new = new[:3, 3]
    t_norm = float(np.linalg.norm(t_new))
    print("\n=== camera->lidar ICP refinement ===")
    print(f"                 inlier_RMSE (@{fine*100:.0f}cm)   overlap(fitness)   |t|      roll/pitch/yaw")
    print(f"  seed  ({seed_key:20s}) {seed_eval.inlier_rmse*1e3:7.0f} mm      "
          f"{seed_eval.fitness:5.2f}          {np.linalg.norm(seed[:3,3])*100:5.1f}cm  "
          f"{sr:+.1f}/{sp:+.1f}/{sy:+.1f}")
    print(f"  ICP-refined                    {ref_eval.inlier_rmse*1e3:7.0f} mm      "
          f"{ref_eval.fitness:5.2f}          {t_norm*100:5.1f}cm  {nr:+.1f}/{np_:+.1f}/{ny:+.1f}")
    print(f"  ICP moved: rotation {d_rot_deg:.1f} deg, translation "
          f"{np.linalg.norm(delta[:3,3])*100:.1f} cm")
    print("Refined transform (camera_optical -> lidar_sensor):")
    print(np.array2string(new, precision=5, suppress_small=True))
    print(f"Baseline |t|={t_norm*100:.1f} cm  (t=[{t_new[0]:+.2f} {t_new[1]:+.2f} {t_new[2]:+.2f}] m)")

    # --- health guards ---
    improved = ref_eval.inlier_rmse < seed_eval.inlier_rmse
    plausible = 0.1 <= t_norm <= 0.8
    small_move = d_rot_deg <= args.max_rot_deg
    reasons = []
    if not improved:
        reasons.append(f"RMSE did not improve ({ref_eval.inlier_rmse*1e3:.0f} >= {seed_eval.inlier_rmse*1e3:.0f} mm)")
    if not plausible:
        reasons.append(f"|t|={t_norm*100:.0f} cm is outside the plausible 10-80 cm rig baseline")
    if not small_move:
        reasons.append(f"ICP moved rotation {d_rot_deg:.1f} deg > --max_rot_deg {args.max_rot_deg:.0f} (likely diverged)")
    ok = improved and plausible and small_move
    if args.overlay and overlay_data:
        odir = Path(args.overlay_dir or (Path(args.calibration_json).parent / "validation" / "icp_check"))
        print(f"\nRendering seed-vs-refined overlays -> {odir}")
        _render_overlays(overlay_data, seed, new, calib, odir, args.overlay_max_range_m)

    if ok:
        print("\nHealth guards: PASS (RMSE improved, |t| plausible, rotation move within bound).")
    else:
        print("\nHealth guards: FAIL --")
        for r in reasons:
            print(f"  - {r}")

    if not args.write:
        print("\nDry run (no --write): nothing written. Re-run with --write to store as camera_to_lidar_auto.")
        return 0
    if not ok and not args.force:
        print("\nRefusing to write (guards failed). Inspect the result, pick near pairs with more "
              "vertical structure, or pass --force to override.")
        return 1

    write_extrinsic(
        args.calibration_json,
        key="camera_to_lidar_auto",
        matrix=new,
        n_points=int(len(src_pcd.points)),
        rms_m=float(ref_eval.inlier_rmse),
        max_error_m=float(fine),
        correspondences_path=None,
        note=("ICP point-to-plane refinement of the hand-picked camera_to_lidar seed "
              f"({seed_key}); pairs={list(args.pairs)}"),
        method="cloud registration (point-to-plane ICP) seeded from hand-picked extrinsic",
    )
    print("Wrote camera_to_lidar_auto (hand-picked camera_to_lidar seed preserved).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
