"""Validate calibration visually by reprojecting lidar into the camera image.

RMS alone can hide a subtly-wrong transform; the real test is a picture. For each
pair this projects the lidar point cloud into the camera image via the solved
``camera_to_lidar`` extrinsic and overlays it, colored by range:

    p_cam = inv(camera_to_lidar) @ p_lidar ;  u = fx*x/z + cx, v = fy*y/z + cy

If the calibration is good, lidar points land on the matching scene structure
(a wall's returns on the wall, a pole's on the pole). If it's off, they're
visibly shifted. Optionally (`--radar`) also overlays radar strong-returns
transformed by ``radar_to_lidar`` onto the lidar BEV.

    python code/sync/src/calibration/validate_calibration.py \
        -d 2026.05.08/17-34-27 --pairs 300 800 1500

PNGs are written to ``--out_dir`` (default: res/<capture>/validation/).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")  # headless PNG output

_CALIB_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _CALIB_DIR.parent
_CODE_ROOT = _SYNC_DIR.parent
for _p in (_SYNC_DIR, _CODE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--calibration_json", default=None)
    p.add_argument("--camera_bag", default=None)
    p.add_argument("--camera_sync_csv", default=None)
    p.add_argument("--lidar_pcap", default=None)
    p.add_argument("--lidar_metadata", default=None)
    p.add_argument("--pairs", type=int, nargs="+", default=None, help="Sync pair indices.")
    p.add_argument("--out_dir", default=None, help="Where to save overlay PNGs.")
    p.add_argument("--max_range_m", type=float, default=20.0, help="Only project lidar within this range.")
    p.add_argument("--point_size", type=float, default=4.0)
    p.add_argument(
        "--radar",
        action="store_true",
        help="Also validate radar->lidar (radar BEV warped into the lidar frame + lidar).",
    )
    p.add_argument(
        "--radar_only",
        action="store_true",
        help="Skip the camera/depth overlay (and its bag scan); do the radar check only.",
    )
    p.add_argument("--radar_h5", default=None)
    p.add_argument("--cfg_file", default=None)
    p.add_argument("--sync_csv", default=None)
    p.add_argument("--sync_summary", default=None)
    p.add_argument("--radar_processing", choices=["legacy", "ti"], default="ti",
                   help="Radar range-azimuth chain (ti = clutter removal + windows).")
    p.add_argument("--radar_doppler_mode", choices=["all", "moving", "static"], default=None,
                   help="Which Doppler bins feed the BEV (default: moving — the canonical "
                   "calibration chain, matching the pick GUI). 'all' shows static structure too.")
    p.add_argument("--radar_aoa", choices=["capon", "dbf_nci", "dbf_coh"], default=None,
                   help="Angle-of-arrival estimator (default: dbf_nci for moving-target).")
    p.add_argument("--radar_bev_res", type=float, default=0.1,
                   help="Radar BEV Cartesian grid resolution (m); finer = smoother.")
    add_dataset_arguments(p)
    return p


def _load_pair_rows(csv_path: Path) -> Dict[int, dict]:
    import csv as _csv

    rows: Dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            rows[int(row["pair_idx"])] = row
    return rows


def _intrinsics(payload: dict, key: str):
    d = payload.get(key)
    if not d:
        return None
    return float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]), int(d.get("width", 0)), int(d.get("height", 0))


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())
    if not args.pairs:
        raise SystemExit("Give --pairs (sync pair indices) to validate.")

    with open(args.calibration_json, "r", encoding="utf-8") as f:
        calib = json.load(f)
    from extrinsic_io import get_extrinsic  # prefer an ICP-refined _auto over the seed
    try:
        cam2lid, cam_src = get_extrinsic(calib, "camera_to_lidar")
    except KeyError:
        raise SystemExit("No camera_to_lidar in calibration JSON — solve it first.")
    print(f"Using extrinsic: {cam_src}")
    lid2cam = np.linalg.inv(cam2lid)
    if not calib.get(f"{cam_src}_meta", {}).get("solved", False):
        print(f"WARNING: {cam_src} is not marked solved — overlay will likely be way off.")
    tnorm = float(np.linalg.norm(cam2lid[:3, 3]))
    if tnorm > 1.0:
        print(f"NOTE: camera_to_lidar |t|={tnorm:.2f} m is large for a one-rig camera+lidar "
              "(expect < ~0.5 m). If overlays are empty/off, the calibration is likely wrong.")

    # camera_to_lidar is solved in the DEPTH optical frame, so project with DEPTH
    # intrinsics onto the DEPTH image (color intrinsics/image would mis-scale
    # vertically + add a baseline offset -> beams dropped/misplaced).
    intr = _intrinsics(calib, "intrinsics")
    if intr is None:
        raise SystemExit("No depth intrinsics in calibration JSON.")
    fx, fy, cx, cy, _w, _h = intr
    depth_scale = float(calib.get("depth_scale", 0.001))

    import matplotlib.pyplot as plt

    from lib.camera_io import (  # noqa: E402
        DEFAULT_DEPTH_TOPIC,
        iter_topic_messages,
        parse_ros1_image,
    )
    from lib.bev_render import LidarScanReader  # noqa: E402

    rows = _load_pair_rows(Path(args.camera_sync_csv))
    plan = []
    for pid in sorted(args.pairs):
        if pid not in rows:
            print(f"  pair {pid} not in CSV; skipping.")
            continue
        l_idx = int(rows[pid].get("lidar_idx", -1))
        d_idx = int(rows[pid].get("camera_depth_idx", -1))
        if l_idx < 0 or d_idx < 0:
            print(f"  pair {pid}: no lidar/depth match; skipping.")
            continue
        plan.append((pid, l_idx, d_idx))
    if not plan:
        raise SystemExit("No usable pairs.")

    out_dir = Path(args.out_dir or (Path(args.calibration_json).parent / "validation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    lidar = LidarScanReader(args.lidar_pcap, args.lidar_metadata)

    depth_map: Dict[int, np.ndarray] = {}
    if not args.radar_only:
        # One sequential bag pass for all needed depth frames.
        needed = sorted({d for _, _, d in plan})
        print(f"Scanning bag for {len(needed)} depth frame(s) (up to {needed[-1]}) + PCAP...", flush=True)
        mx, ns = needed[-1], set(needed)
        for i, (_ts, raw) in enumerate(iter_topic_messages(args.camera_bag, DEFAULT_DEPTH_TOPIC)):
            if i > mx:
                break
            if i in ns:
                depth_map[i] = np.array(parse_ros1_image(raw).data, copy=True)

    for pid, l_idx, d_idx in ([] if args.radar_only else plan):
        dimg = depth_map.get(d_idx)
        if dimg is None:
            print(f"  pair {pid}: no depth frame; skipping.")
            continue
        h, w = dimg.shape
        depth_m = dimg.astype(np.float32) * depth_scale
        disp = np.where((depth_m > 0.3) & (depth_m < args.max_range_m), depth_m, np.nan)

        pts = lidar.points_from_scan_idx(l_idx)
        if pts.size == 0:
            continue
        pts = pts[np.linalg.norm(pts[:, :2], axis=1) <= args.max_range_m]
        n_total = len(pts)
        pc = pts @ lid2cam[:3, :3].T + lid2cam[:3, 3]  # lidar -> depth optical
        z = pc[:, 2]
        front = z > 0.1
        pc, z = pc[front], z[front]
        n_front = len(z)
        u = fx * pc[:, 0] / z + cx
        v = fy * pc[:, 1] / z + cy
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[inb], v[inb], z[inb]
        print(f"  pair {pid}: {n_total} lidar pts (<{args.max_range_m:.0f}m), "
              f"{n_front} in front, {len(u)} on image ({w}x{h} depth)")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(disp, cmap="gray")  # D435 depth as grayscale background
        sc = ax.scatter(u, v, s=args.point_size, c=z, cmap="turbo", alpha=0.8)
        fig.colorbar(sc, ax=ax, label="lidar range (m)", fraction=0.03)
        ax.set_title(f"pair {pid}: lidar (colored) vs D435 depth (gray) — good = they overlap  [{len(u)} pts]")
        ax.axis("off")
        out = out_dir / f"reproj_pair{pid}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.name}")

    if args.radar or args.radar_only:  # --radar_only implies the radar overlay
        _overlay_radar(args, calib, plan, lidar, out_dir)

    print(f"\nDone. Overlays in {out_dir}\n"
          "Good calibration = lidar points sit on the matching scene structure.")
    return 0


def _overlay_radar(args, calib, plan, lidar, out_dir) -> None:
    """Validate radar->lidar: warp the radar BEV into the lidar frame (via radar_to_lidar)
    and overlay lidar points. Good calibration = radar reflectors (bright) sit on the
    lidar structure (white points). Mirrors the camera check but for radar."""
    from extrinsic_io import get_extrinsic, has_extrinsic

    if not has_extrinsic(calib, "radar_to_lidar"):
        print("No radar_to_lidar in calibration; skipping radar overlay.")
        return
    import matplotlib.pyplot as plt

    from make_video import RadarLoader, build_argparser as _video_argparser  # noqa: E402

    # Prefer an auto-solved refinement (radar_to_lidar_auto) over the hand-picked one.
    R, _r_src = get_extrinsic(calib, "radar_to_lidar")
    print(f"Radar overlay using '{_r_src}'.")
    # Build a fully-populated radar args namespace (TI processing, fine grid) from the
    # make_video defaults, overridden with the paths we already resolved.
    from lib.bev_render import (
        CALIB_RADAR_BEV_RES, apply_calib_radar_defaults, radar_bev_vlimits,
    )

    rargs = _video_argparser().parse_args([])
    for k in ("radar_h5", "cfg_file", "sync_csv", "sync_summary", "lidar_pcap", "lidar_metadata"):
        v = getattr(args, k, None)
        if v:
            setattr(rargs, k, v)
    rargs.radar_processing = getattr(args, "radar_processing", None)
    for k in ("radar_doppler_mode", "radar_aoa"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(rargs, k, v)
    apply_calib_radar_defaults(rargs)  # canonical moving-target chain (shared with GUI)
    print(f"Radar BEV: processing={rargs.radar_processing}, doppler={rargs.radar_doppler_mode}, "
          f"aoa={rargs.radar_aoa}")
    radar = RadarLoader(rargs)
    if hasattr(radar, "display_bev_kw"):
        radar.display_bev_kw["res"] = float(getattr(args, "radar_bev_res", None) or CALIB_RADAR_BEV_RES)

    rng = float(args.max_range_m)
    for pid, l_idx, _d_idx in plan:
        bev, _ = radar.load_radar_panel(pid)
        n_fwd, n_lat = bev.shape
        la = np.linspace(radar.lateral_range[0], radar.lateral_range[1], n_lat)
        fa = np.linspace(radar.forward_range[0], radar.forward_range[1], n_fwd)
        RL, RF = np.meshgrid(la, fa)                       # radar-frame lateral, forward
        LF = R[0, 0] * RF + R[0, 1] * RL + R[0, 3]         # -> lidar forward
        LL = R[1, 0] * RF + R[1, 1] * RL + R[1, 3]         # -> lidar lateral
        pts = lidar.points_from_scan_idx(l_idx, indexed=True)

        vmin, vmax = radar_bev_vlimits(bev)  # shared 60-99.5 pct contrast
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pcolormesh(LL, LF, bev, cmap="viridis", shading="gouraud",
                      vmin=vmin, vmax=vmax)
        if pts.size:
            m = np.linalg.norm(pts[:, :2], axis=1) <= rng
            ax.scatter(pts[m, 1], pts[m, 0], s=2, c="red", alpha=0.6, linewidths=0,
                       label="lidar points")
        ax.set_xlim(rng, -rng)
        ax.set_ylim(0, rng)
        ax.set_aspect("equal")
        ax.set_xlabel("Lateral Y (m)")
        ax.set_ylabel("Forward X (m)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        _src = "radar_to_lidar_auto" if calib.get("radar_to_lidar_auto") is not None else "radar_to_lidar"
        ax.set_title(f"pair {pid}: radar BEV (viridis) vs lidar (red), via {_src}\n"
                     "good = radar reflectors sit on lidar structure")
        out = out_dir / f"radar_overlay_pair{pid}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    raise SystemExit(main())
