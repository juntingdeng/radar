"""Interactive camera->lidar extrinsic calibration via 3D<->3D correspondences.

Workflow
--------
1. Pick a few sync pairs that contain sharp, static structure visible to BOTH
   the D435 depth camera and the lidar (wall corners, pole bases, curbs).
2. For each pair the tool shows the colorized depth image (left) and the lidar
   BEV (right). Click the SAME physical point in each: left first, then right.
   - Left click  -> depth at that pixel is back-projected to a 3D camera point.
   - Right click -> snapped to the nearest real lidar return (full 3D XYZ).
3. Press ``s`` to solve. It runs a rigid Umeyama fit (target = R*source + t),
   prints the RMS residual, writes ``camera_to_lidar`` into the calibration JSON
   and saves the raw correspondences alongside.

Collect >= 4 correspondences spread in depth and lateral extent for a stable fit.

Run::

    python code/sync/calibration/calibrate_camera_lidar.py \
        -d 2026.05.10/18-05-08 \
        --pairs 300 1200 2600 4000

Headless solve from an existing correspondence file (no GUI)::

    python code/sync/calibration/calibrate_camera_lidar.py \
        -d 2026.05.10/18-05-08 --solve_only path/to/camera_lidar_correspondences.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_CALIB_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _CALIB_DIR.parent
_CODE_ROOT = _SYNC_DIR.parent
for _p in (_SYNC_DIR, _CODE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.camera_io import (  # noqa: E402
    DEFAULT_COLOR_TOPIC,
    DEFAULT_DEPTH_TOPIC,
    iter_topic_messages,
    parse_ros1_image,
)
from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402
from camera_projection.geometry import load_calibration  # noqa: E402

from correspondences import CorrespondenceSet  # noqa: E402
from extrinsic_io import (  # noqa: E402
    ensure_calibration_json,
    reuse_calibration,
    write_extrinsic,
)
from rigid_solve import fit_with_rejection, umeyama_3d  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Camera->lidar extrinsic calibration.")
    p.add_argument("--camera_sync_csv", default=None, help="camera_sync_pairs.csv")
    p.add_argument("--camera_bag", default=None, help="RealSense ROS bag")
    p.add_argument("--lidar_pcap", default=None, help="Ouster .pcap")
    p.add_argument("--lidar_metadata", default=None, help="Ouster metadata .json")
    p.add_argument("--calibration_json", default=None, help="Calibration JSON to update")
    p.add_argument("--depth_topic", default=DEFAULT_DEPTH_TOPIC)
    p.add_argument("--color_topic", default=DEFAULT_COLOR_TOPIC)
    p.add_argument(
        "--no_color",
        action="store_true",
        help="Skip the RGB context panel (avoids a second camera-bag scan).",
    )
    p.add_argument(
        "--pairs",
        type=int,
        nargs="+",
        default=None,
        help="Sync pair indices to calibrate from (choose frames with clear structure).",
    )
    p.add_argument(
        "--correspondences",
        default=None,
        help="Where to save/load picked correspondences (default: next to calibration_json).",
    )
    p.add_argument(
        "--solve_only",
        default=None,
        help="Skip picking; solve from this correspondence JSON and write the extrinsic.",
    )
    p.add_argument(
        "--reuse_from",
        default=None,
        help="Same rig, already calibrated elsewhere: copy the solved camera_to_lidar "
        "(+ intrinsics) from this reference calibration JSON and exit (no picking).",
    )
    p.add_argument(
        "--refresh_intrinsics",
        action="store_true",
        help="With --reuse_from, re-read intrinsics from THIS bag instead of copying them.",
    )
    p.add_argument("--overwrite", action="store_true", help="With --reuse_from, replace an existing output.")
    p.add_argument("--depth_scale", type=float, default=0.001, help="Metres per depth unit (D435: 0.001).")
    p.add_argument("--min_depth_m", type=float, default=0.3)
    p.add_argument(
        "--max_depth_m",
        type=float,
        default=6.0,
        help="Reject PICKS beyond this depth (m). D435 stereo depth is only reliable "
        "to ~6 m; far picks have garbage depth and wreck the fit. Raise only if needed.",
    )
    p.add_argument(
        "--depth_display_m",
        type=float,
        default=15.0,
        help="Upper depth (m) shown in the depth panel (display only; picking still "
        "uses --max_depth_m). Larger = more scene context but less near contrast.",
    )
    p.add_argument(
        "--pick_search_px",
        type=int,
        default=6,
        help="Snap a depth click to the nearest valid pixel within this radius "
        "(handles D435 depth holes on reflective/edge surfaces).",
    )
    p.add_argument(
        "--box",
        action="store_true",
        help="Box mode: DRAG a box around an object in each panel; the correspondence "
        "is the centroid of the enclosed points. More robust than exact-point clicks.",
    )
    p.add_argument(
        "--min_box_points",
        type=int,
        default=15,
        help="Minimum valid points required inside a box (box mode).",
    )
    p.add_argument(
        "--box_depth_pct",
        type=float,
        default=60.0,
        help="Box mode: keep the nearest this-percent of depths in a camera box "
        "(foreground bias, avoids background contamination).",
    )
    p.add_argument(
        "--reject_outliers_m",
        type=float,
        default=0.0,
        help="If >0, drop correspondences whose fit residual exceeds this (m) and "
        "refit — safety net for a few gross mis-picks (won't fix inconsistent data).",
    )
    p.add_argument(
        "--snap_radius_m",
        type=float,
        default=1.0,
        help="Max XY distance a lidar click may snap to a real return.",
    )
    p.add_argument(
        "--max_range_m",
        type=float,
        default=30.0,
        help="Upper bound for the lidar BEV view (m). Auto-fit stays within this.",
    )
    p.add_argument(
        "--no_auto_range",
        action="store_true",
        help="Disable per-frame auto-zoom of the lidar BEV (use fixed +/- max_range_m).",
    )
    p.add_argument(
        "--no_flip_lidar",
        action="store_true",
        help="Don't mirror the lidar BEV horizontally. By default it is flipped so its "
        "left/right matches the forward camera (display only; picks are unaffected).",
    )
    add_dataset_arguments(p)
    return p




def _solve_and_write(cs: CorrespondenceSet, args, corr_path: Path) -> int:
    src, tgt = cs.as_arrays()
    if len(cs) < 3:
        print(f"Need >= 3 correspondences to solve; have {len(cs)}. Nothing written.")
        return 1

    # Forward-consistency check: camera depth z should ~= lidar forward X (the two
    # sensors are physically close). A large ABSOLUTE gap means a near camera object
    # was paired with a far lidar object (D435 can't see far) — a bad correspondence.
    fwd_gap = np.abs(tgt[:, 0] - src[:, 2])
    bad = np.where(fwd_gap > 2.5)[0]
    if bad.size:
        print("WARNING: camera depth vs lidar forward mismatch (>2.5 m) on:")
        for i in bad:
            print(f"    #{i} pair={cs.points[i].pair_idx} cam_z={src[i,2]:.1f} m "
                  f"lidar_X={tgt[i,0]:.1f} m  -> different objects; re-box a NEAR one.")

    if args.reject_outliers_m > 0:
        fit, keep = fit_with_rejection(
            umeyama_3d, src, tgt, max_residual=args.reject_outliers_m, min_points=4
        )
        dropped = [int(i) for i in np.where(~keep)[0]]
        if dropped:
            print(f"Robust fit dropped {len(dropped)} correspondence(s) "
                  f"> {args.reject_outliers_m} m: {dropped}")
    else:
        fit = umeyama_3d(src, tgt)
    print("\n=== camera->lidar rigid fit ===")
    print(fit.summary())
    print("Transform (camera_optical -> lidar_sensor):")
    print(np.array2string(fit.transform, precision=5, suppress_small=True))
    worst = int(np.argmax(fit.residuals))
    print(f"Worst kept correspondence: err={fit.residuals[worst] * 1e3:.1f} mm")
    if fit.rms > 0.15:
        print(
            f"WARNING: RMS {fit.rms * 1e3:.0f} mm is high. Re-check picks or add points "
            "spread across depth/lateral extent."
        )
    cs.save(corr_path)
    write_extrinsic(
        args.calibration_json,
        key="camera_to_lidar",
        matrix=fit.transform,
        n_points=fit.n_points,
        rms_m=fit.rms,
        max_error_m=fit.max_error,
        correspondences_path=corr_path,
        note="camera depth optical frame -> Ouster sensor frame (X fwd, Y left, Z up)",
    )
    return 0


def _batch_read_frames(bag_path, topic, indices, label="frame") -> Dict[int, np.ndarray]:
    """Read all requested frames on one topic in a single sequential bag pass."""
    needed = sorted({int(i) for i in indices if int(i) >= 0})
    if not needed:
        return {}
    needed_set = set(needed)
    max_idx = needed[-1]
    print(
        f"Scanning camera bag for {len(needed)} {label} frame(s) "
        f"(up to frame {max_idx}; several minutes over USB)...",
        flush=True,
    )
    out: Dict[int, np.ndarray] = {}
    for i, (_ts, raw) in enumerate(iter_topic_messages(bag_path, topic)):
        if i > max_idx:
            break
        if i in needed_set:
            out[i] = np.array(parse_ros1_image(raw).data, copy=True)
            print(f"  {label} frame {i} loaded ({len(out)}/{len(needed)})", flush=True)
    return out


def _load_pair_rows(csv_path: Path) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["pair_idx"])] = row
    return rows


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())

    # --- reuse path: same rig already calibrated elsewhere (no picking) ---
    if args.reuse_from:
        reuse_calibration(
            args.calibration_json,
            args.reuse_from,
            camera_bag=args.camera_bag,
            refresh_intrinsics=args.refresh_intrinsics,
            depth_scale=args.depth_scale,
            keys=("camera_to_lidar",),
            overwrite=args.overwrite,
        )
        return 0

    corr_path = Path(
        args.correspondences
        or (Path(args.calibration_json).parent / "camera_lidar_correspondences.json")
    )

    cs = CorrespondenceSet(
        kind="camera_lidar",
        dim=3,
        source_frame="camera_depth_optical",
        target_frame="lidar_sensor",
        dataset=getattr(args, "dataset", "") or "",
    )

    # --- headless solve path ---
    if args.solve_only:
        cs = CorrespondenceSet.load(args.solve_only)
        return _solve_and_write(cs, args, Path(args.solve_only))

    # --- interactive path ---
    if not args.pairs:
        print("No --pairs given. Choose sync pair indices with clear shared structure.")
        return 2

    from interactive import bev_view_limits, ensure_interactive_backend

    ensure_interactive_backend()  # must run before pyplot is imported
    import matplotlib.pyplot as plt

    from lib.bev_render import LidarScanReader  # heavy import; only for picking

    # Create the calibration JSON (intrinsics from the bag + identity extrinsic)
    # if it does not exist yet, so a fresh collection needs no separate step.
    ensure_calibration_json(
        args.calibration_json, camera_bag=args.camera_bag, depth_scale=args.depth_scale
    )
    calib = load_calibration(args.calibration_json)
    rows = _load_pair_rows(Path(args.camera_sync_csv))

    # Resolve depth/color/lidar indices for every requested pair up front.
    plan: List[tuple] = []  # (pid, depth_idx, color_idx, lidar_idx)
    for pid in sorted(args.pairs):
        if pid not in rows:
            print(f"  pair {pid} not in CSV; skipping.")
            continue
        row = rows[pid]
        d_idx = int(row.get("camera_depth_idx", -1))
        c_idx = int(row.get("camera_color_idx", -1))
        l_idx = int(row.get("lidar_idx", -1))
        if d_idx < 0 or l_idx < 0:
            print(f"  pair {pid}: no camera/lidar match; skipping.")
            continue
        plan.append((pid, d_idx, c_idx, l_idx))

    if not plan:
        print("No usable pairs loaded.")
        return 2

    print(
        f"\nLoading {len(plan)} pair(s): {[p for p, _, _, _ in plan]}\n"
        "This reads the camera bag and the ~12 GB lidar PCAP over USB and can take "
        "several minutes with no output during each scan — it is NOT stuck.\n",
        flush=True,
    )

    # 1) Depth (+ optional RGB) frames: one sequential bag pass per topic.
    depth_map = _batch_read_frames(
        args.camera_bag, args.depth_topic, [d for _, d, _, _ in plan], label="depth"
    )
    color_map: Dict[int, np.ndarray] = {}
    if not args.no_color:
        color_map = _batch_read_frames(
            args.camera_bag, args.color_topic, [c for _, _, c, _ in plan], label="color"
        )

    # 2) Lidar XYZ: open the PCAP once, read scans in ascending order.
    print(
        "Opening lidar PCAP (indexing ~12 GB; a few minutes over USB)...", flush=True
    )
    lidar = LidarScanReader(args.lidar_pcap, args.lidar_metadata)

    frames: List[dict] = []
    for pid, d_idx, c_idx, l_idx in plan:
        depth_img = depth_map.get(d_idx)
        if depth_img is None:
            print(f"  pair {pid}: depth frame {d_idx} not found in bag; skipping.")
            continue
        print(f"  reading lidar scan {l_idx} for pair {pid} ...", flush=True)
        pts = lidar.points_from_scan_idx(l_idx)
        frames.append(
            {
                "pair_idx": pid,
                "depth": depth_img,
                "color": color_map.get(c_idx),
                "lidar_xyz": pts,
            }
        )

    if not frames:
        print("No usable pairs loaded.")
        return 2
    print(f"Loaded {len(frames)} pair(s). Opening interactive window...", flush=True)

    state = {"i": 0}

    def current_pair_idx() -> Optional[int]:
        return frames[state["i"]]["pair_idx"]

    def depth_backproject(u: float, v: float) -> Optional[np.ndarray]:
        depth = frames[state["i"]]["depth"]
        h, w = depth.shape
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            return None
        # D435 depth has holes on reflective/edge surfaces, so the exact clicked
        # pixel may be invalid. Snap to the nearest VALID depth pixel in a window.
        rad = int(args.pick_search_px)
        v0, v1 = max(0, vi - rad), min(h, vi + rad + 1)
        u0, u1 = max(0, ui - rad), min(w, ui + rad + 1)
        vs, us = np.mgrid[v0:v1, u0:u1]
        zwin = depth[v0:v1, u0:u1].astype(np.float64) * calib.depth_scale
        valid = (zwin >= args.min_depth_m) & (zwin <= args.max_depth_m)
        if not valid.any():
            return None
        dist2 = (us - ui) ** 2 + (vs - vi) ** 2
        dist2 = np.where(valid, dist2, np.iinfo(np.int64).max)
        k = int(np.argmin(dist2))
        uu, vv, z = int(us.flat[k]), int(vs.flat[k]), float(zwin.flat[k])
        x = (uu - calib.cx) / calib.fx * z
        y = (vv - calib.cy) / calib.fy * z
        return np.array([x, y, z], dtype=np.float64)

    def lidar_snap(lateral_y: float, forward_x: float) -> Optional[np.ndarray]:
        pts = frames[state["i"]]["lidar_xyz"]
        if pts.size == 0:
            return None
        # BEV axes: x=lateral (Y=pts[:,1]), y=forward (X=pts[:,0]).
        d2 = (pts[:, 1] - lateral_y) ** 2 + (pts[:, 0] - forward_x) ** 2
        j = int(np.argmin(d2))
        if np.sqrt(d2[j]) > args.snap_radius_m:
            return None
        return pts[j].astype(np.float64)

    # --- box-mode extractors (centroid of the enclosed object) ---
    def depth_box(x0: float, x1: float, y0: float, y1: float) -> Optional[np.ndarray]:
        depth = frames[state["i"]]["depth"]
        h, w = depth.shape
        u0, u1 = max(0, int(np.floor(x0))), min(w, int(np.ceil(x1)) + 1)
        v0, v1 = max(0, int(np.floor(y0))), min(h, int(np.ceil(y1)) + 1)
        if u1 <= u0 or v1 <= v0:
            return None
        vs, us = np.mgrid[v0:v1, u0:u1]
        z = depth[v0:v1, u0:u1].astype(np.float64) * calib.depth_scale
        valid = (z >= args.min_depth_m) & (z <= args.max_depth_m)
        if int(valid.sum()) < args.min_box_points:
            return None
        zc, uc, vc = z[valid], us[valid], vs[valid]
        # Foreground bias: keep the nearest box_depth_pct% of depths.
        thr = np.percentile(zc, args.box_depth_pct)
        fg = zc <= thr
        zc, uc, vc = zc[fg], uc[fg], vc[fg]
        xc = (uc - calib.cx) / calib.fx * zc
        yc = (vc - calib.cy) / calib.fy * zc
        return np.array([np.median(xc), np.median(yc), np.median(zc)], dtype=np.float64)

    def lidar_box(x0: float, x1: float, y0: float, y1: float) -> Optional[np.ndarray]:
        # x=lateral (Y=pts[:,1]), y=forward (X=pts[:,0]).
        pts = frames[state["i"]]["lidar_xyz"]
        if pts.size == 0:
            return None
        m = (pts[:, 1] >= x0) & (pts[:, 1] <= x1) & (pts[:, 0] >= y0) & (pts[:, 0] <= y1)
        if int(m.sum()) < args.min_box_points:
            return None
        q = pts[m]
        return np.array([np.median(q[:, 0]), np.median(q[:, 1]), np.median(q[:, 2])], dtype=np.float64)

    # --- figure setup ---
    use_color = not args.no_color
    ncols = 3 if use_color else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if use_color:
        ax_c, ax_d, ax_l = axes
    else:
        ax_c, (ax_d, ax_l) = None, axes
    fig.suptitle(
        "Pick the SAME physical point in DEPTH then LIDAR (corners/poles/curbs). "
        "Hover for coordinates.  keys: u=undo  n=next  s=solve  q=quit"
    )

    def draw_frame() -> None:
        f = frames[state["i"]]
        depth_m = f["depth"].astype(np.float32) * calib.depth_scale
        # Gray context for the whole scene (up to depth_display_m); colored turbo
        # only for the PICKABLE near range (<= max_depth_m) so it's obvious what
        # can be clicked. Beyond that shows gray (visible but not pickable).
        context = np.where(
            (depth_m >= args.min_depth_m) & (depth_m <= args.depth_display_m), depth_m, np.nan
        )
        pickable = np.where(
            (depth_m >= args.min_depth_m) & (depth_m <= args.max_depth_m), depth_m, np.nan
        )

        if ax_c is not None:
            ax_c.clear()
            if f["color"] is not None:
                ax_c.imshow(f["color"])
            else:
                ax_c.text(0.5, 0.5, "no RGB frame", ha="center", va="center")
            ax_c.set_title("RGB (context only, not clickable)")
            ax_c.axis("off")

        ax_d.clear()
        ax_d.imshow(context, cmap="gray")  # far scene as gray context (not pickable)
        ax_d.imshow(pickable, cmap="turbo")  # near, pickable range in color, on top
        ax_d.set_title(
            f"DEPTH pair {f['pair_idx']} — CLICK the COLORED region (< "
            f"{args.max_depth_m:.0f} m); gray is too far to pick"
        )

        def _depth_fmt(x, y, _d=depth_m):
            ui, vi = int(round(x)), int(round(y))
            if 0 <= vi < _d.shape[0] and 0 <= ui < _d.shape[1]:
                return f"px=({ui},{vi})  depth={_d[vi, ui]:.2f} m"
            return f"px=({ui},{vi})"

        ax_d.format_coord = _depth_fmt

        ax_l.clear()
        pts = f["lidar_xyz"]
        if pts.size:
            ax_l.scatter(pts[:, 1], pts[:, 0], s=4, c=pts[:, 2], cmap="viridis")
        if args.no_auto_range:
            lat_lim, fwd_lim = (-args.max_range_m, args.max_range_m), (0.0, args.max_range_m)
        else:
            lat_lim, fwd_lim = bev_view_limits(pts, args.max_range_m)
        # Flip lateral so left-of-scene is on the LEFT, matching the forward camera
        # (data coords under the cursor are unchanged, so picks/solve are unaffected).
        if args.no_flip_lidar:
            ax_l.set_xlim(lat_lim[0], lat_lim[1])
        else:
            ax_l.set_xlim(lat_lim[1], lat_lim[0])
        ax_l.set_ylim(*fwd_lim)
        ax_l.set_xlabel("Lateral Y (m)  [L/R matches camera]")
        ax_l.set_ylabel("Forward X (m)")
        span = f"{lat_lim[1] - lat_lim[0]:.0f}x{fwd_lim[1] - fwd_lim[0]:.0f} m"
        ax_l.set_title(f"LIDAR BEV ({span})  —  CLICK same point (2nd)")
        ax_l.set_aspect("equal")
        ax_l.grid(True, alpha=0.3)
        fig.canvas.draw_idle()

    _picker = {"p": None}  # holder so advance() can reset box selectors after redraw

    def advance() -> bool:
        if state["i"] + 1 < len(frames):
            state["i"] += 1
            draw_frame()
            p = _picker["p"]
            if p is not None and hasattr(p, "reset_selectors"):
                p.reset_selectors()  # RectangleSelectors are removed by ax.clear()
            return True
        print("Already at last preloaded pair.")
        return False

    def on_solve() -> None:
        _solve_and_write(cs, args, corr_path)

    draw_frame()

    # Keep a strong reference: matplotlib holds event callbacks weakly, so an
    # unreferenced picker gets garbage-collected and clicks silently stop working.
    if args.box:
        from interactive import BoxPairPicker

        picker = BoxPairPicker(
            fig, ax_d, ax_l,
            extract_source_box=depth_box,
            extract_target_box=lidar_box,
            corr_set=cs,
            current_pair_idx=current_pair_idx,
            advance_frame=advance,
            on_solve=on_solve,
        )
        mode_msg = "  DRAG a box around an object in the DEPTH panel, then the same object in LIDAR."
    else:
        from interactive import DualPanelPicker

        picker = DualPanelPicker(
            fig, ax_d, ax_l,
            extract_source=depth_backproject,
            extract_target=lidar_snap,
            corr_set=cs,
            current_pair_idx=current_pair_idx,
            advance_frame=advance,
            on_solve=on_solve,
        )
        mode_msg = "  Click a point in the DEPTH panel, then the same point in the LIDAR panel."
    _picker["p"] = picker
    fig._calib_picker = picker  # extra safety: lives as long as the figure
    print(
        "Interactive calibration ready.\n"
        f"{mode_msg}\n"
        "  u=undo  n=next pair  s=solve+save  q=quit"
    )
    plt.show()

    # Save whatever was picked even if the user quit without pressing 's'.
    if cs.points:
        cs.save(corr_path)
        print(f"Saved {len(cs)} correspondences to {corr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
