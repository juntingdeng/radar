"""Auto-label sync pairs by running YOLO on D435 color and lifting to the lidar frame.

Pipeline (RGB -> lidar):

    color frame --YOLO--> 2D boxes (color pixels)
    color box  --coaxial intrinsic remap--> depth-image box
    depth box  --sample depth + camera_to_lidar--> sensor-topdown (BEV) box

Why the remap: on this D435 the color (640x480, fx~604) and depth (848x480,
fx~430) streams are NOT aligned and the calibration + depth live in the DEPTH
optical frame. A YOLO box in color pixels is mapped to the matching depth pixels
through the shared viewing ray (cx/cy/fx/fy of each stream). The color<->depth
baseline (~15 mm) is ignored -- a few px at indoor range, absorbed by the robust
5-95 percentile box bounds -- so treat outputs as auto-labels for review.

Frames are cached to a LOCAL dir (default ~/.cache/ti_setup_frames/<bag>) so
reruns never re-stream the bag from Google Drive.

    python code/sync/src/camera_projection/yolo_to_lidar.py \
        -d 2026.05.08/17-34-27 --pairs 500 1500 2000

Writes annotations/yolo.json (sync_annotations format). To inspect the labels (camera +
oriented box3d BEV, optional radar panel), use viz_labels.py -- the single label viewer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

_THIS_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _THIS_DIR.parent
# calibration/ holds extrinsic_io (get_extrinsic) used for the camera->lidar + radar extrinsics.
for _p in (_SYNC_DIR, _THIS_DIR, _SYNC_DIR / "calibration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402
from lib.camera_io import (  # noqa: E402
    DEFAULT_COLOR_TOPIC,
    DEFAULT_DEPTH_TOPIC,
    iter_topic_messages,
    parse_ros1_image,
)
from geometry import (  # noqa: E402
    bbox_depth_to_sensor_points,
    load_calibration,
    sensor_points_to_topdown_box,
)
from annotations import empty_annotations, save_annotations  # noqa: E402

# Approx real-world (length, width, height) in meters per COCO class, for a plausibility
# gate on the fitted oriented box (a box within 0.4x-2.5x of these is deemed plausible).
_CLASS_SIZE = {
    "person": (0.6, 0.6, 1.7),
    "bicycle": (1.7, 0.6, 1.4),
    "motorcycle": (2.0, 0.8, 1.4),
    "car": (4.3, 1.8, 1.5),
    "bus": (12.0, 2.6, 3.2),
    "truck": (7.0, 2.5, 3.0),
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--calibration_json", default=None)
    p.add_argument("--camera_bag", default=None)
    p.add_argument("--camera_index", default=None, help="camera_index.npz for chunk seeking.")
    p.add_argument("--camera_sync_csv", default=None)
    p.add_argument("--lidar_pcap", default=None)
    p.add_argument("--lidar_metadata", default=None)
    p.add_argument("--pairs", type=int, nargs="+", default=None,
                   help="Sync pair indices. Omit to use ALL sync pairs (see --pair_stride).")
    p.add_argument("--pair_stride", type=int, default=1,
                   help="Sample every Nth pair (after --pairs, or over ALL pairs if --pairs omitted). "
                        "N>1 gives a fast representative pass over the whole recording.")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model name/path.")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    p.add_argument("--yolo_cache", default=None,
                   help="Reuse pre-computed 2D detections from cache_yolo_detections.py instead of "
                        "running YOLO + fetching camera frames from Drive. Detections are looked up by "
                        "the pair's camera_color_idx; missing frames fall back to live YOLO. This makes "
                        "re-labeling / pad-tuning / calibration adjustments local-only (no Drive I/O).")
    p.add_argument(
        "--classes",
        nargs="+",
        default=["person", "bicycle", "car", "motorcycle", "bus", "truck"],
        help="COCO class names to keep (all others ignored). Vehicle subtype is preserved.",
    )
    p.add_argument("--min_points", type=int, default=20, help="Min depth pts to keep a box.")
    p.add_argument("--min_depth_m", type=float, default=0.2)
    p.add_argument("--max_depth_m", type=float, default=20.0)
    p.add_argument(
        "--fg_band_m",
        type=float,
        default=3.0,
        help="Keep only the foreground depth slab: pixels within this many meters of the "
        "nearest surface in the box (drops background seen through/around the object). "
        "0 disables.",
    )
    p.add_argument(
        "--no_lidar_frustum",
        action="store_true",
        help="Disable lidar-in-frustum range; use D435 depth only (default: use lidar).",
    )
    p.add_argument(
        "--lidar_only",
        action="store_true",
        help="Drop detections without enough lidar points instead of falling back to "
        "D435 depth (keeps only lidar-frustum boxes).",
    )
    p.add_argument(
        "--min_lidar_pts",
        type=int,
        default=4,
        help="Min lidar points inside a detection frustum to build the box from lidar. Low (4) so "
        "far/side/occluded parked cars -- sparse in lidar -- still get a (class-floored, "
        "unreliable-flagged) box instead of being dropped.",
    )
    p.add_argument(
        "--lidar_gap_m",
        type=float,
        default=2.0,
        help="Range gap that separates the nearest lidar cluster (object) from what is "
        "behind it inside the same 2D box.",
    )
    p.add_argument(
        "--ground_pct",
        type=float,
        default=10.0,
        help="Ground level = this height percentile of the in-frustum points.",
    )
    p.add_argument(
        "--min_height_m",
        type=float,
        default=0.15,
        help="Drop the ground slab: keep only lidar points this many meters above the "
        "local ground before fitting the box (0 disables). Low (0.15) so sparse far/parked "
        "cars keep enough body points to survive -- box bottom is still pinned to ground.",
    )
    p.add_argument(
        "--no_split_shared",
        action="store_true",
        help="Disable pair-level point assignment (each detection then fits from all its "
        "in-frustum points; close objects may collapse onto one cluster).",
    )
    p.add_argument(
        "--scene_yaw_deg",
        type=float,
        default=0.0,
        help="Scene forward direction (deg, in the lidar frame) that ALL oriented boxes are "
        "aligned to. 0 = sensor +X (ego/street forward). Set if the sensor is yawed vs the "
        "street so parked vehicles come out slightly off.",
    )
    p.add_argument(
        "--box_sparsity_pad",
        type=float,
        default=0.5,
        help="Loosen each box side by this multiple of the cluster's point spacing (median "
        "nearest-neighbor distance). ~0 for dense/near clusters, larger for sparse/far ones, "
        "so a partially-sampled far object isn't shrink-wrapped too tight. 0 disables.",
    )
    p.add_argument(
        "--box_sparsity_pad_max",
        type=float,
        default=0.5,
        help="Cap (m) on the per-side sparsity pad, so a very sparse cluster can't balloon.",
    )
    p.add_argument(
        "--box_radar_cover_frac",
        type=float,
        default=0.10,
        help="Loosen the box LATERALLY by this fraction of range (+1/4 that forward) to cover "
        "the radar cluster's azimuth blur (~0.1*range = ~12 deg beam) + calibration offset. "
        "So a 10 m object gets ~1 m of lateral slack. 0 disables (lidar-tight box).",
    )
    p.add_argument(
        "--box_radar_cover_max",
        type=float,
        default=1.5,
        help="Cap (m) on the per-side radar-coverage lateral pad.",
    )
    p.add_argument(
        "--box_height_pad",
        type=float,
        default=0.3,
        help="Extra headroom (m) added to the TOP of the box (floor stays at ground). Lidar "
        "scan rings often miss the very top of an object, so this recovers a bit of height. "
        "0 = height from the points only.",
    )
    p.add_argument(
        "--max_size_factor",
        type=float,
        default=2.0,
        help="Reject (drop) a box whose LENGTH or HEIGHT exceeds this multiple of the class prior "
        "(car 4.3 L / 1.5 H, person 0.6 L / 1.7 H) -- catches bad fits merged with a wall/pole "
        "(10 m 'car', 4 m 'person'). Width is excluded (range-padded for radar). 0 disables.",
    )
    p.add_argument(
        "--nms_iou",
        type=float,
        default=0.5,
        help="Drop a box whose top-down footprint overlaps a higher-priority box (reliable > "
        "point-count > score) by more than this BEV IoU AND whose center is within "
        "--nms_dup_dist (so only true duplicates are dropped, not distinct adjacent cars). 0 disables.",
    )
    p.add_argument(
        "--nms_dup_dist",
        type=float,
        default=1.5,
        help="NMS only suppresses an overlapping box if its center is within this many meters of "
        "the kept box (a duplicate of the same object). Larger => more aggressive. 0 = distance-gate off.",
    )
    p.add_argument(
        "--gap_limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap each box's far (away-from-sensor) edge at the next car in front, so class-flooring "
        "doesn't extend a parked car's box into the next one. Keeps the visible near face + fitted extent.",
    )
    p.add_argument(
        "--gap_limit_m",
        type=float,
        default=0.5,
        help="Gap (m) left between a gap-limited box and the car in front of it.",
    )
    p.add_argument(
        "--cover_class_size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Floor each box to at least its class prior size (car ~4.3x1.8x1.5, person "
        "~0.6x0.6x1.7). Single-view lidar sees only the near face/side, so boxes shrink-wrap "
        "to half a car or a torso; this extends length AWAY from the sensor + spans "
        "ground..class-height, so the object (and its radar returns) are covered. --no-cover_class_size "
        "keeps the tight point-fit box.",
    )
    p.add_argument(
        "--lidar_propose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also raise 'car' boxes directly from RADAR-CONFIRMED lidar clusters that no YOLO box "
        "covers -- catches vehicles OUTSIDE the ~57 deg camera FOV (near-lateral parked cars the "
        "camera can't see but lidar + radar clearly do). Clusters the ground-removed cloud, keeps "
        "vehicle-footprint + vehicle-height clusters with a radar return, class-floors to a car.",
    )
    p.add_argument(
        "--lidar_propose_min_pts", type=int, default=12,
        help="Minimum lidar points in a cluster to raise a --lidar_propose box.",
    )
    p.add_argument(
        "--lidar_propose_radar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require a radar strong-return near a --lidar_propose cluster. OFF by default: the "
        "tightened vehicle-geometry filter (substantial major+minor footprint at car height) is "
        "car-specific enough on its own, and per-frame radar is too sparse to gate on (a static "
        "car goes unlabeled in frames where radar doesn't fire). Radar still tags radar_confirmed. "
        "Use --lidar_propose_radar for the stricter (cleaner but intermittent) radar-gated variant.",
    )
    p.add_argument(
        "--lidar_propose_radar_m", type=float, default=2.5,
        help="Radar association distance (m) for --lidar_propose confirmation (more generous than "
        "--radar_assoc_m: proposed cars are farther/lateral where the radar mover-calib bias is larger).",
    )
    p.add_argument(
        "--radar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cross-check every box against radar BEV strong-returns (radar_to_lidar) and "
        "persist radar_confirmed/radar_hits per box. ON by default -- radar is the target "
        "modality, so labels carry their radar association. Use --no-radar to skip (faster).",
    )
    p.add_argument("--radar_h5", default=None)
    p.add_argument("--cfg_file", default=None)
    p.add_argument("--sync_csv", default=None)
    p.add_argument("--sync_summary", default=None)
    p.add_argument(
        "--radar_percentile", type=float, default=99.9,
        help="Percentile of the radar BEV kept as strong returns (higher = stricter).",
    )
    p.add_argument(
        "--radar_bev_res", type=float, default=0.1,
        help="Radar BEV Cartesian grid resolution (m); finer = less blocky "
        "(matches the calibration tool's default 0.1).",
    )
    p.add_argument(
        "--radar_assoc_m", type=float, default=1.5,
        help="A box is radar-confirmed if a radar return lands within this margin (m) "
        "of its footprint.",
    )
    p.add_argument("--output_annotations", default=None)
    p.add_argument("--frame_cache", default=None, help="Local dir for extracted frames.")
    add_dataset_arguments(p)
    return p


def _load_pair_rows(csv_path: Path) -> Dict[int, dict]:
    import csv as _csv

    rows: Dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            rows[int(row["pair_idx"])] = row
    return rows


def _frame_cache_dir(args) -> Path:
    if args.frame_cache:
        base = Path(args.frame_cache)
    else:
        tag = hashlib.md5(str(Path(args.camera_bag).resolve()).encode()).hexdigest()[:10]
        base = Path.home() / ".cache" / "ti_setup_frames" / tag
    base.mkdir(parents=True, exist_ok=True)
    return base


def _extract_frames(
    bag: str,
    topic: str,
    needed: List[int],
    cache: Path,
    kind: str,
    bag_ts_ns: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """Return {frame_idx: array} from local cache, seeking the bag for misses.

    With ``bag_ts_ns`` (per-frame bag-record timestamps from camera_index.npz) each
    miss is fetched by seeking the bag's chunk index to that frame — so a late
    (outdoor) frame streams only its own chunk, not the whole 0..idx prefix.
    Without it, falls back to one sequential scan.
    """
    out: Dict[int, np.ndarray] = {}
    misses: List[int] = []
    for i in needed:
        f = cache / f"{kind}_{i:06d}.npy"
        if f.is_file():
            out[i] = np.load(f)
        else:
            misses.append(i)
    if not misses:
        print(f"  {kind}: all {len(needed)} frame(s) from local cache.")
        return out

    if bag_ts_ns is not None:
        print(f"  {kind}: {len(misses)} miss(es), seeking via chunk index...", flush=True)
        for i in misses:
            if i >= len(bag_ts_ns):
                print(f"    {kind} {i} out of index range; skipping.")
                continue
            start = int(bag_ts_ns[i])
            _ts, raw = next(iter(iter_topic_messages(bag, topic, start_ns=start)))
            arr = np.array(parse_ros1_image(raw).data, copy=True)
            np.save(cache / f"{kind}_{i:06d}.npy", arr)
            out[i] = arr
            print(f"    seeked+cached {kind} {i}", flush=True)
        return out

    mx, ns = max(misses), set(misses)
    print(f"  {kind}: {len(misses)} miss(es), scanning bag to frame {mx}...", flush=True)
    for i, (_ts, raw) in enumerate(iter_topic_messages(bag, topic)):
        if i > mx:
            break
        if i in ns:
            arr = np.array(parse_ros1_image(raw).data, copy=True)
            np.save(cache / f"{kind}_{i:06d}.npy", arr)
            out[i] = arr
            print(f"    cached {kind} {i}", flush=True)
    return out


def _foreground_slab(points: dict, band_m: float) -> dict:
    """Keep the nearest depth slab in a bbox's points, dropping far background.

    A 2D detection box also spans background (road/trees/building behind the object,
    seen through windows and around edges). Those far pixels inflate the projected box
    length. The object is the *nearest* coherent surface, so keep only points within
    ``band_m`` of the near depth (10th percentile, robust to a few closer outliers).
    """
    z = np.asarray(points["depth_m"], dtype=np.float64)
    finite = np.isfinite(z)
    if band_m <= 0 or int(finite.sum()) == 0:
        return points
    near = float(np.percentile(z[finite], 10))
    keep = finite & (z <= near + float(band_m))
    return {k: np.asarray(v)[keep] for k, v in points.items()}


def _project_to_image(lidar_pts, lid2cam, dcal):
    """Project sensor-frame lidar into the depth image. Returns (u, v, front_mask)."""
    cam = lidar_pts @ lid2cam[:3, :3].T + lid2cam[:3, 3]
    z = cam[:, 2]
    front = z > 0.1
    dfx, dfy, dcx, dcy = dcal
    u = np.full(z.shape, -1e9)
    v = np.full(z.shape, -1e9)
    zf = z[front]
    u[front] = dfx * cam[front, 0] / zf + dcx
    v[front] = dfy * cam[front, 1] / zf + dcy
    return u, v, front


def _drop_ground(pts: np.ndarray, ground_pct: float, min_height_m: float) -> np.ndarray:
    """Drop the ground slab: keep points more than ``min_height_m`` above the local
    ground level (estimated as the ``ground_pct`` percentile of height z in this set).

    A 2D detection box back-projects to a wedge that skims the continuous ground plane
    (no range gap, so range clustering can't remove it) — the ground is what inflates the
    top-down footprint. Cutting the low slab leaves the object's vertical column, whose XY
    extent is the true footprint. Ouster sensor frame: column 2 is height (Z up).
    """
    if pts.shape[0] == 0 or min_height_m <= 0:
        return pts
    ground = _ground_level(pts, ground_pct)
    keep = pts[:, 2] > ground + float(min_height_m)
    return pts[keep]


def _ground_level(pts: np.ndarray, ground_pct: float) -> float:
    """Local ground height = the ``ground_pct`` percentile of point height (Z up)."""
    return float(np.percentile(pts[:, 2], ground_pct)) if pts.shape[0] else 0.0


def _largest_xy_cluster(pts: np.ndarray, radius: float = 0.4) -> np.ndarray:
    """Keep the largest spatially-connected component in XY.

    A detection frustum's nearest-range cluster still catches ground scan-rings that
    leaked past the height cut — sparse arcs that spread out in lateral and drag the
    oriented-box fit (rotating/inflating it off the object). Those arcs are separated
    from the compact object body by gaps > ``radius``, so keeping the largest connected
    component drops them and leaves the object's clean rectangle.
    """
    n = pts.shape[0]
    if n < 8:
        return pts
    try:
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
    except Exception:  # noqa: BLE001 - scipy always present here, but be safe
        return pts
    pairs = cKDTree(pts[:, :2]).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return pts
    g = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _ncomp, lab = connected_components(g, directed=False)
    big = int(np.argmax(np.bincount(lab)))
    keep = lab == big
    return pts[keep] if int(keep.sum()) >= 6 else pts


def _footprint_iou(a: dict, b: dict) -> float:
    """BEV IoU of two axis-aligned (yaw=0) box footprints (forward x lateral)."""
    fx0, fx1 = max(a["forward"][0], b["forward"][0]), min(a["forward"][1], b["forward"][1])
    lx0, lx1 = max(a["lateral"][0], b["lateral"][0]), min(a["lateral"][1], b["lateral"][1])
    if fx1 <= fx0 or lx1 <= lx0:
        return 0.0
    inter = (fx1 - fx0) * (lx1 - lx0)
    aa = (a["forward"][1] - a["forward"][0]) * (a["lateral"][1] - a["lateral"][0])
    ab = (b["forward"][1] - b["forward"][0]) * (b["lateral"][1] - b["lateral"][0])
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0


def _cluster_bev(xy: np.ndarray, roi, res: float = 0.35, link_cells: int = 1) -> np.ndarray:
    """Fast BEV clustering via connected components on a voxel occupancy grid (labels per point).

    Rasterises the points into a ``res``-metre grid, dilates by ``link_cells`` (so points within
    ~2*link_cells*res link across small gaps -- ~0.7 m at defaults), and connected-components the
    occupancy with 8-connectivity. O(occupied cells); avoids the query_pairs blow-up on dense
    walls/foliage. Returns a per-point cluster id (>=1)."""
    from scipy import ndimage
    n = xy.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)
    x0, x1, y0, y1 = roi
    nx = int((x1 - x0) / res) + 2
    ny = int((y1 - y0) / res) + 2
    ix = np.clip(np.floor((xy[:, 0] - x0) / res).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor((xy[:, 1] - y0) / res).astype(np.int64), 0, ny - 1)
    grid = np.zeros((nx, ny), dtype=bool)
    grid[ix, iy] = True
    if link_cells > 0:
        grid = ndimage.binary_dilation(grid, iterations=link_cells)
    lab, _n = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
    return lab[ix, iy]


def _lidar_box_proposals(pts, rf, rl, existing, *, ground_pct, min_height_m, scene_yaw,
                         radar_assoc_m, min_pts, class_size, require_radar=True,
                         roi=(0.0, 30.0, -16.0, 16.0),
                         major_min=2.0, major_max=6.5, minor_min=1.2, minor_max=3.5,
                         h_min=0.5, h_max=2.4, iou_skip=0.15) -> list:
    """Propose 'car' boxes for vehicle-sized lidar clusters that no existing (YOLO) box covers --
    catches objects OUTSIDE the ~57 deg camera FOV that lidar (+ radar) see.

    Clusters the ground-removed cloud in the ROI, keeps vehicle-footprint + vehicle-height
    clusters that don't overlap an existing box, and class-floors them to a car. With
    ``require_radar`` (default) a cluster must also have a radar strong-return nearby -- clean but
    intermittent (per-frame radar is sparse; a static low-RCS car may not exceed threshold every
    frame). ``require_radar=False`` proposes on lidar geometry alone (radar still tags the
    ``radar_confirmed`` flag downstream). Returns box dicts (source='lidar_radar_proposal')."""
    if pts is None or pts.shape[0] < min_pts:
        return []
    if require_radar and (rf is None or np.asarray(rf).size == 0):
        return []
    x0, x1, y0, y1 = roi
    m = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    reg = pts[m]
    ground_z = _ground_level(reg, ground_pct)
    obj = _drop_ground(reg, ground_pct, min_height_m)
    if obj.shape[0] < min_pts:
        return []
    labels = _cluster_bev(obj[:, :2], roi)
    rf = np.asarray(rf); rl = np.asarray(rl)
    _dbg = os.environ.get("PROP_DEBUG")
    out = []
    for lb in np.unique(labels):
        cl = obj[labels == lb]
        if cl.shape[0] < min_pts:
            continue
        L = float(cl[:, 0].max() - cl[:, 0].min())
        W = float(cl[:, 1].max() - cl[:, 1].min())
        H = float(cl[:, 2].max() - cl[:, 2].min())
        mx, mn = max(L, W), min(L, W)
        # Vehicle geometry: a substantial major AND minor footprint (an L-shape / angled view --
        # what near-lateral parked cars present) at car height. This rejects thin fences/railings
        # (small minor), small bushes/hydrants (small major), and buildings (height culls them).
        sizeok = (major_min <= mx <= major_max and minor_min <= mn <= minor_max
                  and h_min <= H <= h_max)
        cen = cl[:, :2].mean(0)
        rok = bool((np.hypot(rf - cen[0], rl - cen[1]) < radar_assoc_m + mx / 2.0).any()) if rf.size else False
        if _dbg:
            print(f"    [prop] cluster n={cl.shape[0]:4d} cen=({cen[0]:5.1f},{cen[1]:6.1f}) "
                  f"L={L:4.1f} W={W:4.1f} H={H:4.1f} sizeok={sizeok} radar={rok}")
        if not sizeok:
            continue
        if require_radar and not rok:
            continue   # no radar return -> not confirmed
        b3 = _oriented_box_from_points(cl, ground_z=ground_z, scene_yaw=scene_yaw,
                                       class_size=class_size)
        hl, hw = b3["length"] / 2.0, b3["width"] / 2.0
        ca, sa = abs(np.cos(b3["yaw"])), abs(np.sin(b3["yaw"]))
        hx, hy = hl * ca + hw * sa, hl * sa + hw * ca
        box = {"lateral": [b3["cy"] - hy, b3["cy"] + hy],
               "forward": [max(0.0, b3["cx"] - hx), b3["cx"] + hx], "box3d": b3,
               "n_pts": int(cl.shape[0])}
        if any(_footprint_iou(box, e) > iou_skip for e in existing):
            continue   # already covered by a YOLO box
        out.append(box)
    return out


def _gap_limit_flooring(boxes: list, gap_m: float) -> list:
    """Cap each box's FAR (away-from-sensor) edge at the near edge of the next box in front of
    it (laterally overlapping), so class-flooring doesn't extend one parked car's box into the
    next. The visible near face is preserved; never shrinks below the fitted extent (length_fit).
    Boxes are scene-forward (yaw=0), so near/far = cx-/+length/2 along forward X.
    """
    if gap_m < 0 or len(boxes) < 2:
        return boxes
    for bi in boxes:
        oi = bi.get("box3d")
        if not oi:
            continue
        near_i = oi["cx"] - oi["length"] / 2.0
        far_i = oi["cx"] + oi["length"] / 2.0
        lat0_i, lat1_i = oi["cy"] - oi["width"] / 2.0, oi["cy"] + oi["width"] / 2.0
        cap = far_i
        for bj in boxes:
            oj = bj.get("box3d")
            if bj is bi or not oj:
                continue
            near_j = oj["cx"] - oj["length"] / 2.0
            lat0_j, lat1_j = oj["cy"] - oj["width"] / 2.0, oj["cy"] + oj["width"] / 2.0
            if near_j > near_i and not (lat1_j < lat0_i or lat0_j > lat1_i):   # j in front + lat overlap
                cap = min(cap, near_j - gap_m)
        new_far = max(cap, near_i + float(oi.get("length_fit", oi["length"])))  # keep visible extent
        if new_far < far_i - 1e-3:
            new_len = new_far - near_i
            oi["length"] = float(new_len)
            oi["cx"] = float(near_i + new_len / 2.0)
            bi["forward"] = [max(0.0, near_i), float(new_far)]
    return boxes


def _nms_footprint(boxes: list, iou_thresh: float, dup_dist: float = 1.5) -> list:
    """Drop a box whose footprint overlaps a higher-priority box by > iou_thresh AND whose
    CENTER is within ``dup_dist`` m of it (i.e. a duplicate of the SAME object, not a distinct
    adjacent car). Priority: reliable, then lidar point count, then YOLO score. yaw=0 so BEV
    IoU is a simple rectangle IoU.
    """
    if iou_thresh <= 0 or len(boxes) < 2:
        return boxes

    def area(b):
        return (max(0.0, b["forward"][1] - b["forward"][0])
                * max(0.0, b["lateral"][1] - b["lateral"][0]))

    def iou(a, b):
        fx0, fx1 = max(a["forward"][0], b["forward"][0]), min(a["forward"][1], b["forward"][1])
        lx0, lx1 = max(a["lateral"][0], b["lateral"][0]), min(a["lateral"][1], b["lateral"][1])
        if fx1 <= fx0 or lx1 <= lx0:
            return 0.0
        inter = (fx1 - fx0) * (lx1 - lx0)
        union = area(a) + area(b) - inter
        return inter / union if union > 0 else 0.0

    def cen(b):
        o = b.get("box3d") or {}
        return float(o.get("cx", 0.0)), float(o.get("cy", 0.0))

    def prio(b):
        o = b.get("box3d") or {}
        return (bool(o.get("reliable")), int(b.get("n_depth_points", 0)), float(b.get("score") or 0))

    order = sorted(range(len(boxes)), key=lambda i: prio(boxes[i]), reverse=True)
    dropped = set()
    for k, i in enumerate(order):
        if i in dropped:
            continue
        cxi, cyi = cen(boxes[i])
        for j in order[k + 1:]:
            if j in dropped:
                continue
            cxj, cyj = cen(boxes[j])
            same_obj = (dup_dist <= 0) or (np.hypot(cxi - cxj, cyi - cyj) < dup_dist)
            if same_obj and iou(boxes[i], boxes[j]) > iou_thresh:
                dropped.add(j)
    return [b for idx, b in enumerate(boxes) if idx not in dropped]


def _cluster_spacing(xy: np.ndarray) -> float:
    """Characteristic point spacing (m) = median nearest-neighbor distance in XY.

    Grows with range / sparsity (a far, partially-seen object samples coarsely) and is
    ~0 for a dense near cluster. Used to pad the box: the outermost sampled point is inset
    from the true object edge by ~half a point spacing, so a sparse cluster shrink-wraps
    smaller than the object it under-samples.
    """
    if xy.shape[0] < 2:
        return 0.0
    from scipy.spatial import cKDTree
    d, _ = cKDTree(xy).query(xy, k=2)   # col 0 = self (0), col 1 = nearest neighbor
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if nn.size else 0.0


def _sparsity_pad(xy: np.ndarray, mult: float, cap: float) -> float:
    """Per-side box padding (m) from cluster sparsity: ``mult`` * point spacing, capped.

    ``mult`` <= 0 disables it (exact shrink-wrap). Default ~0.5 pads each side by about
    half the point spacing, which is negligible for dense clusters and meaningful for far
    sparse ones -- so the box loosens exactly where the points under-represent the object.
    """
    if mult <= 0.0:
        return 0.0
    return float(min(mult * _cluster_spacing(xy), max(0.0, cap)))


def _oriented_box_from_points(sel: np.ndarray, percentiles=(5.0, 95.0),
                              min_pts_reliable: int = 15, ground_z=None,
                              scene_yaw: float = 0.0,
                              pad_major: float = 0.0, pad_minor: float = 0.0,
                              z_pad: float = 0.0, height_pad: float = 0.0,
                              class_size=None) -> dict:
    """3D box (x,y,z,yaw,l,w,h) aligned to the SCENE FORWARD direction.

    In a street/driving scene the objects are oriented along the ego/street axis, and a
    per-object edge fit on partial single-view lidar is noisy (it locks onto whichever
    face is visible — e.g. a car's front). So EVERY box is aligned to the scene forward
    direction ``scene_yaw`` (default 0 = sensor +X = ego forward); ``length`` is the extent
    along forward and ``width`` the extent across (lateral). Frame: sensor (X forward, Y
    lateral, Z up), yaw about +Z from +X. NOTE: ``h`` is the above-ground-slab height.
    """
    from geometry import robust_bounds

    xy = sel[:, :2].astype(np.float64)          # [forward X, lateral Y]
    c = xy.mean(axis=0)
    d = xy - c
    ext_pct = (2.0, 98.0)                       # capture the full footprint
    yaw = float(scene_yaw)
    major = np.array([np.cos(yaw), np.sin(yaw)])   # scene forward
    minor = np.array([-np.sin(yaw), np.cos(yaw)])  # scene lateral
    l0, l1 = robust_bounds(d @ major, ext_pct)
    w0, w1 = robust_bounds(d @ minor, ext_pct)
    # padding: major (=forward/length) by pad_major, minor (=lateral/width) by pad_minor.
    # See _box_from_points -- pad_minor is larger (radar azimuth blur is lateral).
    l0, l1, w0, w1 = l0 - pad_major, l1 + pad_major, w0 - pad_minor, w1 + pad_minor
    length, width = float(l1 - l0), float(w1 - w0)
    center = c + 0.5 * (l0 + l1) * major + 0.5 * (w0 + w1) * minor
    z = sel[:, 2].astype(np.float64)
    z0, z1 = robust_bounds(z, percentiles)
    # full object height: bottom = local ground (the dropped slab), top = robust max + z_pad
    # (sparsity) + height_pad (extra headroom -- lidar rings often miss the very top).
    z_bottom = float(ground_z) if ground_z is not None else float(z0)
    z_top = float(z1) + z_pad + height_pad
    # Class-size flooring: partial single-view lidar sees only the near face/side, so the box
    # shrink-wraps to a fraction of the object (half a car; a person's torso, no feet/head).
    # Floor each dimension to the class prior -- LENGTH extends AWAY from the sensor (keep the
    # visible near face), WIDTH inflates symmetrically, HEIGHT spans ground..class-height.
    length_fit = float(length)   # pre-flooring extent (covers the visible points + pads)
    if class_size is not None:
        cl, cw, ch = class_size
        if length < cl:
            out = major if float(center @ major) >= 0 else -major
            center = center + 0.5 * (cl - length) * out
            length = float(cl)
        if width < cw:
            width = float(cw)
        if (z_top - z_bottom) < ch:
            z_top = z_bottom + float(ch)
    return {
        "cx": float(center[0]), "cy": float(center[1]), "cz": float(0.5 * (z_bottom + z_top)),
        "yaw": yaw, "length": length, "width": width, "height": float(z_top - z_bottom),
        "length_fit": length_fit,   # for gap-limiting: never shrink below this
        # orientation/extent is only meaningful with enough points; sparse/far clusters
        # give a thin, unreliable box (see the pair-4500 far detections).
        "n_points": int(sel.shape[0]),
        "reliable": bool(sel.shape[0] >= int(min_pts_reliable)),
    }


def _box_from_points(
    pts: np.ndarray,
    *,
    min_pts: int,
    gap_m: float,
    ground_pct: float = 10.0,
    min_height_m: float = 0.3,
    percentiles=(5.0, 95.0),
    scene_yaw: float = 0.0,
    sparsity_pad: float = 0.5,
    sparsity_pad_max: float = 0.5,
    radar_cover_frac: float = 0.10,
    radar_cover_max: float = 1.5,
    height_pad: float = 0.3,
    class_size=None,
) -> Optional[dict]:
    """Fit a top-down box from a set of sensor-frame points: drop ground, take the
    nearest range cluster (drop what's behind), then robust XY bounds + an oriented 3D box.

    ``class_size`` (l, w, h): if given, the box is floored to at least that size (partial
    single-view lidar under-covers -- half a car, a torso). See _oriented_box_from_points.

    Two padding terms loosen the box:
      * ``sparsity_pad`` (isotropic): ``sparsity_pad`` * point-spacing per side (cap
        ``sparsity_pad_max``) -- so a sparse/far cluster that under-samples the object isn't
        shrink-wrapped. 0 disables.
      * ``radar_cover_frac`` (lateral-dominant, range-scaled): the radar return is an azimuth
        blur ~``range*tan(6deg)`` wide plus a ~0.5 m calibration offset, so to actually COVER
        the (inaccurate) radar cluster the box is loosened laterally by ``radar_cover_frac *
        range`` (cap ``radar_cover_max``) and forward by a quarter of that (range is precise).
        0 disables.
    """
    from geometry import robust_bounds

    if pts is None or pts.shape[0] < int(min_pts):
        return None
    # local ground (from the pre-removal in-frustum points) -> full-height boxes.
    ground_z = _ground_level(pts, ground_pct) if min_height_m > 0 else None
    pts = _drop_ground(pts, ground_pct, min_height_m)
    if pts.shape[0] < int(min_pts):
        return None
    rng = np.hypot(pts[:, 0], pts[:, 1])
    order = np.argsort(rng)
    gaps = np.diff(rng[order])
    keep_n = int(np.argmax(gaps > gap_m)) + 1 if np.any(gaps > gap_m) else order.size
    sel = pts[order[:keep_n]]
    # drop only genuinely disconnected outliers; keep the whole object body (walls AND
    # roof/far-side points that define the lateral width).
    sel = _largest_xy_cluster(sel)
    if sel.shape[0] < int(min_pts):
        return None
    fwd, lat = sel[:, 0], sel[:, 1]
    lat0, lat1 = robust_bounds(lat, percentiles)
    fwd0, fwd1 = robust_bounds(fwd, percentiles)
    if fwd1 <= 0:
        return None
    # padding (shared by the footprint and the oriented 3D box):
    #  sp = lidar-sparsity (isotropic); rc = radar-coverage (lateral, range-scaled).
    sp = _sparsity_pad(sel[:, :2], sparsity_pad, sparsity_pad_max)
    rng_c = float(np.median(np.hypot(fwd, lat)))
    rc = min(radar_cover_frac * rng_c, radar_cover_max) if radar_cover_frac > 0 else 0.0
    pad_lat = sp + rc                # lateral: sparsity + full radar azimuth blur
    pad_fwd = sp + 0.25 * rc         # forward: range is precise -> small radar share
    b3d = _oriented_box_from_points(sel, percentiles, ground_z=ground_z, scene_yaw=scene_yaw,
                                    pad_major=pad_fwd, pad_minor=pad_lat, z_pad=sp,
                                    height_pad=height_pad, class_size=class_size)
    # Derive the top-down footprint from the FINAL (padded + class-floored) box3d, so the
    # radar-association footprint matches the drawn box (axis-aligned bbox of the oriented box).
    hl, hw = b3d["length"] / 2.0, b3d["width"] / 2.0
    ca, sa = abs(np.cos(b3d["yaw"])), abs(np.sin(b3d["yaw"]))
    half_x, half_y = hl * ca + hw * sa, hl * sa + hw * ca
    return {
        "lateral": [b3d["cy"] - half_y, b3d["cy"] + half_y],
        "forward": [max(0.0, b3d["cx"] - half_x), b3d["cx"] + half_x],
        "box3d": b3d,
        "n_depth_points": int(sel.shape[0]),
        "median_depth_m": float(np.median(np.hypot(fwd, lat))),
    }


def _lidar_frustum_box(
    lidar_pts: Optional[np.ndarray],
    lid2cam: np.ndarray,
    dcal,
    dbox: List[float],
    *,
    min_pts: int,
    gap_m: float,
    ground_pct: float = 10.0,
    min_height_m: float = 0.3,
    percentiles=(5.0, 95.0),
    scene_yaw: float = 0.0,
    sparsity_pad: float = 0.5,
    sparsity_pad_max: float = 0.5,
    radar_cover_frac: float = 0.10,
    radar_cover_max: float = 1.5,
    height_pad: float = 0.3,
    class_size=None,
) -> Optional[dict]:
    """Build a top-down box from the lidar points inside a detection's frustum.

    Lidar has accurate range where D435 stereo does not (outdoors, > ~6-8 m). We project
    the sensor-frame lidar cloud into the depth image, keep points landing inside the
    (color->depth mapped) 2D box, drop the ground slab (which otherwise inflates the
    footprint), take the *nearest* range cluster, and fit the box. Returns ``None`` when
    too few lidar points hit the box, so the caller can fall back to depth.
    """
    if lidar_pts is None or lidar_pts.shape[0] == 0:
        return None
    u, v, _front = _project_to_image(lidar_pts, lid2cam, dcal)
    x0, y0, x1, y1 = dbox
    inb = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
    if int(np.count_nonzero(inb)) < int(min_pts):
        return None
    return _box_from_points(
        lidar_pts[inb], min_pts=min_pts, gap_m=gap_m,
        ground_pct=ground_pct, min_height_m=min_height_m, percentiles=percentiles,
        scene_yaw=scene_yaw, sparsity_pad=sparsity_pad, sparsity_pad_max=sparsity_pad_max,
        radar_cover_frac=radar_cover_frac, radar_cover_max=radar_cover_max,
        height_pad=height_pad, class_size=class_size,
    )


def _assign_frustum_points(dboxes: List[List[float]], lidar_pts, lid2cam, dcal):
    """Assign each in-frustum lidar point to ONE detection (smallest 2D box containing it).

    Two close objects yield overlapping frustums; without unique assignment each box grabs
    the shared cluster and they collapse onto the same footprint. Giving a shared point to
    the smallest (most specific) box separates a person next to / in front of a vehicle.
    Returns a list (per dbox) of the sensor-frame point subsets assigned to it.
    """
    if lidar_pts is None or lidar_pts.shape[0] == 0 or not dboxes:
        return [np.empty((0, 3)) for _ in dboxes]
    u, v, _front = _project_to_image(lidar_pts, lid2cam, dcal)
    areas = [max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for b in dboxes]
    n = lidar_pts.shape[0]
    best = np.full(n, -1, dtype=int)
    best_area = np.full(n, np.inf)
    for j, (x0, y0, x1, y1) in enumerate(dboxes):
        inb = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
        take = inb & (areas[j] < best_area)
        best[take] = j
        best_area[take] = areas[j]
    return [lidar_pts[best == j] for j in range(len(dboxes))]


def _radar_setup(args, calib, *, doppler_mode=None, processing=None):
    """Build a RadarLoader + radar_to_lidar transform, or (None, None) if unavailable.

    ``doppler_mode``/``processing`` override the canonical moving-target CALIB chain (used by
    label generation) -- e.g. viz_labels passes ``doppler_mode="all", processing="legacy"`` so
    the DISPLAY panel keeps the static returns (a static wall would be notched out by ti/moving).
    """
    _cal = _SYNC_DIR / "calibration"
    if str(_cal) not in sys.path:
        sys.path.insert(0, str(_cal))
    from extrinsic_io import get_extrinsic, has_extrinsic  # noqa: E402

    if not has_extrinsic(calib, "radar_to_lidar"):
        print("  no radar_to_lidar in calibration; skipping radar cross-check.")
        return None, None
    import make_video as mv

    radar_args = mv.build_argparser().parse_args([])
    for k in ("radar_h5", "cfg_file", "sync_csv", "sync_summary", "lidar_pcap", "lidar_metadata"):
        v = getattr(args, k, None)
        if v:
            setattr(radar_args, k, v)
    # Use the SAME clutter-suppressed moving-target chain (ti + moving + dbf_nci) that
    # calibrate_radar_lidar solved radar_to_lidar on, so the cross-check BEV matches the
    # calibration and the other radar tools (not make_video's noisy legacy/all/capon default).
    # Set explicitly: make_video's default radar_processing="legacy" is truthy, so
    # apply_calib_radar_defaults' `or` would NOT override it -- force the canonical constants.
    from lib.bev_render import (
        CALIB_RADAR_AOA, CALIB_RADAR_DOPPLER_MODE, CALIB_RADAR_PROCESSING,
    )
    radar_args.radar_processing = processing or CALIB_RADAR_PROCESSING   # "ti": clutter + zero-Doppler notch
    radar_args.radar_doppler_mode = doppler_mode or CALIB_RADAR_DOPPLER_MODE  # "moving": drop static clutter
    radar_args.radar_aoa = CALIB_RADAR_AOA                 # "dbf_nci"
    loader = mv.RadarLoader(radar_args)
    # Finer Cartesian BEV grid so it isn't blocky (matches calibrate_radar_lidar.py).
    if hasattr(loader, "display_bev_kw"):
        loader.display_bev_kw["res"] = float(getattr(args, "radar_bev_res", 0.1))
    # Prefer an auto-solved refinement (radar_to_lidar_auto) over the hand-picked one.
    R, src = get_extrinsic(calib, "radar_to_lidar")
    print(f"  radar cross-check using '{src}'.")
    return loader, R


def _radar_panel(loader, pid):
    """Return the radar BEV heatmap and its (lateral, forward) axes for one pair."""
    bev, _ = loader.load_radar_panel(int(pid))
    n_fwd, n_lat = bev.shape
    lat_axis = np.linspace(loader.lateral_range[0], loader.lateral_range[1], n_lat)
    fwd_axis = np.linspace(loader.forward_range[0], loader.forward_range[1], n_fwd)
    return bev, lat_axis, fwd_axis


def _peaks_from_bev(bev, lat_axis, fwd_axis, R, percentile):
    """Strong radar-return cells -> (forward, lateral) in the lidar frame."""
    thr = np.nanpercentile(bev, float(percentile))
    fi, li = np.where(np.nan_to_num(bev) >= thr)
    rf, rl = fwd_axis[fi], lat_axis[li]  # radar frame (forward, lateral)
    lf = R[0, 0] * rf + R[0, 1] * rl + R[0, 3]
    ll = R[1, 0] * rf + R[1, 1] * rl + R[1, 3]
    return lf, ll  # lidar frame (forward, lateral)


def _radar_hits(box, rf, rl, margin) -> int:
    """Count radar returns landing within `margin` m of a box's top-down footprint."""
    if rf is None or rf.size == 0:
        return 0
    fwd0, fwd1 = box["forward"]
    lat0, lat1 = box["lateral"]
    inb = (
        (rf >= fwd0 - margin) & (rf <= fwd1 + margin)
        & (rl >= lat0 - margin) & (rl <= lat1 + margin)
    )
    return int(np.count_nonzero(inb))


def _color_box_to_depth(bbox_xyxy, ccal, dcal) -> List[float]:
    """Map a color-image xyxy box to depth-image pixels via the shared viewing ray."""
    x0, y0, x1, y1 = bbox_xyxy
    (cfx, cfy, ccx, ccy) = ccal
    (dfx, dfy, dcx, dcy) = dcal
    ud0 = (x0 - ccx) / cfx * dfx + dcx
    ud1 = (x1 - ccx) / cfx * dfx + dcx
    vd0 = (y0 - ccy) / cfy * dfy + dcy
    vd1 = (y1 - ccy) / cfy * dfy + dcy
    return [min(ud0, ud1), min(vd0, vd1), max(ud0, ud1), max(vd0, vd1)]


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())
    if not args.pairs and args.pair_stride <= 1:
        raise SystemExit("Give --pairs (sync pair indices), or --pair_stride N to sample ALL pairs.")

    with open(args.calibration_json, "r", encoding="utf-8") as f:
        calib_raw = json.load(f)
    # depth-frame intrinsics + camera_to_lidar (depth optical -> lidar). Prefer an
    # ICP-refined camera_to_lidar_auto over the hand-picked seed (get_extrinsic).
    from extrinsic_io import get_extrinsic
    try:
        cam2lid, cam_src = get_extrinsic(calib_raw, "camera_to_lidar")
    except KeyError:
        raise SystemExit("No camera_to_lidar in calibration JSON.")
    print(f"Using camera->lidar extrinsic: {cam_src}")
    calib = load_calibration(args.calibration_json)
    di = calib_raw["intrinsics"]
    dcal = (float(di["fx"]), float(di["fy"]), float(di["cx"]), float(di["cy"]))
    ci = calib_raw.get("color_intrinsics")
    if not ci:
        raise SystemExit("No color_intrinsics in calibration JSON.")
    ccal = (float(ci["fx"]), float(ci["fy"]), float(ci["cx"]), float(ci["cy"]))
    lid2cam = np.linalg.inv(cam2lid)
    use_lidar = not args.no_lidar_frustum

    rows = _load_pair_rows(Path(args.camera_sync_csv))
    sel = sorted(args.pairs) if args.pairs else sorted(rows)
    if args.pair_stride > 1:
        sel = sel[:: args.pair_stride]
        print(f"pair_stride {args.pair_stride}: {len(sel)} of {len(rows)} pairs "
              f"({sel[0]}..{sel[-1]}).")
    plan: List[Tuple[int, int, int, int]] = []
    for pid in sel:
        r = rows.get(pid)
        if not r:
            print(f"  pair {pid} not in CSV; skipping.")
            continue
        l_idx = int(r.get("lidar_idx", -1))
        c_idx = int(r.get("camera_color_idx", r.get("camera_idx", -1)))
        d_idx = int(r.get("camera_depth_idx", -1))
        if min(l_idx, c_idx, d_idx) < 0:
            print(f"  pair {pid}: missing lidar/color/depth idx; skipping.")
            continue
        plan.append((pid, l_idx, c_idx, d_idx))
    if not plan:
        raise SystemExit("No usable pairs.")

    color_bag_ts = depth_bag_ts = None
    ipath = args.camera_index
    if ipath and Path(ipath).is_file():
        _idx = np.load(ipath, allow_pickle=False)
        if "color_bag_ts_ns" in _idx:
            color_bag_ts = _idx["color_bag_ts_ns"]
        if "depth_bag_ts_ns" in _idx:
            depth_bag_ts = _idx["depth_bag_ts_ns"]
        print(f"Using camera index for chunk seeking: {ipath}")
    else:
        print("No camera_index.npz; falling back to sequential scan (slow for late frames).")

    cache = _frame_cache_dir(args)
    # Optional pre-computed YOLO cache (cache_yolo_detections.py): skip the Drive frame fetch
    # AND live YOLO for any frame present in it; only cache-miss frames need the camera.
    yolo_cache = None
    if args.yolo_cache:
        with open(args.yolo_cache) as f:
            yolo_cache = json.load(f)["dets"]
        print(f"Using YOLO cache: {args.yolo_cache} ({len(yolo_cache)} frames).")
    needed_color = sorted({c for _, _, c, _ in plan})
    if yolo_cache is not None:
        miss = [c for c in needed_color if str(c) not in yolo_cache]
        if miss:
            print(f"  {len(miss)} pair(s) not in YOLO cache -> live YOLO (fetch their camera frame).")
        needed_color = miss
    color_map = _extract_frames(
        args.camera_bag, DEFAULT_COLOR_TOPIC, needed_color, cache, "color", color_bag_ts,
    ) if needed_color else {}
    # Depth frames are only used by the D435 fallback path; --lidar_only disables it, so
    # skip the (slow, Drive-streamed) depth extraction entirely in that case.
    if args.lidar_only:
        depth_map: Dict[int, np.ndarray] = {}
    else:
        depth_map = _extract_frames(
            args.camera_bag, DEFAULT_DEPTH_TOPIC,
            sorted({d for _, _, _, d in plan}), cache, "depth", depth_bag_ts,
        )

    keep_names = set(args.classes)
    # Load YOLO only if we actually need to run it live (no cache, or some frames are missing).
    model = None
    keep_ids: List[int] = []
    if yolo_cache is None or needed_color:
        from ultralytics import YOLO

        model = YOLO(args.model)
        name_to_id = {v: k for k, v in model.names.items()}
        for nm in args.classes:
            if nm in name_to_id:
                keep_ids.append(name_to_id[nm])
            else:
                print(f"  WARNING: class '{nm}' not in model; ignoring.")
        if not keep_ids:
            raise SystemExit("No valid --classes matched the model's class names.")
        print(f"Keeping classes: {[model.names[i] for i in keep_ids]}")
    else:
        print(f"Keeping classes (from YOLO cache): {sorted(keep_names)}")

    # Open the lidar pcap once (indexed) for in-frustum range.
    lidar_source = lidar_meta = None
    lidar_by_pair: Dict[int, Optional[np.ndarray]] = {}
    if use_lidar:
        from lib.lidar_io import open_pcap_scan_source, sensor_info_from_source

        lidar_source = open_pcap_scan_source(args.lidar_pcap, args.lidar_metadata, index=True)
        lidar_meta = sensor_info_from_source(lidar_source)
    if use_lidar:
        for pid, l_idx, _c, _d in plan:
            try:
                lidar_by_pair[pid] = _lidar_points_at(lidar_source, lidar_meta, l_idx)
            except Exception as exc:  # noqa: BLE001
                print(f"  pair {pid}: lidar read failed ({exc}); will use depth.")
                lidar_by_pair[pid] = None

    # Optional radar cross-check (BEV strong-returns via radar_to_lidar).
    radar_loader = radar_R = None
    if args.radar:
        radar_loader, radar_R = _radar_setup(args, calib_raw)

    annotations = empty_annotations()
    objects = annotations["objects"]
    n_boxes = 0

    for pid, l_idx, c_idx, d_idx in plan:
        depth = depth_map.get(d_idx)
        boxes = []
        # Pass 1: collect detections + their depth-image boxes. Use the YOLO cache for this
        # frame if present (no camera / no live YOLO), else run YOLO live on the color frame.
        dets = []
        cached = yolo_cache.get(str(c_idx)) if yolo_cache is not None else None
        if cached is not None:
            for d in cached:
                if d["label"] in keep_names and float(d["score"]) >= args.conf:
                    xyxy = [float(v) for v in d["xyxy"]]
                    dets.append((xyxy, d["label"], float(d["score"]),
                                 _color_box_to_depth(xyxy, ccal, dcal)))
        else:
            color = color_map.get(c_idx)
            # depth is only needed for the D435 fallback; --lidar_only doesn't use it.
            if color is None or (depth is None and not args.lidar_only):
                print(f"  pair {pid}: missing frame; skipping.")
                continue
            # frames are rgb8; ultralytics predict on ndarray assumes BGR -> flip.
            res = model.predict(
                color[..., ::-1], conf=args.conf, classes=keep_ids, verbose=False
            )[0]
            for b in res.boxes:
                cls = int(b.cls[0])
                label = model.names.get(cls, str(cls))
                score = float(b.conf[0])
                xyxy = [float(v) for v in b.xyxy[0].tolist()]
                dets.append((xyxy, label, score, _color_box_to_depth(xyxy, ccal, dcal)))
        # Pair-level lidar point assignment: give each shared frustum point to a single
        # (smallest) detection so two close objects don't collapse onto one cluster.
        assigned = None
        if use_lidar and not args.no_split_shared:
            assigned = _assign_frustum_points(
                [d[3] for d in dets], lidar_by_pair.get(pid), lid2cam, dcal
            )
        # Pass 2: build a box per detection.
        for di, (xyxy, label, score, dbox) in enumerate(dets):
            box, box_source = None, None
            _csz = _CLASS_SIZE.get(label) if args.cover_class_size else None
            if use_lidar:
                _scene_yaw = np.radians(args.scene_yaw_deg)
                if assigned is not None:
                    box = _box_from_points(
                        assigned[di], min_pts=args.min_lidar_pts, gap_m=args.lidar_gap_m,
                        ground_pct=args.ground_pct, min_height_m=args.min_height_m,
                        scene_yaw=_scene_yaw,
                        sparsity_pad=args.box_sparsity_pad, sparsity_pad_max=args.box_sparsity_pad_max,
                        radar_cover_frac=args.box_radar_cover_frac, radar_cover_max=args.box_radar_cover_max,
                        height_pad=args.box_height_pad, class_size=_csz,
                    )
                else:
                    box = _lidar_frustum_box(
                        lidar_by_pair.get(pid), lid2cam, dcal, dbox,
                        min_pts=args.min_lidar_pts, gap_m=args.lidar_gap_m,
                        ground_pct=args.ground_pct, min_height_m=args.min_height_m,
                        scene_yaw=_scene_yaw,
                        sparsity_pad=args.box_sparsity_pad, sparsity_pad_max=args.box_sparsity_pad_max,
                        radar_cover_frac=args.box_radar_cover_frac, radar_cover_max=args.box_radar_cover_max,
                        height_pad=args.box_height_pad, class_size=_csz,
                    )
                if box is not None:
                    box_source = "yolo_lidar_frustum"
            if box is None and not args.lidar_only:
                pts = bbox_depth_to_sensor_points(depth, dbox, calib, bbox_format="xyxy")
                pts = _foreground_slab(pts, args.fg_band_m)
                box = sensor_points_to_topdown_box(
                    pts,
                    min_depth_m=args.min_depth_m,
                    max_depth_m=args.max_depth_m,
                    min_points=args.min_points,
                )
                box_source = "yolo_camera_projection"
            if box is None:
                continue
            b3 = box.get("box3d")
            if b3 is not None and label in _CLASS_SIZE:
                el, ew, eh = _CLASS_SIZE[label]
                # REJECT (drop) implausibly LARGE boxes -- a bad fit where the cluster merged
                # with a wall/neighbour (10 m "car", 4 m "person"). Check LENGTH + HEIGHT only;
                # WIDTH is intentionally range-padded for radar coverage so it's not a size cue.
                if args.max_size_factor > 0 and (b3["length"] > args.max_size_factor * el
                                                 or b3["height"] > args.max_size_factor * eh):
                    continue
                # class-size sanity: FLAG (not drop) dims outside 0.4x-2.5x.
                dims = ((b3["length"], el), (b3["width"], ew), (b3["height"], eh))
                plausible = all(0.4 * e <= d <= 2.5 * e for d, e in dims)
                b3["size_plausible"] = bool(plausible)
                b3["reliable"] = bool(b3.get("reliable", True) and plausible)
            boxes.append(
                {
                    "id": f"cam{c_idx:06d}_{len(boxes):03d}",
                    "label": label,
                    "lateral": box["lateral"],
                    "forward": box["forward"],
                    # camera 2D box (color pixels) persisted so the label editor can
                    # show/adjust it and re-project it to the lidar frame.
                    "bbox_xyxy": [float(v) for v in xyxy],
                    "color": "red" if box_source == "yolo_lidar_frustum" else "orange",
                    "score": score,
                    "source": box_source,
                    "camera_idx": c_idx,
                    # oriented 3D box (x,y,z,yaw,l,w,h); present for lidar-frustum boxes,
                    # None for the depth-fallback path.
                    "box3d": box.get("box3d"),
                    "n_depth_points": box["n_depth_points"],
                    "median_depth_m": box["median_depth_m"],
                }
            )
        # Radar returns for this pair (needed for both lidar proposals and confirmation).
        rf = rl = np.array([])
        if radar_loader is not None:
            try:
                bev, la, fa = _radar_panel(radar_loader, pid)
                rf, rl = _peaks_from_bev(bev, la, fa, radar_R, args.radar_percentile)
            except Exception as exc:  # noqa: BLE001
                print(f"  pair {pid}: radar load failed ({exc}).")
                rf = rl = np.array([])
        # Lidar-cluster proposals: cars OUTSIDE the camera FOV that YOLO can't see but lidar +
        # radar clearly do. Cluster the ground-removed cloud, keep vehicle-sized radar-confirmed
        # clusters that no YOLO box already covers, class-floor to a car.
        n_prop = 0
        if args.lidar_propose and radar_loader is not None:
            props = _lidar_box_proposals(
                lidar_by_pair.get(pid), rf, rl, boxes,
                ground_pct=args.ground_pct, min_height_m=args.min_height_m,
                scene_yaw=np.radians(args.scene_yaw_deg),
                radar_assoc_m=args.lidar_propose_radar_m,
                require_radar=args.lidar_propose_radar,
                min_pts=max(args.min_lidar_pts, args.lidar_propose_min_pts),
                class_size=_CLASS_SIZE.get("car"),
            )
            for pb in props:
                b3 = pb["box3d"]
                cen = np.array([b3["cx"], b3["cy"]])
                boxes.append({
                    "id": f"lid{pid:06d}_{len(boxes):03d}",
                    "label": "car",
                    "lateral": pb["lateral"],
                    "forward": pb["forward"],
                    "color": "cyan",
                    "score": None,
                    "source": "lidar_radar_proposal",
                    "box3d": b3,
                    "n_depth_points": pb["n_pts"],
                    "median_depth_m": float(np.hypot(cen[0], cen[1])),
                })
            n_prop = len(props)
        # gap-limit flooring (stop each box at the next car in front) then centroid-gated NMS
        # (drop only true duplicates, keep distinct adjacent cars).
        if args.gap_limit:
            boxes = _gap_limit_flooring(boxes, args.gap_limit_m)
        boxes = _nms_footprint(boxes, args.nms_iou, args.nms_dup_dist)
        n_radar = 0
        if radar_loader is not None:
            for bx in boxes:
                hits = _radar_hits(bx, rf, rl, args.radar_assoc_m)
                bx["radar_hits"] = hits
                bx["radar_confirmed"] = bool(hits > 0)
            n_radar = sum(1 for bx in boxes if bx["radar_confirmed"])

        if boxes:
            objects[str(pid)] = boxes
            n_boxes += len(boxes)
        n_lid = sum(1 for b in boxes if b["source"] == "yolo_lidar_frustum")
        prop_msg = f", {n_prop} lidar-proposed" if args.lidar_propose else ""
        radar_msg = f", {n_radar} radar-confirmed" if radar_loader is not None else ""
        print(
            f"  pair {pid}: YOLO {len(dets)} det(s) -> {len(boxes)} box(es) "
            f"({n_lid} lidar-frustum{prop_msg}, {len(boxes) - n_lid - n_prop} depth-fallback{radar_msg})"
        )

    out_json = args.output_annotations or str(
        Path(args.calibration_json).parent / "annotations" / "yolo.json"
    )
    save_annotations(annotations, out_json)
    print(f"\nWrote {n_boxes} boxes across {len(objects)} pair(s) -> {out_json}")
    print("Inspect with: viz_labels.py (camera + oriented box3d BEV; add --radar for the radar panel).")
    return 0


def _bev_window(boxes, pad_m: float, min_half: float, cap: float):
    """Auto-zoom BEV limits to the projected boxes (+pad), else a near-field default."""
    if not boxes:
        return (min_half, -min_half), (0.0, 2 * min_half)
    lat0 = min(b["lateral"][0] for b in boxes)
    lat1 = max(b["lateral"][1] for b in boxes)
    fwd1 = max(b["forward"][1] for b in boxes)
    latc = 0.5 * (lat0 + lat1)
    half = max(min_half, (lat1 - lat0) * 0.5 + pad_m, (fwd1 * 0.5) + pad_m)
    half = min(half, cap)
    fwd_hi = min(cap * 2, max(2 * min_half, fwd1 + pad_m))
    return (latc + half, latc - half), (0.0, fwd_hi)  # (xlim already flipped)


def _lidar_points_at(source, metadata, scan_idx, min_range_mm=500.0) -> np.ndarray:
    """Indexed (random-access) XYZ points for one scan, via lidar_io only.

    Reads the pcap directly (no bev_render import, which drags in radar deps) and uses
    the pcap index so a late scan streams only its chunk, not the whole prefix.
    """
    from lib.lidar_io import scan_range_flat, scan_to_xyz, _scan_from_scan_set

    scan = _scan_from_scan_set(source[int(scan_idx)])
    xyz = scan_to_xyz(metadata, scan).reshape(-1, 3)
    ranges = scan_range_flat(scan).reshape(-1)
    valid = np.isfinite(xyz).all(axis=1) & (ranges >= float(min_range_mm))
    return xyz[valid]



if __name__ == "__main__":
    raise SystemExit(main())
