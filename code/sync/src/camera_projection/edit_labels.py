"""Manually adjust auto-generated labels in BOTH the camera image and the lidar BEV.

Loads an ``annotations/yolo.json`` (from yolo_to_lidar.py) and opens a two-panel
editor per sync pair:

    [ D435 color + 2D boxes ]      [ lidar BEV + top-down boxes ]

The camera 2D box (``bbox_xyxy``, color pixels) and the lidar box (``lateral`` /
``forward``, meters) are edited INDEPENDENTLY — drag to move a box, drag a corner to
resize — because the auto camera->lidar projection is sometimes wrong (depth/frustum
noise) and you must be able to fix the lidar box directly, not only the image box.
Press ``r`` to re-derive the selected box's lidar footprint from its (possibly
just-edited) camera box when you DO want them consistent.

Keys
----
  click        select the box under the cursor (smallest box wins when they overlap;
               click the SAME spot again to cycle to the box underneath)
  drag         move the selected box; drag a corner (cyan handles) to resize
  a            add a box: click-drag a new rectangle in either panel
  d            delete the selected box
  l            cycle the selected box's label (--classes order)
  r            re-project selected camera box -> lidar footprint (lidar-in-frustum)
  n / p        next / previous pair
  s            save annotations
  q            quit (prompts if unsaved)

Run::

    python code/sync/src/camera_projection/edit_labels.py \
        -d 2026.05.08/17-34-27 --annotations res/.../annotations/yolo.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SYNC_DIR = _THIS_DIR.parent
_CALIB_DIR = _SYNC_DIR / "calibration"
for _p in (_SYNC_DIR, _THIS_DIR, _CALIB_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402
from lib.camera_io import DEFAULT_COLOR_TOPIC  # noqa: E402
from annotations import empty_annotations, load_annotations, save_annotations  # noqa: E402
from yolo_to_lidar import (  # noqa: E402
    _bev_window,
    _color_box_to_depth,
    _extract_frames,
    _frame_cache_dir,
    _lidar_frustum_box,
    _lidar_points_at,
)

DEFAULT_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
CAM_TOL_PX = 15.0     # corner-grab tolerance, camera panel (pixels)
LID_TOL_M = 0.5       # corner-grab tolerance, lidar panel (meters)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--calibration_json", default=None)
    p.add_argument("--camera_bag", default=None)
    p.add_argument("--camera_index", default=None)
    p.add_argument("--camera_sync_csv", default=None)
    p.add_argument("--lidar_pcap", default=None)
    p.add_argument("--lidar_metadata", default=None)
    p.add_argument("--annotations", default=None,
                   help="Annotations JSON to edit (default: <calib dir>/annotations/yolo.json).")
    p.add_argument("--output_annotations", default=None,
                   help="Where to save (default: overwrite --annotations).")
    p.add_argument("--pairs", type=int, nargs="+", default=None,
                   help="Pairs to edit (default: those present in the annotations).")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                   help="Label set cycled by the 'l' key / used for new boxes.")
    p.add_argument("--max_depth_m", type=float, default=20.0)
    p.add_argument("--min_lidar_pts", type=int, default=6)
    p.add_argument("--lidar_gap_m", type=float, default=2.0)
    p.add_argument("--frame_cache", default=None)
    # radar panel (dense BEV heatmap, same loader as adjust_box3d.py / make_video RadarLoader)
    p.add_argument("--radar", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--radar_h5", default="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11/radar/2026.05.11-21.37.07.h5")
    p.add_argument("--cfg_file", default=None)   # resolved from common.CFG if None
    p.add_argument("--sync_csv", default="sync/res/2026.05.11/21.37.07/sync_pairs.csv")
    p.add_argument("--sync_summary", default="sync/res/2026.05.11/21.37.07/sync_summary.json")
    p.add_argument("--radar_percentile", type=float, default=99.9)
    p.add_argument("--radar_bev_res", type=float, default=0.2)
    p.add_argument("--radar_doppler_mode", default="all", choices=["all", "moving", "static"])
    p.add_argument("--radar_processing", default="legacy", choices=["legacy", "ti"])
    p.add_argument("--dry_render", default=None, metavar="OUT.png",
                   help="Headless: render the first pair to this PNG and exit (no GUI).")
    add_dataset_arguments(p)
    return p


def _load_pair_rows(csv_path: Path) -> Dict[int, dict]:
    import csv as _csv

    rows: Dict[int, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            rows[int(row["pair_idx"])] = row
    return rows


def reproject_camera_box(box: dict, lidar_pts, lid2cam, dcal, ccal, *,
                         min_pts: int, gap_m: float) -> bool:
    """Re-derive ``box`` lidar footprint from its camera ``bbox_xyxy`` (lidar-in-frustum).

    Returns True and updates lateral/forward in place on success; False if the box has
    no camera box or too few lidar points fall in the frustum.
    """
    xyxy = box.get("bbox_xyxy")
    if xyxy is None:
        return False
    dbox = _color_box_to_depth(xyxy, ccal, dcal)
    lb = _lidar_frustum_box(lidar_pts, lid2cam, dcal, dbox, min_pts=min_pts, gap_m=gap_m)
    if lb is None:
        return False
    box["lateral"] = lb["lateral"]
    box["forward"] = lb["forward"]
    if lb.get("box3d") is not None:
        box["box3d"] = lb["box3d"]
    box["n_depth_points"] = lb["n_depth_points"]
    box["median_depth_m"] = lb["median_depth_m"]
    box["source"] = "manual_reproject"
    return True


def _rect_of(box: dict, kind: str) -> Optional[List[float]]:
    """Return [xa, ya, xb, yb] (min<max) in the given panel's coords, or None."""
    if kind == "cam":
        xy = box.get("bbox_xyxy")
        if xy is None:
            return None
        x0, y0, x1, y1 = xy
    else:
        if "lateral" not in box or "forward" not in box:
            return None
        x0, x1 = box["lateral"]
        y0, y1 = box["forward"]
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _write_rect(box: dict, kind: str, r: List[float]) -> None:
    xa, ya, xb, yb = r
    if kind == "cam":
        box["bbox_xyxy"] = [min(xa, xb), min(ya, yb), max(xa, xb), max(yb, ya)]
    else:
        box["lateral"] = [min(xa, xb), max(xa, xb)]
        box["forward"] = [min(ya, yb), max(ya, yb)]


def _hit_handle(r: List[float], x: float, y: float, tol: float) -> Optional[str]:
    """Which part of rect r=[xa,ya,xb,yb] is at (x,y): a corner, 'move', or None.

    Corners get a ``tol`` grab radius; 'move' requires the click STRICTLY inside the
    box (no padding) so two adjacent boxes don't both claim the gap between them.
    """
    xa, ya, xb, yb = r
    corners = {"a": (xa, ya), "b": (xb, ya), "c": (xa, yb), "d": (xb, yb)}
    for name, (cx, cy) in corners.items():
        if abs(x - cx) <= tol and abs(y - cy) <= tol:
            return name
    if xa <= x <= xb and ya <= y <= yb:
        return "move"
    return None


def _candidate_key(r: List[float], handle: str, x: float, y: float):
    """Sort key so the best box under (x,y) wins: corners first (closest corner),
    then the SMALLEST containing box (tie-break: nearest center)."""
    xa, ya, xb, yb = r
    if handle in ("a", "b", "c", "d"):
        cx, cy = {"a": (xa, ya), "b": (xb, ya), "c": (xa, yb), "d": (xb, yb)}[handle]
        return (0, (x - cx) ** 2 + (y - cy) ** 2)
    area = abs((xb - xa) * (yb - ya))
    mx, my = 0.5 * (xa + xb), 0.5 * (ya + yb)
    return (1, area, (x - mx) ** 2 + (y - my) ** 2)


def _apply_drag(r0: List[float], handle: str, dx: float, dy: float) -> List[float]:
    """New rect from start rect r0 given the grabbed handle and mouse delta."""
    xa, ya, xb, yb = r0
    if handle == "move":
        return [xa + dx, ya + dy, xb + dx, yb + dy]
    if handle == "a":  # (xa,ya)
        return [xa + dx, ya + dy, xb, yb]
    if handle == "b":  # (xb,ya)
        return [xa, ya + dy, xb + dx, yb]
    if handle == "c":  # (xa,yb)
        return [xa + dx, ya, xb, yb + dy]
    if handle == "d":  # (xb,yb)
        return [xa, ya, xb + dx, yb + dy]
    return r0


class LabelEditor:
    def __init__(self, args, calib, objects, plan, color_map, lidar_source, lidar_meta):
        self.args = args
        self.objects = objects            # {str(pid): [box, ...]}
        self.plan = plan                  # [(pid, c_idx, l_idx), ...]
        self.color_map = color_map
        self.lidar_source = lidar_source
        self.lidar_meta = lidar_meta
        self.classes = args.classes
        self.lid2cam = np.linalg.inv(np.asarray(calib["camera_to_lidar"], float))
        di, ci = calib["intrinsics"], calib["color_intrinsics"]
        self.dcal = (di["fx"], di["fy"], di["cx"], di["cy"])
        self.ccal = (ci["fx"], ci["fy"], ci["cx"], ci["cy"])
        self.i = 0
        self.selected: Optional[int] = None
        self.dirty = False
        self._add_mode = False
        self._drag = None                 # {box, kind, handle, r0, x0, y0}
        self._pending_add = None          # {kind, x0, y0}
        self._last_click = None           # (kind, x, y) for click-again-to-cycle
        self._cycle = 0                   # index into overlapping candidates
        self._lidar_cache: Dict[int, np.ndarray] = {}
        self.fig = self.ax_cam = self.ax_lid = self.ax_radar = None
        # --- radar panel: reuse the dense-BEV-heatmap loader (same as adjust_box3d / make_video) ---
        self._radar_cache: Dict[int, object] = {}
        self.radar_loader = self.radar_R = None
        if getattr(args, "radar", True):
            try:
                if not args.cfg_file:
                    import sys as _s, os as _o
                    _s.path.insert(0, _o.path.join(_o.path.dirname(__file__), "..", "fusion", "radar_first"))
                    import common as _c            # noqa: E402
                    args.cfg_file = _c.CFG
                from yolo_to_lidar import _radar_setup  # noqa: E402
                self.radar_loader, self.radar_R = _radar_setup(
                    args, calib, doppler_mode=args.radar_doppler_mode, processing=args.radar_processing)
            except Exception as exc:  # noqa: BLE001
                print(f"radar panel unavailable ({exc}); editing without radar.")

    # --- data helpers ---
    def pid(self) -> int:
        return self.plan[self.i][0]

    def boxes(self) -> List[dict]:
        return self.objects.setdefault(str(self.pid()), [])

    def lidar_points(self, pid: int) -> np.ndarray:
        if pid not in self._lidar_cache:
            l_idx = dict((p, l) for p, _c, l in self.plan)[pid]
            try:
                self._lidar_cache[pid] = _lidar_points_at(self.lidar_source, self.lidar_meta, l_idx)
            except Exception:  # noqa: BLE001
                self._lidar_cache[pid] = np.empty((0, 3))
        return self._lidar_cache[pid]

    def radar_panel(self, pid: int):
        """Cached (bev_heatmap, lat_axis, fwd_axis, peak_fwd, peak_lat) for a pair, or None."""
        if not self.radar_loader:
            return None
        if pid in self._radar_cache:
            return self._radar_cache[pid]
        from yolo_to_lidar import _radar_panel, _peaks_from_bev  # noqa: E402
        try:
            bev, la, fa = _radar_panel(self.radar_loader, pid)
            rf, rl = _peaks_from_bev(bev, la, fa, self.radar_R, self.args.radar_percentile)
            res = (bev, la, fa, rf, rl)
        except Exception as exc:  # noqa: BLE001
            print(f"  pair {pid}: radar load failed ({exc}).")
            res = None
        self._radar_cache[pid] = res
        return res

    # --- drawing ---
    def draw(self) -> None:
        from matplotlib.patches import Rectangle

        pid, c_idx, l_idx = self.plan[self.i]
        boxes = self.boxes()
        self.ax_cam.clear()
        self.ax_lid.clear()
        self.ax_radar.clear()

        color = self.color_map.get(c_idx)
        if color is not None:
            self.ax_cam.imshow(color)
        self.ax_cam.set_title(f"pair {pid}: camera (2D boxes)  [{self.i+1}/{len(self.plan)}]")
        self.ax_cam.axis("off")

        pts = self.lidar_points(pid)
        if pts.size:
            m = np.linalg.norm(pts[:, :2], axis=1) <= self.args.max_depth_m
            self.ax_lid.scatter(pts[m, 1], pts[m, 0], s=2, c="0.6")
        rd = self.radar_panel(pid)        # dense radar BEV heatmap + strong-return peaks (lidar frame)
        if rd is not None:
            _bev, _la, _fa, rpf, rpl = rd
            self.ax_lid.scatter(rpl, rpf, s=16, marker="x", c="0.3", linewidths=1.0, zorder=2)  # radar peaks
        (xhi, xlo), (fy0, fy1) = _bev_window(
            [b for b in boxes if "lateral" in b], pad_m=1.0, min_half=3.0, cap=self.args.max_depth_m
        )
        self.ax_lid.set_xlim(xhi, xlo)
        self.ax_lid.set_ylim(fy0, fy1)
        self.ax_lid.set_aspect("equal")
        self.ax_lid.set_xlabel("Lateral Y (m)")
        self.ax_lid.set_ylabel("Forward X (m)")
        self.ax_lid.set_title(f"pair {pid}: lidar BEV (x = radar strong-return peaks)")

        # ---- RADAR panel: dense radar BEV heatmap warped to the lidar frame (editable) ----
        self.ax_radar.set_facecolor("0.08")
        if rd is not None:
            _bev, _la, _fa, rpf, rpl = rd
            RL, RF = np.meshgrid(_la, _fa)
            LF = self.radar_R[0, 0]*RF + self.radar_R[0, 1]*RL + self.radar_R[0, 3]
            LL = self.radar_R[1, 0]*RF + self.radar_R[1, 1]*RL + self.radar_R[1, 3]
            self.ax_radar.pcolormesh(LL, LF, _bev, cmap="viridis", shading="gouraud", zorder=1)
            self.ax_radar.scatter(rpl, rpf, s=12, c="white", marker="x", linewidths=0.8, zorder=3)  # strong returns
        self.ax_radar.set_xlim(xhi, xlo)
        self.ax_radar.set_ylim(fy0, fy1)
        self.ax_radar.set_aspect("equal")
        self.ax_radar.set_xlabel("Lateral Y (m)")
        self.ax_radar.set_title(f"pair {pid}: RADAR BEV heatmap (x=strong return) -- editable")

        # Draw unselected boxes first, then the selected one ON TOP (so an active box
        # is never hidden under an adjacent one's edges).
        order = [j for j in range(len(boxes)) if j != self.selected]
        if self.selected is not None and 0 <= self.selected < len(boxes):
            order.append(self.selected)
        for j in order:
            b = boxes[j]
            sel = (j == self.selected)
            ec = "cyan" if sel else str(b.get("color", "red"))
            lw = 3.0 if sel else 1.8
            rc = _rect_of(b, "cam")
            if rc is not None:
                self.ax_cam.add_patch(Rectangle((rc[0], rc[1]), rc[2] - rc[0], rc[3] - rc[1],
                                                fill=False, ec=ec, lw=lw, zorder=5 if sel else 3))
                self.ax_cam.text(rc[0], max(0, rc[1] - 4), b.get("label", ""), color=ec, fontsize=9)
                if sel:  # corner handles so resize targets are visible
                    self.ax_cam.scatter([rc[0], rc[2], rc[0], rc[2]], [rc[1], rc[1], rc[3], rc[3]],
                                        s=30, c="cyan", edgecolors="k", zorder=6)
            rl = _rect_of(b, "lid")
            if rl is not None:
                for axb in (self.ax_lid, self.ax_radar):   # same box geometry in both BEV panels
                    axb.add_patch(Rectangle((rl[0], rl[1]), rl[2] - rl[0], rl[3] - rl[1],
                                            fill=False, ec=ec, lw=lw, zorder=5 if sel else 3))
                    axb.text(rl[2], rl[3], b.get("label", ""), color=ec, fontsize=9)
                    if sel:
                        axb.scatter([rl[0], rl[2], rl[0], rl[2]], [rl[1], rl[1], rl[3], rl[3]],
                                    s=30, c="cyan", edgecolors="k", zorder=6)

        title = "  a=add  d=del  l=label  r=reproject  n/p=nav  s=save  q=quit"
        star = " *UNSAVED*" if self.dirty else ""
        if self._add_mode:
            title = "  ADD MODE: click-drag a new box in either panel" + star
        self.fig.suptitle(title + star, fontsize=10)
        if self.fig.canvas is not None:
            self.fig.canvas.draw_idle()

    # --- events ---
    def _axkind(self, ax) -> Optional[str]:
        if ax is self.ax_cam:
            return "cam"
        if ax is self.ax_lid or ax is self.ax_radar:   # radar panel edits the same BEV box geometry
            return "lid"
        return None

    def _tol(self, kind: str) -> float:
        return CAM_TOL_PX if kind == "cam" else LID_TOL_M

    def _candidates(self, kind: str, x: float, y: float):
        """All boxes under (x,y) in this panel, best-first (see _candidate_key)."""
        tol = self._tol(kind)
        out = []
        for j, b in enumerate(self.boxes()):
            r = _rect_of(b, kind)
            if r is None:
                continue
            h = _hit_handle(r, x, y, tol)
            if h is not None:
                out.append((j, h, _candidate_key(r, h, x, y)))
        out.sort(key=lambda t: t[2])
        return out

    def on_press(self, event):
        kind = self._axkind(event.inaxes)
        if kind is None or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if self._add_mode:
            self._pending_add = {"kind": kind, "x0": x, "y0": y}
            return
        cands = self._candidates(kind, x, y)
        if not cands:
            self.selected = None
            self._last_click = None
            self.draw()
            return
        # Click again on ~the same spot to cycle through overlapping boxes.
        eps = self._tol(kind) * 0.6
        same = (self._last_click is not None and self._last_click[0] == kind
                and abs(self._last_click[1] - x) < eps and abs(self._last_click[2] - y) < eps)
        self._cycle = (self._cycle + 1) % len(cands) if same else 0
        self._last_click = (kind, x, y)
        j, h, _ = cands[self._cycle]
        self.selected = j
        r = _rect_of(self.boxes()[j], kind)
        self._drag = {"box": j, "kind": kind, "handle": h, "r0": list(r), "x0": x, "y0": y}
        if len(cands) > 1:
            lbl = self.boxes()[j].get("label", "")
            print(f"  {len(cands)} boxes here — selected '{lbl}' [{self._cycle + 1}/{len(cands)}]; "
                  "click same spot again to cycle.")
        self.draw()

    def on_motion(self, event):
        if event.xdata is None or event.inaxes is None:
            return
        if self._drag is not None and self._axkind(event.inaxes) == self._drag["kind"]:
            dx = float(event.xdata) - self._drag["x0"]
            dy = float(event.ydata) - self._drag["y0"]
            r = _apply_drag(self._drag["r0"], self._drag["handle"], dx, dy)
            _write_rect(self.boxes()[self._drag["box"]], self._drag["kind"], r)
            self.dirty = True
            self.draw()

    def on_release(self, event):
        if self._pending_add is not None and event.xdata is not None:
            kind = self._pending_add["kind"]
            if self._axkind(event.inaxes) == kind:
                r = [self._pending_add["x0"], self._pending_add["y0"],
                     float(event.xdata), float(event.ydata)]
                if abs(r[2] - r[0]) > (2 if kind == "cam" else 0.2) and \
                   abs(r[3] - r[1]) > (2 if kind == "cam" else 0.2):
                    box = {"id": f"manual_{self.pid()}_{len(self.boxes())}",
                           "label": self.classes[0], "color": "lime", "source": "manual"}
                    _write_rect(box, kind, r)
                    self.boxes().append(box)
                    self.selected = len(self.boxes()) - 1
                    self.dirty = True
            self._pending_add = None
            self._add_mode = False
            self.draw()
            return
        if self._drag is not None:
            self._drag = None  # coords already normalized by _write_rect

    def on_key(self, event):
        key = (event.key or "").lower()
        if key == "a":
            self._add_mode = True
            self.draw()
        elif key == "d" and self.selected is not None:
            del self.boxes()[self.selected]
            self.selected = None
            self.dirty = True
            self.draw()
        elif key == "l" and self.selected is not None:
            b = self.boxes()[self.selected]
            cur = b.get("label", self.classes[0])
            nxt = self.classes[(self.classes.index(cur) + 1) % len(self.classes)] \
                if cur in self.classes else self.classes[0]
            b["label"] = nxt
            self.dirty = True
            self.draw()
        elif key == "r" and self.selected is not None:
            b = self.boxes()[self.selected]
            ok = reproject_camera_box(b, self.lidar_points(self.pid()), self.lid2cam,
                                      self.dcal, self.ccal,
                                      min_pts=self.args.min_lidar_pts, gap_m=self.args.lidar_gap_m)
            print("reprojected." if ok else "reproject failed (no camera box / too few lidar pts).")
            if ok:
                self.dirty = True
            self.draw()
        elif key in ("n", "p"):
            self.i = min(self.i + 1, len(self.plan) - 1) if key == "n" else max(self.i - 1, 0)
            self.selected = None
            self._add_mode = False
            self._last_click = None
            self.draw()
        elif key == "s":
            self.save()
        elif key == "q":
            if self.dirty:
                print("Unsaved changes — press 's' to save, or 'q' again to discard.")
                self.dirty = False  # arm discard on next q
                return
            import matplotlib.pyplot as plt
            plt.close(self.fig)

    def save(self) -> None:
        out = self.args.output_annotations or self.args.annotations
        payload = empty_annotations()
        dropped = 0
        for pid_key, boxes in self.objects.items():
            kept = []
            for b in boxes:
                if "lateral" not in b or "forward" not in b:
                    # camera-only box: try to reproject so it has a lidar footprint
                    ok = reproject_camera_box(b, self.lidar_points(int(pid_key)), self.lid2cam,
                                              self.dcal, self.ccal,
                                              min_pts=self.args.min_lidar_pts,
                                              gap_m=self.args.lidar_gap_m)
                    if not ok:
                        dropped += 1
                        continue
                kept.append(b)
            if kept:
                payload["objects"][pid_key] = kept
        save_annotations(payload, out)
        n = sum(len(v) for v in payload["objects"].values())
        msg = f"Saved {n} box(es) across {len(payload['objects'])} pair(s) -> {out}"
        if dropped:
            msg += f"  ({dropped} camera-only box(es) dropped: no lidar footprint)"
        print(msg)
        self.dirty = False
        self.draw()

    def connect(self) -> None:
        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("motion_notify_event", self.on_motion)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("key_press_event", self.on_key)


def _build_plan(args, objects) -> List:
    rows = _load_pair_rows(Path(args.camera_sync_csv))
    pids = args.pairs or sorted(int(k) for k in objects.keys())
    plan = []
    for pid in pids:
        r = rows.get(pid)
        if not r:
            print(f"  pair {pid} not in CSV; skipping.")
            continue
        c_idx = int(r.get("camera_color_idx", r.get("camera_idx", -1)))
        l_idx = int(r.get("lidar_idx", -1))
        if min(c_idx, l_idx) < 0:
            print(f"  pair {pid}: missing color/lidar idx; skipping.")
            continue
        plan.append((pid, c_idx, l_idx))
    return plan


def main() -> int:
    args = apply_dataset_config(build_argparser().parse_args())
    if not args.annotations:
        args.annotations = str(Path(args.calibration_json).parent / "annotations" / "yolo.json")

    import json
    with open(args.calibration_json, "r", encoding="utf-8") as f:
        calib = json.load(f)
    for k in ("camera_to_lidar", "intrinsics", "color_intrinsics"):
        if k not in calib:
            raise SystemExit(f"Calibration JSON missing '{k}'.")

    objects = load_annotations(args.annotations)["objects"]
    plan = _build_plan(args, objects)
    if not plan:
        raise SystemExit("No usable pairs to edit.")

    # frames (local cache; seek the bag only for misses)
    if args.dry_render:
        os.environ.setdefault("MPLBACKEND", "Agg")
    cache = _frame_cache_dir(args)
    bag_ts = None
    if args.camera_index and Path(args.camera_index).is_file():
        idx = np.load(args.camera_index)
        bag_ts = idx["color_bag_ts_ns"] if "color_bag_ts_ns" in idx else None
    color_map = _extract_frames(args.camera_bag, DEFAULT_COLOR_TOPIC,
                                sorted({c for _, c, _ in plan}), cache, "color", bag_ts)

    from lib.lidar_io import open_pcap_scan_source, sensor_info_from_source
    lsrc = open_pcap_scan_source(args.lidar_pcap, args.lidar_metadata, index=True)
    lmeta = sensor_info_from_source(lsrc)

    import matplotlib.pyplot as plt

    editor = LabelEditor(args, calib, objects, plan, color_map, lsrc, lmeta)
    if not args.dry_render:
        # Must run AFTER LabelEditor(): the radar loader imports make_video/bev_render,
        # which call matplotlib.use("Agg") and would clobber an earlier interactive backend.
        from interactive import ensure_interactive_backend
        ensure_interactive_backend()
    editor.fig, (editor.ax_cam, editor.ax_lid, editor.ax_radar) = plt.subplots(1, 3, figsize=(22, 7))
    editor.draw()

    if args.dry_render:
        editor.fig.savefig(args.dry_render, dpi=110, bbox_inches="tight")
        print(f"Rendered pair {plan[0][0]} -> {args.dry_render}")
        return 0

    editor.connect()
    print(__doc__.split("Keys")[1] if "Keys" in __doc__ else "")
    plt.show()
    if editor.dirty:
        print("Note: exited with unsaved changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
