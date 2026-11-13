"""Interactive slider editor for oriented box3d: center, size, heading -- live on the panels.

Loads ``annotations/yolo.json`` (from yolo_to_lidar.py) and, per sync pair, shows:
    [ lidar BEV: X-forward vs Y-lateral ]   [ elevation: Y-lateral vs Z-height ]
with sliders for the SELECTED box's ``cx, cy, cz, length, width, height, yaw``. Dragging a
slider updates the box on BOTH panels immediately. This is the numeric/precise complement to
``edit_labels.py`` (which is drag-based and has no height, since its BEV is top-down).

Saving writes ``box3d`` back AND recomputes the top-down footprint (``lateral``/``forward``)
from the box so downstream tools (radar association, K-Radar export) stay consistent.

Keys / buttons
--------------
  Next/Prev box    : cycle the boxes in this pair (selected box is highlighted)
  Next/Prev pair   : move between sync pairs
  Reset            : restore the selected box to its on-disk values
  Save             : write annotations back to the JSON

Run::
    python code/sync/src/camera_projection/adjust_box3d.py -d 2026.05.08/17-34-27 --pairs 5000 6000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import os as _os
import numpy as np
import matplotlib
# This is an INTERACTIVE tool -> needs a GUI backend, but many envs default to (or export)
# the non-GUI 'Agg' backend (then plt.show() does nothing). Since Agg is never useful here,
# force a GUI backend BEFORE importing pyplot whenever the requested backend is unset or Agg
# -- even if MPLBACKEND=Agg is exported (that is almost always leftover, not intentional here).
# A non-Agg MPLBACKEND (e.g. Qt5Agg) is respected. `--dry_render` opts out (stays Agg, headless).
# _GUI_BK = the GUI backend to (re)assert -- some shared imports (make_video does
# matplotlib.use('Agg') at import) clobber it later, so we re-apply it before making the figure.
_GUI_BK = None
_req = _os.environ.get("MPLBACKEND", "").lower()
if _req in ("", "agg") and "--dry_render" not in sys.argv:
    for _bk in ("macosx", "qtagg", "tkagg", "qt5agg", "gtk3agg"):
        try:
            matplotlib.use(_bk, force=True)
            _GUI_BK = _bk
            break
        except Exception:
            continue
elif _req:  # a non-Agg backend was explicitly requested -> keep it
    _GUI_BK = _req
import matplotlib.pyplot as plt


def _ensure_gui_backend():
    """Re-assert the GUI backend if a shared import flipped it to Agg (make_video does that)."""
    if _GUI_BK and "agg" in matplotlib.get_backend().lower():
        try:
            matplotlib.use(_GUI_BK, force=True)
        except Exception:
            pass
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Polygon, Rectangle

_THIS = Path(__file__).resolve().parent
for _p in (_THIS.parent, _THIS.parent.parent, _THIS / "calibration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.dataset_config import add_dataset_arguments, apply_dataset_config  # noqa: E402

# Slider (name, min, max) for each box3d field.
FIELDS = [
    ("cx", 0.0, 30.0), ("cy", -16.0, 16.0), ("cz", -2.0, 3.0),
    ("length", 0.2, 10.0), ("width", 0.2, 5.0), ("height", 0.2, 4.0),
    ("yaw_deg", -180.0, 180.0),
]


def _corners(o):
    """4 BEV corners (X-fwd, Y-lat) of an oriented box3d dict."""
    yaw = o["yaw"]
    hl, hw = o["length"] / 2.0, o["width"] / 2.0
    major = np.array([np.cos(yaw), np.sin(yaw)])
    minor = np.array([-np.sin(yaw), np.cos(yaw)])
    c = np.array([o["cx"], o["cy"]])
    return np.array([c + sl * hl * major + sw * hw * minor
                     for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", default=None, help="yolo.json (default res/<ds>/annotations/yolo.json).")
    p.add_argument("--pairs", type=int, nargs="+", default=None,
                   help="Pairs to edit (default: all pairs that have a box3d).")
    p.add_argument("--max_depth_m", type=float, default=25.0)
    p.add_argument("--dry_render", default=None,
                   help="Headless: render the first selected pair's panels to this PNG and exit "
                        "(no GUI window). For previews / CI.")
    p.add_argument("--camera_sync_csv", default=None)
    p.add_argument("--lidar_pcap", default=None)
    p.add_argument("--lidar_metadata", default=None)
    # camera panel (D435 color + the detection's 2D box, for context while editing)
    p.add_argument("--camera", action=argparse.BooleanOptionalAction, default=True,
                   help="Show a camera panel (D435 color + 2D YOLO box). The frame is fetched per "
                        "pair from the Drive bag (~slow for uncached frames); --no-camera skips it.")
    p.add_argument("--camera_bag", default=None)
    p.add_argument("--camera_index", default=None)
    p.add_argument("--frame_cache", default=None, help="Local dir for extracted camera frames.")
    # radar panel (align the box to radar returns while editing)
    p.add_argument("--radar", action=argparse.BooleanOptionalAction, default=True,
                   help="Show a radar-BEV panel (warped to the lidar frame) with the box overlaid. "
                        "--no-radar for a faster lidar-only editor.")
    p.add_argument("--calibration_json", default=None)
    p.add_argument("--radar_h5", default=None)
    p.add_argument("--cfg_file", default=None)
    p.add_argument("--sync_csv", default=None)
    p.add_argument("--sync_summary", default=None)
    p.add_argument("--radar_percentile", type=float, default=99.9)
    p.add_argument("--radar_bev_res", type=float, default=0.2)
    p.add_argument("--radar_assoc_m", type=float, default=1.5)
    p.add_argument("--radar_doppler_mode", default="all", choices=["all", "moving", "static"])
    p.add_argument("--radar_processing", default="legacy", choices=["legacy", "ti"])
    add_dataset_arguments(p)
    return p


class Editor:
    def __init__(self, args):
        self.args = args
        self.ann_path = Path(args.annotations or f"res/{args.dataset}/annotations/yolo.json")
        self.data = json.load(open(self.ann_path))
        self.objects = self.data["objects"]
        # pairs to edit = requested (that exist + have box3d) or all with a box3d
        want = [str(p) for p in args.pairs] if args.pairs else sorted(self.objects, key=int)
        self.pairs = [p for p in want if any(b.get("box3d") for b in self.objects.get(p, []))]
        if not self.pairs:
            raise SystemExit("No pairs with a box3d to edit.")
        self.pi = 0
        self.bi = 0
        self.dirty = False

        # lidar source (for BEV context points)
        import csv
        rows = {}
        with open(args.camera_sync_csv, newline="") as f:
            for r in csv.DictReader(f):
                rows[int(r["pair_idx"])] = r
        self.rows = rows
        from lib.lidar_io import open_pcap_scan_source, sensor_info_from_source
        self.lsrc = open_pcap_scan_source(args.lidar_pcap, args.lidar_metadata, index=True)
        self.lmeta = sensor_info_from_source(self.lsrc)
        self._pts_cache = {}

        # radar loader (optional) -- warped BEV panel to align the box to radar returns.
        self.radar_loader = self.radar_R = None
        self._radar_cache = {}
        if args.radar:
            try:
                calib = json.load(open(args.calibration_json))
                from yolo_to_lidar import _radar_setup
                self.radar_loader, self.radar_R = _radar_setup(
                    args, calib, doppler_mode=args.radar_doppler_mode,
                    processing=args.radar_processing)
            except Exception as exc:  # noqa: BLE001
                print(f"radar unavailable ({exc}); editing without the radar panel.")

        # camera frame cache (per-pair, fetched on demand from the Drive bag)
        self._cam_cache = {}
        self._cam_bag_ts = None
        self._cam_reader = self._cam_conns = None   # persistent bag reader (opened once)
        self._lid2cam = self._ccal = None
        if args.camera and args.camera_index and Path(args.camera_index).is_file():
            _idx = np.load(args.camera_index)
            self._cam_bag_ts = _idx["color_bag_ts_ns"] if "color_bag_ts_ns" in _idx else None
        if args.camera:
            # for projecting the edited 3D box onto the color image (moves with the sliders)
            try:
                from extrinsic_io import get_extrinsic
                _cal = json.load(open(args.calibration_json))
                _c2l, _ = get_extrinsic(_cal, "camera_to_lidar")
                self._lid2cam = np.linalg.inv(np.asarray(_c2l, float).reshape(4, 4))
                _ci = _cal["color_intrinsics"]
                self._ccal = (_ci["fx"], _ci["fy"], _ci["cx"], _ci["cy"])
            except Exception as exc:  # noqa: BLE001
                print(f"camera 3D-box projection unavailable ({exc}); showing 2D box only.")

        self._build_ui()
        self._load_pair()

    # ---- data helpers ----
    def _pair(self):
        return self.pairs[self.pi]

    def _boxes(self):
        return [b for b in self.objects.get(self._pair(), []) if b.get("box3d")]

    def _sel(self):
        bx = self._boxes()
        return bx[self.bi] if bx else None

    def _lidar_pts(self, pid):
        if pid in self._pts_cache:
            return self._pts_cache[pid]
        from yolo_to_lidar import _lidar_points_at
        try:
            pts = _lidar_points_at(self.lsrc, self.lmeta, int(self.rows[pid]["lidar_idx"]))
        except Exception:
            pts = np.zeros((0, 3))
        self._pts_cache[pid] = pts
        return pts

    def _radar(self, pid):
        """Cached (bev, la, fa, rf, rl) for a pair, or None. rf/rl = strong-return points."""
        if not self.radar_loader:
            return None
        if pid in self._radar_cache:
            return self._radar_cache[pid]
        from yolo_to_lidar import _radar_panel, _peaks_from_bev
        try:
            bev, la, fa = _radar_panel(self.radar_loader, pid)
            rf, rl = _peaks_from_bev(bev, la, fa, self.radar_R, self.args.radar_percentile)
            res = (bev, la, fa, rf, rl)
        except Exception as exc:  # noqa: BLE001
            print(f"  pair {pid}: radar load failed ({exc}).")
            res = None
        self._radar_cache[pid] = res
        return res

    def _camera(self, pid):
        """Cached D435 color frame for a pair (fetched on demand from the bag), or None."""
        if not self.args.camera:
            return None
        c_idx = int(self.rows[pid].get("camera_color_idx", self.rows[pid].get("camera_idx", -1)))
        if c_idx in self._cam_cache:
            return self._cam_cache[c_idx]
        from yolo_to_lidar import _frame_cache_dir
        f = _frame_cache_dir(self.args) / f"color_{c_idx:06d}.npy"
        img = None
        if f.is_file():
            img = np.load(f)                      # local .npy cache -> instant
        elif self._cam_bag_ts is not None and 0 <= c_idx < len(self._cam_bag_ts):
            try:
                # Open the bag reader ONCE (reads the ~114 GB bag's chunk index, ~40 s), then
                # SEEK per frame (fast). Re-opening per pair was the slowdown.
                if self._cam_reader is None:
                    from rosbags.rosbag1 import Reader
                    from lib.camera_io import DEFAULT_COLOR_TOPIC
                    print("opening camera bag (one-time index read, ~40 s)...", flush=True)
                    self._cam_reader = Reader(Path(self.args.camera_bag))
                    self._cam_reader.open()
                    self._cam_conns = [c for c in self._cam_reader.connections
                                       if c.topic == DEFAULT_COLOR_TOPIC]
                from lib.camera_io import parse_ros1_image
                ts = int(self._cam_bag_ts[c_idx])
                _c, _t, raw = next(self._cam_reader.messages(connections=self._cam_conns, start=ts))
                img = np.array(parse_ros1_image(raw).data, copy=True)
                try:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    np.save(f, img)               # persist so next session is instant
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:  # noqa: BLE001
                print(f"  pair {pid}: camera fetch failed ({exc}).")
        self._cam_cache[c_idx] = img
        return img

    def _project_box(self, o):
        """Project the box3d's 8 corners to color-image pixels. Returns (u, v, in_front)."""
        cor = _corners(o)                       # (4,2) BEV corners X,Y (lidar frame)
        zc, hh = o["cz"], o["height"] / 2.0
        P = np.array([[X, Y, z] for (X, Y) in cor for z in (zc - hh, zc + hh)])  # (8,3)
        Pc = P @ self._lid2cam[:3, :3].T + self._lid2cam[:3, 3]                  # camera frame
        z = np.clip(Pc[:, 2], 1e-3, None)
        u = self._ccal[0] * Pc[:, 0] / z + self._ccal[2]
        v = self._ccal[1] * Pc[:, 1] / z + self._ccal[3]
        return u, v, Pc[:, 2] > 0.1

    # ---- UI ----
    def _build_ui(self):
        _ensure_gui_backend()  # radar setup imported make_video, which flips backend to Agg
        # dynamic panel layout: [camera?] [BEV] [radar?] [elevation]
        panels = ((["camera"] if self.args.camera else [])
                  + ["bev"] + (["radar"] if self.radar_loader is not None else []) + ["elev"])
        n = len(panels)
        self.fig = plt.figure(figsize=(min(4.6 * n, 22), 8))
        self.ax_camera = self.ax_bev = self.ax_radar = self.ax_elev = None
        gap = 0.015
        w = (0.97 - gap * (n + 1)) / n
        attr = {"camera": "ax_camera", "bev": "ax_bev", "radar": "ax_radar", "elev": "ax_elev"}
        for i, name in enumerate(panels):
            ax = self.fig.add_axes([gap + i * (w + gap), 0.42, w, 0.52])
            setattr(self, attr[name], ax)
        self.sliders = {}
        self._step_btns = []
        # per-field nudge step for the -/+ buttons (fine control without dragging)
        self._step = {"cx": 0.05, "cy": 0.05, "cz": 0.05, "length": 0.05,
                      "width": 0.05, "height": 0.05, "yaw_deg": 1.0}
        base = 0.34
        for i, (name, lo, hi) in enumerate(FIELDS):
            y = base - i * 0.035
            self.sliders[name] = Slider(self.fig.add_axes([0.28, y, 0.40, 0.022]),
                                        name, lo, hi, valinit=0.0)
            self.sliders[name].on_changed(self._on_slider)
            # -/+ buttons to step the value left/right by self._step[name]
            bm = Button(self.fig.add_axes([0.80, y, 0.028, 0.022]), "−")
            bp = Button(self.fig.add_axes([0.832, y, 0.028, 0.022]), "+")
            st = self._step[name]
            bm.on_clicked(lambda e, nm=name, d=-st: self._nudge(nm, d))
            bp.on_clicked(lambda e, nm=name, d=st: self._nudge(nm, d))
            self._step_btns += [bm, bp]
        # buttons
        def btn(x, label, cb):
            b = Button(self.fig.add_axes([x, 0.03, 0.11, 0.05]), label)
            b.on_clicked(cb)
            return b
        self._btns = [
            btn(0.06, "Prev box", lambda e: self._cycle_box(-1)),
            btn(0.18, "Next box", lambda e: self._cycle_box(+1)),
            btn(0.34, "Prev pair", lambda e: self._cycle_pair(-1)),
            btn(0.46, "Next pair", lambda e: self._cycle_pair(+1)),
            btn(0.62, "Reset", self._reset_box),
            btn(0.74, "Save", self._save),
        ]
        self._orig = None  # on-disk snapshot of the selected box3d (for Reset)

    def _load_pair(self):
        self.bi = 0
        self._snapshot()
        self._sync_sliders()
        self._redraw()

    def _snapshot(self):
        o = self._sel()
        self._orig = dict(o["box3d"]) if o else None

    def _sync_sliders(self):
        o = self._sel()
        if not o:
            return
        b = o["box3d"]
        self._muted = True
        for name, _, _ in FIELDS:
            val = np.degrees(b["yaw"]) if name == "yaw_deg" else b.get(name, 0.0)
            self.sliders[name].set_val(float(val))
        self._muted = False

    def _nudge(self, name, delta):
        """Step a slider by ``delta`` (from the -/+ buttons); set_val fires _on_slider."""
        s = self.sliders[name]
        s.set_val(float(np.clip(s.val + delta, s.valmin, s.valmax)))

    def _on_slider(self, _v):
        if getattr(self, "_muted", False):
            return
        o = self._sel()
        if not o:
            return
        b = o["box3d"]
        for name, _, _ in FIELDS:
            v = float(self.sliders[name].val)
            if name == "yaw_deg":
                b["yaw"] = float(np.radians(v))
            else:
                b[name] = v
        self.dirty = True
        self._redraw()

    def _redraw(self):
        pid = int(self._pair())
        pts = self._lidar_pts(pid)
        sel = self._sel()
        cap = self.args.max_depth_m
        # CAMERA panel: D435 color + each detection's 2D box (context; box3d edits don't move it)
        if self.ax_camera is not None:
            ac = self.ax_camera
            ac.clear()
            img = self._camera(pid)
            if img is not None:
                ac.imshow(img)
                H, W = img.shape[:2]
                # faint gray dashed = original YOLO 2D box (static anchor)
                for b in self._boxes():
                    xy = b.get("bbox_xyxy")
                    if xy:
                        ac.add_patch(Rectangle((xy[0], xy[1]), xy[2] - xy[0], xy[3] - xy[1],
                                               fill=False, ec="0.5", lw=1, ls="--"))
                # projected 3D box (MOVES with the sliders): 12 cuboid edges
                if self._lid2cam is not None:
                    edges = [(0, 2), (2, 4), (4, 6), (6, 0), (1, 3), (3, 5), (5, 7), (7, 1),
                             (0, 1), (2, 3), (4, 5), (6, 7)]
                    for j, b in enumerate(self._boxes()):
                        o = b["box3d"]
                        u, v, fr = self._project_box(o)
                        is_sel = (j == self.bi)
                        ec = "red" if is_sel else ("lime" if o.get("reliable", True) else "orange")
                        for a2, c2 in edges:
                            if fr[a2] and fr[c2]:
                                ac.plot([u[a2], u[c2]], [v[a2], v[c2]], "-",
                                        color=ec, lw=2.0 if is_sel else 1.2)
                ac.set_xlim(0, W)
                ac.set_ylim(H, 0)
            else:
                ac.text(0.5, 0.5, "camera unavailable", ha="center", va="center",
                        transform=ac.transAxes)
            ac.set_title(f"pair {pid}  camera (gray=2D YOLO, color=3D box)")
            ac.axis("off")
        # BEV
        ax = self.ax_bev
        ax.clear()
        if pts.size:
            m = np.linalg.norm(pts[:, :2], axis=1) <= cap
            ax.scatter(pts[m, 1], pts[m, 0], s=2, c="0.7")
        for j, b in enumerate(self._boxes()):
            o = b["box3d"]
            cor = _corners(o)
            is_sel = (j == self.bi)
            ec = "red" if is_sel else ("lime" if o.get("reliable", True) else "orange")
            ax.add_patch(Polygon(cor[:, [1, 0]], closed=True, fill=False, ec=ec,
                                 lw=2.5 if is_sel else 1.5))
            front = 0.5 * (cor[0] + cor[1])
            ax.plot([o["cy"], front[1]], [o["cx"], front[0]], "-", color=ec, lw=1.5)
        ax.set_xlim(cap * 0.6, -cap * 0.6)
        ax.set_ylim(0, cap)
        ax.set_aspect("equal")
        ax.set_xlabel("Lateral Y (m)"); ax.set_ylabel("Forward X (m)")
        ax.set_title(f"pair {pid}  box {self.bi + 1}/{len(self._boxes())}  BEV (top-down)")
        # RADAR BEV (warped to lidar frame) + boxes -- align the box to the radar returns
        if self.ax_radar is not None:
            arad = self.ax_radar
            arad.clear()
            rd = self._radar(pid)
            LF = LL = bev = finite = None
            if rd is not None:
                bev, la, fa, rf, rl = rd
                RL, RF = np.meshgrid(la, fa)
                LF = self.radar_R[0, 0] * RF + self.radar_R[0, 1] * RL + self.radar_R[0, 3]
                LL = self.radar_R[1, 0] * RF + self.radar_R[1, 1] * RL + self.radar_R[1, 3]
                finite = np.isfinite(bev)
                arad.pcolormesh(LL, LF, bev, cmap="viridis", shading="gouraud")
                arad.scatter(rl, rf, s=14, c="0.35", marker="x", linewidths=0.9)  # global strong
            for j, b in enumerate(self._boxes()):
                o = b["box3d"]
                cor = _corners(o)
                is_sel = (j == self.bi)
                arad.add_patch(Polygon(cor[:, [1, 0]], closed=True, fill=False,
                                       ec="red" if is_sel else "white",
                                       lw=2.5 if is_sel else 1.5))
                # red = strongest radar cells STRICTLY inside the ORIENTED box (exact for any yaw)
                if finite is not None:
                    dx, dy = LF - o["cx"], LL - o["cy"]
                    cyaw, syaw = np.cos(o["yaw"]), np.sin(o["yaw"])
                    along = dx * cyaw + dy * syaw          # along box length (forward axis)
                    across = -dx * syaw + dy * cyaw        # across box width (lateral axis)
                    inb = (finite & (np.abs(along) <= o["length"] / 2.0)
                           & (np.abs(across) <= o["width"] / 2.0))
                    if inb.any():
                        s2 = inb & (bev >= bev[inb].max() - 6.0)
                        arad.scatter(LL[s2], LF[s2], s=18, c="red", marker="o",
                                     edgecolors="black", linewidths=0.3, zorder=5)
            arad.set_xlim(cap * 0.6, -cap * 0.6)
            arad.set_ylim(0, cap)
            arad.set_aspect("equal")
            arad.set_xlabel("Lateral Y (m)")
            arad.set_title(f"pair {pid}  radar BEV + points (red=in-box)")
        # ELEVATION (Y-lat vs Z-height) for the selected box
        ae = self.ax_elev
        ae.clear()
        if sel is not None:
            o = sel["box3d"]
            if pts.size:
                # CONTEXT (light gray): lidar in a forward slab + a wide lateral window --
                # includes neighbours/background at the box's range, NOT just its contents.
                slab = np.abs(pts[:, 0] - o["cx"]) <= max(o["length"] / 2.0 + 0.5, 1.0)
                near = slab & (np.abs(pts[:, 1] - o["cy"]) <= o["width"] / 2.0 + 3.0)
                ae.scatter(pts[near, 1], pts[near, 2], s=3, c="0.82")
                # IN-BOX (blue): lidar actually inside the box footprint -- what the box holds.
                inbox = (np.abs(pts[:, 0] - o["cx"]) <= o["length"] / 2.0) & \
                        (np.abs(pts[:, 1] - o["cy"]) <= o["width"] / 2.0)
                ae.scatter(pts[inbox, 1], pts[inbox, 2], s=7, c="tab:blue")
            zc, hh = o["cz"], o["height"] / 2.0
            yc, hw = o["cy"], o["width"] / 2.0
            ae.add_patch(Rectangle((yc - hw, zc - hh), 2 * hw, 2 * hh, fill=False, ec="red", lw=2.5))
            ae.plot([yc], [zc], "r+", ms=10)
            ae.set_xlim(yc + max(hw + 3, 4), yc - max(hw + 3, 4))
            ae.set_ylim(-2.2, max(zc + hh + 0.5, 3.0))
            ae.set_xlabel("Lateral Y (m)"); ae.set_ylabel("Height Z (m)")
            lab = sel.get("label", "?")
            ae.set_title(f"{lab}  L{o['length']:.2f} W{o['width']:.2f} H{o['height']:.2f} "
                         f"yaw{np.degrees(o['yaw']):+.0f}")
        self.fig.canvas.draw_idle()

    # ---- buttons ----
    def _cycle_box(self, d):
        n = len(self._boxes())
        if n:
            self.bi = (self.bi + d) % n
            self._snapshot(); self._sync_sliders(); self._redraw()

    def _cycle_pair(self, d):
        self.pi = (self.pi + d) % len(self.pairs)
        self._load_pair()

    def _reset_box(self, _e):
        o = self._sel()
        if o and self._orig:
            o["box3d"] = dict(self._orig)
            self._sync_sliders(); self._redraw()

    def _save(self, _e):
        # recompute the top-down footprint from each box3d so it stays consistent.
        for boxes in self.objects.values():
            for b in boxes:
                o = b.get("box3d")
                if not o:
                    continue
                cor = _corners(o)
                b["lateral"] = [float(cor[:, 1].min()), float(cor[:, 1].max())]
                b["forward"] = [float(max(0.0, cor[:, 0].min())), float(cor[:, 0].max())]
        bak = self.ann_path.with_suffix(".pre_manual_edit.bak.json")
        if not bak.exists():
            import shutil; shutil.copy(self.ann_path, bak)
        json.dump(self.data, open(self.ann_path, "w"), indent=1)
        self.dirty = False
        print(f"saved -> {self.ann_path} (backup {bak.name})")
        self.ax_bev.set_title(self.ax_bev.get_title() + "  [saved]")
        self.fig.canvas.draw_idle()

    def run(self):
        bk = matplotlib.get_backend()
        if "agg" in bk.lower():
            print(
                f"\n[!] matplotlib is using the non-GUI backend '{bk}', so the editor window "
                "cannot open.\n    Fix one of:\n"
                "      - unset the env var:   unset MPLBACKEND\n"
                "      - install a GUI backend: pip install PyQt5   (or conda install pyqt / tk)\n"
                "      - or force one:  MPLBACKEND=macosx python .../adjust_box3d.py ...\n"
            )
            return
        print(f"editor ready (backend {bk}). Adjust sliders; Save writes to the JSON.")
        plt.show()


def main():
    args = apply_dataset_config(build_argparser().parse_args())
    if args.dry_render:
        matplotlib.use("Agg", force=True)
        ed = Editor(args)
        ed.fig.savefig(args.dry_render, dpi=90)
        print(f"dry render -> {args.dry_render}")
        return 0
    Editor(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
