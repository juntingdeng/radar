"""Shared matplotlib click-picking for extrinsic calibration tools.

``DualPanelPicker`` shows a source panel and a target panel side by side and
captures alternating clicks: one click in the source panel, then one in the
target panel, records the resulting correspondence, and repeats. It is agnostic
to what a "point" is — the source/target extractor callbacks return the actual
sensor-frame coordinates (3D for camera->lidar, 2D for radar->lidar), so the
same picker drives both calibrators.

Keys:
  u  undo the last recorded correspondence (or a pending half-pick)
  n  advance to the next frame/pair (if the tool supplied more than one)
  s  solve + save now
  q  quit (also saves correspondences)

This module needs an interactive matplotlib backend, so it is not exercised by
the headless self-tests; the solver and IO layers are tested separately.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from correspondences import CorrespondenceSet


def ensure_interactive_backend() -> None:
    """Select a GUI matplotlib backend so the picker window is clickable.

    Must be called BEFORE ``import matplotlib.pyplot``. Raises SystemExit with
    guidance if only a non-interactive backend is available (e.g. Agg, or the
    VSCode/Jupyter inline renderer, which produce static, non-clickable images).
    """
    import matplotlib

    cur = matplotlib.get_backend().lower()
    if any(k in cur for k in ("macosx", "qt", "tk", "gtk", "wx")):
        return
    for cand in ("macosx", "qtagg", "qt5agg", "tkagg"):
        try:
            matplotlib.use(cand, force=True)
            print(f"Using interactive matplotlib backend: {cand}")
            return
        except Exception:
            continue
    raise SystemExit(
        f"No interactive matplotlib backend available (current: {cur!r}).\n"
        "The picker needs a clickable GUI window. Fixes:\n"
        "  - Run in a normal terminal (NOT the VSCode 'Run in Interactive Window' /\n"
        "    Jupyter, which render static images):\n"
        "      MPLBACKEND=macosx python <this_script>.py ...\n"
        "  - Or install a GUI backend:  pip install pyqt5"
    )


def bev_view_limits(pts, max_range: float, *, min_span: float = 4.0, pad: float = 0.15):
    """Robust per-frame BEV limits ((lat_lo, lat_hi), (fwd_lo, fwd_hi)) from XYZ.

    Zooms to where the returns actually are (indoor -> small, outdoor -> large),
    clipped to +/- max_range and forward >= 0. ``pts`` columns are (X=fwd, Y=lat, Z).
    """
    if pts is None or pts.size == 0:
        return (-max_range, max_range), (0.0, max_range)
    lat, fwd = pts[:, 1], pts[:, 0]
    la, lb = np.percentile(lat, [1, 99])
    fa, fb = np.percentile(fwd, [1, 99])
    lw = max(float(lb - la), min_span)
    fw = max(float(fb - fa), min_span)
    lc = 0.5 * (la + lb)
    lat_lim = (lc - lw * (0.5 + pad), lc + lw * (0.5 + pad))
    fwd_lim = (max(0.0, float(fa) - fw * pad), float(fb) + fw * pad)
    lat_lim = (max(-max_range, lat_lim[0]), min(max_range, lat_lim[1]))
    fwd_lim = (max(0.0, fwd_lim[0]), min(max_range, fwd_lim[1]))
    return lat_lim, fwd_lim


class DualPanelPicker:
    def __init__(
        self,
        fig,
        ax_source,
        ax_target,
        *,
        extract_source: Callable[[float, float], Optional[np.ndarray]],
        extract_target: Callable[[float, float], Optional[np.ndarray]],
        corr_set: CorrespondenceSet,
        current_pair_idx: Callable[[], Optional[int]],
        advance_frame: Optional[Callable[[], bool]] = None,
        on_solve: Optional[Callable[[], None]] = None,
    ):
        self.fig = fig
        self.ax_source = ax_source
        self.ax_target = ax_target
        self.extract_source = extract_source
        self.extract_target = extract_target
        self.corr_set = corr_set
        self.current_pair_idx = current_pair_idx
        self.advance_frame = advance_frame
        self.on_solve = on_solve

        self._pending_source: Optional[np.ndarray] = None
        self._artists: List = []
        self._status = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=9)
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._refresh_status()

    # -- event handlers --------------------------------------------------
    def _on_click(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        if event.inaxes is self.ax_source and self._pending_source is None:
            pt = self.extract_source(event.xdata, event.ydata)
            if pt is None:
                self._flash("No valid source point there (missing depth/return). Try again.")
                return
            self._pending_source = np.asarray(pt, dtype=np.float64)
            m = self.ax_source.plot(event.xdata, event.ydata, "x", color="red", ms=10, mew=2)[0]
            self._artists.append(m)
            self.fig.canvas.draw_idle()
            self._refresh_status()
        elif event.inaxes is self.ax_target and self._pending_source is not None:
            pt = self.extract_target(event.xdata, event.ydata)
            if pt is None:
                self._flash("No valid target point there. Try again.")
                return
            self.corr_set.add(
                self._pending_source, pt, pair_idx=self.current_pair_idx()
            )
            m = self.ax_target.plot(event.xdata, event.ydata, "x", color="red", ms=10, mew=2)[0]
            self._artists.append(m)
            self._pending_source = None
            self.fig.canvas.draw_idle()
            self._refresh_status()

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == "u":
            self._undo()
        elif key == "n" and self.advance_frame is not None:
            self._pending_source = None
            if self.advance_frame():
                self._clear_artists()
            self._refresh_status()
        elif key == "s" and self.on_solve is not None:
            self.on_solve()
        elif key in ("q", "escape"):
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    # -- helpers ---------------------------------------------------------
    def _undo(self) -> None:
        if self._pending_source is not None:
            self._pending_source = None
            if self._artists:
                self._artists.pop().remove()
        elif self.corr_set.points:
            self.corr_set.points.pop()
            # remove the two markers of the completed pair
            for _ in range(2):
                if self._artists:
                    self._artists.pop().remove()
        self.fig.canvas.draw_idle()
        self._refresh_status()

    def _clear_artists(self) -> None:
        for a in self._artists:
            try:
                a.remove()
            except ValueError:
                pass
        self._artists = []
        self.fig.canvas.draw_idle()

    def _flash(self, msg: str) -> None:
        self._status.set_text(msg)
        self.fig.canvas.draw_idle()

    def _refresh_status(self) -> None:
        pending = "  [SOURCE picked — now click TARGET]" if self._pending_source is not None else ""
        pid = self.current_pair_idx()
        pid_txt = f"pair={pid}  " if pid is not None else ""
        self._status.set_text(
            f"{pid_txt}correspondences={len(self.corr_set)}{pending}   "
            "keys: [click src→tgt]  u=undo  n=next  s=solve  q=quit"
        )
        self.fig.canvas.draw_idle()


class BoxPairPicker:
    """Click two opposite corners to box an object in each panel (with rubber-band).

    Unlike a drag-based selector (which conflicts with the toolbar zoom on some
    backends), this reuses plain click events — the same path the point picker
    uses reliably. Click corner 1 then corner 2 in the SOURCE panel, then the two
    corners in the TARGET panel; the correspondence is each box's centroid via the
    extractor callbacks (source/target return the sensor-frame point).
    """

    def __init__(
        self,
        fig,
        ax_source,
        ax_target,
        *,
        extract_source_box: Callable[[float, float, float, float], Optional[np.ndarray]],
        extract_target_box: Callable[[float, float, float, float], Optional[np.ndarray]],
        corr_set: CorrespondenceSet,
        current_pair_idx: Callable[[], Optional[int]],
        advance_frame: Optional[Callable[[], bool]] = None,
        on_solve: Optional[Callable[[], None]] = None,
    ):
        self.fig = fig
        self.ax_source = ax_source
        self.ax_target = ax_target
        self.extract_source_box = extract_source_box
        self.extract_target_box = extract_target_box
        self.corr_set = corr_set
        self.current_pair_idx = current_pair_idx
        self.advance_frame = advance_frame
        self.on_solve = on_solve

        self._stage = 0  # 0=src corner1, 1=src corner2, 2=tgt corner1, 3=tgt corner2
        self._c1: Optional[Tuple[float, float]] = None
        self._pending_source: Optional[np.ndarray] = None
        self._preview = None
        self._artists: List = []
        self._status = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=9)
        fig.canvas.mpl_connect("button_press_event", self._on_click)
        fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._refresh_status()

    def _expected_ax(self):
        return self.ax_source if self._stage in (0, 1) else self.ax_target

    def _rect(self, ax, x0, x1, y0, y1, color="red", ls="-"):
        from matplotlib.patches import Rectangle

        p = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=1.5, ls=ls)
        ax.add_patch(p)
        return p

    def _clear_preview(self):
        if self._preview is not None:
            try:
                self._preview.remove()
            except ValueError:
                pass
            self._preview = None

    def _on_move(self, event):
        if self._c1 is None or event.xdata is None or event.inaxes is not self._expected_ax():
            return
        self._clear_preview()
        x0, x1 = sorted((self._c1[0], event.xdata))
        y0, y1 = sorted((self._c1[1], event.ydata))
        self._preview = self._rect(event.inaxes, x0, x1, y0, y1, color="yellow", ls="--")
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.xdata is None or event.inaxes is not self._expected_ax():
            return
        if self._stage in (0, 2):  # first corner
            self._c1 = (float(event.xdata), float(event.ydata))
            self._stage += 1
            self._refresh_status()
            return
        # second corner -> form box
        x0, x1 = sorted((self._c1[0], float(event.xdata)))
        y0, y1 = sorted((self._c1[1], float(event.ydata)))
        self._clear_preview()
        ax = event.inaxes
        if self._stage == 1:  # source box complete
            pt = self.extract_source_box(x0, x1, y0, y1)
            if pt is None:
                self._flash("No valid points in that box — try a box on the object.")
                self._stage, self._c1 = 0, None
                self._refresh_status()
                return
            self._pending_source = np.asarray(pt, dtype=np.float64)
            self._artists.append(self._rect(ax, x0, x1, y0, y1))
            self._stage, self._c1 = 2, None
        else:  # target box complete
            pt = self.extract_target_box(x0, x1, y0, y1)
            if pt is None:
                self._flash("No valid points in that box.")
                self._stage, self._c1 = 2, None
                self._refresh_status()
                return
            self.corr_set.add(self._pending_source, pt, pair_idx=self.current_pair_idx())
            self._artists.append(self._rect(ax, x0, x1, y0, y1))
            self._pending_source, self._stage, self._c1 = None, 0, None
        self.fig.canvas.draw_idle()
        self._refresh_status()

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key == "u":
            self._undo()
        elif key == "n" and self.advance_frame is not None:
            self._reset_state()
            self.advance_frame()  # redraws; ax.clear() drops the box patches
            self._refresh_status()
        elif key == "s" and self.on_solve is not None:
            self.on_solve()
        elif key in ("q", "escape"):
            import matplotlib.pyplot as plt

            plt.close(self.fig)

    def _reset_state(self):
        self._clear_preview()
        self._stage, self._c1, self._pending_source = 0, None, None
        self._artists = []

    def _undo(self):
        self._clear_preview()
        if self._c1 is not None or self._pending_source is not None:
            # cancel the in-progress box(es)
            if self._pending_source is not None and self._artists:
                self._artists.pop().remove()
            self._stage, self._c1, self._pending_source = 0, None, None
        elif self.corr_set.points:
            self.corr_set.points.pop()
            for _ in range(2):  # remove the source+target rectangles
                if self._artists:
                    self._artists.pop().remove()
        self.fig.canvas.draw_idle()
        self._refresh_status()

    def _flash(self, msg: str):
        self._status.set_text(msg)
        self.fig.canvas.draw_idle()

    def _refresh_status(self):
        prompt = {
            0: "click box CORNER 1 in the SOURCE (left) panel",
            1: "click box CORNER 2 in the SOURCE panel",
            2: "click box CORNER 1 in the TARGET (right) panel",
            3: "click box CORNER 2 in the TARGET panel",
        }[self._stage]
        pid = self.current_pair_idx()
        pid_txt = f"pair={pid}  " if pid is not None else ""
        self._status.set_text(
            f"{pid_txt}correspondences={len(self.corr_set)}  BOX: {prompt}   "
            "u=undo  n=next  s=solve  q=quit"
        )
        self.fig.canvas.draw_idle()
