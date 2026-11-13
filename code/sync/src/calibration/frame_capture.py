"""Save annotated calibration frames (with the picked boxes/marks) + make a video.

Both calibrators draw their picks straight onto the live matplotlib figure, so a
snapshot of that figure IS the labeled frame. ``LabeledFrameRecorder`` grabs one
PNG per sync pair (re-written whenever a pick is added/undone, so the file always
holds every mark made on that pair) and, when picking ends, stitches the PNGs
into a short video in pick order — a quick record of what was actually matched.

Frames are saved with a FIXED figure size/dpi so every PNG has the same pixel
size (ffmpeg needs that); if the user resizes the window mid-session the frames
are padded/cropped to the first frame's size before encoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class LabeledFrameRecorder:
    """Write one labeled PNG per pair, then encode them into a video."""

    def __init__(self, out_dir, *, dpi: int = 110, enabled: bool = True):
        self.enabled = bool(enabled)
        self.out_dir = Path(out_dir)
        self.dpi = int(dpi)
        self._paths: Dict[int, Path] = {}  # pair_idx -> png (insertion = pick order)
        if self.enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- capture ---------------------------------------------------------
    def capture(self, fig, pair_idx: Optional[int]) -> Optional[Path]:
        """Snapshot ``fig`` as the labeled frame for ``pair_idx`` (overwrites)."""
        if not self.enabled or fig is None:
            return None
        tag = "unknown" if pair_idx is None else f"{int(pair_idx):06d}"
        out = self.out_dir / f"pair{tag}_labeled.png"
        try:
            # No bbox_inches="tight": that crops per-frame and breaks a fixed
            # frame size, which the encoder needs.
            fig.savefig(out, dpi=self.dpi)
        except Exception as exc:  # noqa: BLE001 - never let saving kill the GUI
            print(f"  WARNING: could not save labeled frame ({exc})")
            return None
        first = pair_idx not in self._paths
        self._paths[pair_idx] = out
        if first:
            print(f"  labeled frame -> {out}")
        return out

    def frames(self) -> List[Path]:
        return [p for p in self._paths.values() if p.is_file()]

    def __len__(self) -> int:
        return len(self._paths)

    # -- video -----------------------------------------------------------
    def write_video(self, out_path=None, fps: float = 5.0) -> Optional[Path]:
        """Encode the captured PNGs into a video (mp4, GIF fallback)."""
        pngs = self.frames()
        if not self.enabled or not pngs:
            return None
        out_path = Path(out_path or (self.out_dir / "labeled_frames.mp4"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return frames_to_video(pngs, out_path, fps=fps)


def _match_size(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Pad (white) or crop ``img`` to exactly h x w so all frames match."""
    if img.shape[0] == h and img.shape[1] == w:
        return img
    img = img[:h, :w]
    pad_h, pad_w = h - img.shape[0], w - img.shape[1]
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=1.0)
    return img


def frames_to_video(pngs, out_path, fps: float = 5.0) -> Optional[Path]:
    """Stitch PNG paths into a video at ``fps`` (mp4 via ffmpeg, GIF fallback)."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    out_path = Path(out_path)
    imgs = []
    for p in pngs:
        try:
            im = plt.imread(str(p))
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: skipping unreadable frame {p} ({exc})")
            continue
        if im.ndim == 2:
            im = np.dstack([im] * 3)
        imgs.append(im[:, :, :3])
    if not imgs:
        return None

    h, w = imgs[0].shape[:2]
    imgs = [_match_size(im, h, w) for im in imgs]

    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    handle = ax.imshow(imgs[0])

    def update(i: int):
        handle.set_data(imgs[i])
        return [handle]

    ani = animation.FuncAnimation(
        fig, update, frames=len(imgs), interval=1000 / max(0.1, fps), blit=False
    )
    print(f"Encoding {len(imgs)} labeled frame(s) -> {out_path} @ {fps} fps ...", flush=True)
    if out_path.suffix.lower() == ".gif":  # asked for a GIF -> pillow, not ffmpeg
        ani.save(str(out_path), writer="pillow", fps=max(0.1, fps), dpi=dpi)
        plt.close(fig)
        print(f"Labeled-frame video: {out_path}")
        return out_path
    try:
        ani.save(str(out_path), writer="ffmpeg", fps=max(0.1, fps), dpi=dpi)
    except Exception as exc:  # noqa: BLE001 - ffmpeg missing/unusable
        gif = out_path.with_suffix(".gif")
        print(f"  ffmpeg unavailable ({exc}); writing {gif.name} instead.")
        try:
            ani.save(str(gif), writer="pillow", fps=max(0.1, fps), dpi=dpi)
            out_path = gif
        except Exception as exc2:  # noqa: BLE001
            print(f"  WARNING: could not write a video ({exc2}); PNGs are still on disk.")
            plt.close(fig)
            return None
    plt.close(fig)
    print(f"Labeled-frame video: {out_path}")
    return out_path


def add_label_arguments(parser, *, group_title: str = "labeled frames"):
    """Shared CLI flags for both calibrators."""
    g = parser.add_argument_group(group_title)
    g.add_argument(
        "--label_dir",
        default=None,
        help="Where to save the labeled frames (with picked boxes/marks) and the "
        "video. Default: <calibration_json dir>/calibration_labels/<tool>.",
    )
    g.add_argument(
        "--label_video",
        default=None,
        help="Output video path for the labeled frames (default: <label_dir>/labeled_frames.mp4).",
    )
    g.add_argument("--label_fps", type=float, default=5.0, help="Labeled-frame video fps.")
    g.add_argument(
        "--no_label_frames",
        action="store_true",
        help="Don't save labeled frames / build the video.",
    )
    return parser
