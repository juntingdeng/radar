"""Read/assemble/write the per-collection calibration JSON.

- ``read_intrinsics_from_bag`` — pull camera intrinsics from a bag's camera_info.
- ``ensure_calibration_json`` — create the JSON (intrinsics + identity extrinsic)
  if missing, so the interactive calibrators never fail on a fresh collection.
- ``reuse_calibration`` — non-interactive: copy an already-solved extrinsic
  (+ intrinsics) from a reference collection onto another (same rig).
- ``write_extrinsic`` — merge one solved 4x4 into the JSON without clobbering it.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

_IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
_EXTRINSIC_KEYS = ("camera_to_lidar", "radar_to_lidar")


def read_intrinsics_from_bag(camera_bag: str | Path, depth_scale: float = 0.001) -> dict:
    """Return {intrinsics, [color_intrinsics], depth_scale} from bag camera_info."""
    from lib.camera_io import read_camera_infos

    print(f"Reading camera_info from {Path(camera_bag).name} (bag open may take a bit)...", flush=True)
    infos = read_camera_infos(camera_bag)
    if "depth" not in infos:
        raise SystemExit(
            f"No depth camera_info found in bag (streams: {sorted(infos)}). "
            "Pass --reuse_from a reference calibration instead."
        )
    payload = {"intrinsics": infos["depth"], "depth_scale": float(depth_scale)}
    if "color" in infos:
        payload["color_intrinsics"] = infos["color"]
    print(
        f"  depth intrinsics: fx={infos['depth']['fx']:.3f} cx={infos['depth']['cx']:.3f} "
        f"({infos['depth']['width']}x{infos['depth']['height']})"
    )
    return payload


def ensure_calibration_json(
    path: str | Path, *, camera_bag: Optional[str | Path] = None, depth_scale: float = 0.001
) -> None:
    """Create ``path`` (intrinsics from bag + identity extrinsic) if it is missing."""
    path = Path(path)
    if path.is_file():
        return
    if not camera_bag:
        raise SystemExit(f"{path} missing and no --camera_bag to read intrinsics from.")
    print(f"{path.name} not found — creating it from the camera bag intrinsics...")
    payload = read_intrinsics_from_bag(camera_bag, depth_scale)
    payload["camera_to_lidar"] = _IDENTITY
    payload["camera_to_lidar_meta"] = {"solved": False, "note": "IDENTITY placeholder — solve me"}
    payload["_comment"] = "Auto-created by a calibrator from bag camera_info; extrinsic solved on save."
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Created {path}")


def reuse_calibration(
    out_path: str | Path,
    from_calibration: str | Path,
    *,
    camera_bag: Optional[str | Path] = None,
    refresh_intrinsics: bool = False,
    depth_scale: float = 0.001,
    keys: Iterable[str] = _EXTRINSIC_KEYS,
    overwrite: bool = True,
) -> None:
    """Build ``out_path`` by copying solved extrinsic(s) (+intrinsics) from a reference.

    For the same physical rig across collections: intrinsics are copied by default
    (set ``refresh_intrinsics`` to re-read them from this collection's bag).
    """
    out_path = Path(out_path)
    if out_path.is_file() and not overwrite:
        raise SystemExit(f"{out_path} exists. Pass --overwrite to replace it.")
    with open(from_calibration, "r", encoding="utf-8") as f:
        ref = json.load(f)

    if refresh_intrinsics:
        payload = read_intrinsics_from_bag(camera_bag, depth_scale)
    else:
        payload = {k: ref[k] for k in ("intrinsics", "color_intrinsics", "depth_scale") if k in ref}
        print(f"Reusing intrinsics from {from_calibration}")

    carried = []
    for key in keys:
        if key in ref:
            payload[key] = ref[key]
            if f"{key}_meta" in ref:
                payload[f"{key}_meta"] = ref[f"{key}_meta"]
            carried.append(key)
            meta = ref.get(f"{key}_meta", {})
            if not meta.get("solved", False):
                print(f"  NOTE: {key} in the reference is still an unsolved placeholder.")
    if not carried:
        raise SystemExit(f"No extrinsic keys {tuple(keys)} found in {from_calibration}.")

    payload.setdefault(
        "_comment",
        "Assembled by reuse: intrinsics + extrinsic carried from a same-rig reference.",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Carried {', '.join(carried)} from reference -> {out_path}")


def write_extrinsic(
    calibration_json: str | Path,
    *,
    key: str,
    matrix: np.ndarray,
    n_points: int,
    rms_m: float,
    max_error_m: float,
    correspondences_path: Optional[str | Path] = None,
    note: str = "",
) -> None:
    """Merge a 4x4 extrinsic under ``key`` into an existing calibration JSON.

    Preserves every other field (intrinsics, depth_scale, other extrinsics) and
    records provenance under ``<key>_meta`` so a placeholder is never silently
    trusted again.
    """
    path = Path(calibration_json)
    payload: dict = {}
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            payload = {}

    payload[key] = np.asarray(matrix, dtype=np.float64).tolist()
    payload[f"{key}_meta"] = {
        "solved": True,
        "method": "hand-picked correspondences + Umeyama rigid fit",
        "n_correspondences": int(n_points),
        "rms_m": float(rms_m),
        "max_error_m": float(max_error_m),
        "correspondences": str(correspondences_path) if correspondences_path else None,
        "written": _dt.datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {key} (+{key}_meta) to {path}")
