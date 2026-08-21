"""Named dataset profiles for sync scripts (short --dataset instead of long paths)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# This module lives in code/sync/src/lib/; datasets.json and res/ live in code/sync/.
_SYNC_DIR = Path(__file__).resolve().parents[2]
_CODE_ROOT = _SYNC_DIR.parent

# argparse dest -> datasets.json key
DATASET_FIELDS = (
    "radar_h5",
    "lidar_pcap",
    "lidar_metadata",
    "camera_bag",
    "camera_index",
    "camera_sync_csv",
    "calibration_json",
    "annotations_path",
    "cfg_file",
    "sync_csv",
    "sync_summary",
    "out_video",
    "output_csv",
    "output_json",
    "cache_dir",
)

_CAPTURE_CONTAINER_KEYS = frozenset({"captures", "sessions", "sets"})

# Output paths in datasets.json are relative to this sync/ directory (not code/).
_SYNC_RELATIVE_FIELDS = frozenset(
    {
        "sync_csv",
        "sync_summary",
        "output_csv",
        "output_json",
        "out_video",
        "cache_dir",
        "camera_index",
        "camera_sync_csv",
        "calibration_json",
        "annotations_path",
    }
)


def default_datasets_path() -> Path:
    return _SYNC_DIR / "datasets.json"


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        metavar="NAME",
        help="Use paths from datasets.json (e.g. 2026.05.10/18-05-08). "
        "Explicit --radar_h5 / --lidar_pcap still win if passed on CLI.",
    )
    parser.add_argument(
        "--datasets",
        default=str(default_datasets_path()),
        help="Path to datasets.json config file.",
    )
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="Print configured dataset names and exit.",
    )


def _cli_flag_was_set(flag: str) -> bool:
    for arg in sys.argv[1:]:
        if arg == flag or arg.startswith(flag + "="):
            return True
    return False


def resolve_dataset_path(
    value: str,
    *,
    field: Optional[str] = None,
    data_root: Optional[str] = None,
) -> Path:
    """Resolve one datasets.json path (sync-relative, code-relative, or absolute)."""
    if data_root:
        value = value.replace("{data_root}", data_root.rstrip("/"))
    p = Path(value).expanduser()
    if p.is_absolute():
        return p.resolve()
    # Legacy configs used sync/res/... from code/; strip redundant prefix.
    rel = str(p)
    if rel.startswith("sync/"):
        rel = rel[len("sync/") :]
    sync_relative = field in _SYNC_RELATIVE_FIELDS or rel.startswith("res/")
    base = _SYNC_DIR if sync_relative else _CODE_ROOT
    return (base / rel).resolve()


def _resolve_path(
    value: str,
    *,
    data_root: Optional[str] = None,
    field: Optional[str] = None,
) -> str:
    return str(resolve_dataset_path(value, field=field, data_root=data_root))


def load_dataset_paths(name: str, config_path: str | Path | None = None) -> dict:
    """Resolve all paths for a dataset entry (notebooks / scripts)."""
    config_path = config_path or default_datasets_path()
    entry = get_dataset_entry(name, config_path)
    data_root = entry.get("data_root")
    if data_root is not None:
        data_root = str(resolve_dataset_path(str(data_root)))
    paths: dict = {}
    for key in DATASET_FIELDS:
        if key not in entry or entry[key] is None:
            continue
        paths[key] = str(
            resolve_dataset_path(str(entry[key]), field=key, data_root=data_root)
        )
    return paths


def load_datasets_config(path: str | Path) -> dict:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Datasets config not found: {path}\n"
            f"Copy datasets.example.json -> datasets.json and edit paths."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "datasets" not in data:
        raise ValueError(f"Invalid datasets config (missing 'datasets'): {path}")
    return data


def _split_dataset_name(name: str, known_dates: Iterable[str]) -> Tuple[str, Optional[str]]:
    """Return (date_key, capture_key) from NAME, NAME/capture, or NAME.capture."""
    if "/" in name:
        date_key, capture_key = name.split("/", 1)
        return date_key, capture_key or None

    known = set(known_dates)
    if name in known:
        return name, None

    # e.g. 2026.05.10.18-05-08 (avoid splitting 2026.05.10 on its own dots)
    for date_key in sorted(known, key=len, reverse=True):
        prefix = date_key + "."
        if name.startswith(prefix):
            return date_key, name[len(prefix) :]

    return name, None


def _capture_container(day_entry: dict) -> Optional[dict]:
    for key in _CAPTURE_CONTAINER_KEYS:
        captures = day_entry.get(key)
        if isinstance(captures, dict):
            return captures
    return None


def _merge_day_and_capture(day_entry: dict, capture_entry: dict) -> dict:
    merged = dict(day_entry)
    for key in _CAPTURE_CONTAINER_KEYS:
        merged.pop(key, None)
    merged.pop("description", None)
    merged.update(capture_entry)
    return merged


def _is_flat_capture_entry(entry: dict) -> bool:
    """True when the top-level entry is a single capture (legacy layout)."""
    return _capture_container(entry) is None and any(
        k in entry for k in ("radar_h5", "lidar_pcap", "lidar_metadata")
    )


def iter_named_datasets(path: str | Path) -> List[Tuple[str, dict]]:
    """Return [(full_name, merged_entry), ...] for --list_datasets and lookup."""
    data = load_datasets_config(path)
    named: List[Tuple[str, dict]] = []

    for date_key, day_entry in data["datasets"].items():
        if not isinstance(day_entry, dict):
            continue

        captures = _capture_container(day_entry)
        if captures is not None:
            for capture_key, capture_entry in captures.items():
                if not isinstance(capture_entry, dict):
                    continue
                full_name = f"{date_key}/{capture_key}"
                named.append((full_name, _merge_day_and_capture(day_entry, capture_entry)))
            continue

        if _is_flat_capture_entry(day_entry):
            named.append((date_key, dict(day_entry)))

    return sorted(named, key=lambda item: item[0])


def list_dataset_names(path: str | Path) -> List[str]:
    return [name for name, _ in iter_named_datasets(path)]


def get_dataset_entry(name: str, path: str | Path) -> dict:
    data = load_datasets_config(path)
    datasets = data["datasets"]
    date_key, capture_key = _split_dataset_name(name, datasets.keys())

    if date_key not in datasets:
        names = ", ".join(list_dataset_names(path)) or "(none)"
        raise KeyError(f"Unknown dataset {name!r}. Available: {names}")

    day_entry = datasets[date_key]
    if not isinstance(day_entry, dict):
        raise KeyError(f"Invalid dataset entry for {date_key!r}")

    captures = _capture_container(day_entry)
    if captures is not None:
        if capture_key is None:
            if len(captures) == 1:
                capture_key = next(iter(captures))
            else:
                options = ", ".join(f"{date_key}/{k}" for k in sorted(captures))
                raise KeyError(
                    f"Dataset {date_key!r} has multiple captures. "
                    f"Use one of: {options}"
                )
        if capture_key not in captures:
            options = ", ".join(f"{date_key}/{k}" for k in sorted(captures))
            raise KeyError(
                f"Unknown capture {capture_key!r} under {date_key!r}. "
                f"Available: {options}"
            )
        entry = _merge_day_and_capture(day_entry, captures[capture_key])
    else:
        if capture_key is not None:
            raise KeyError(
                f"Dataset {date_key!r} is a single capture (legacy layout); "
                f"use --dataset {date_key!r} without a suffix."
            )
        entry = dict(day_entry)

    entry.pop("description", None)
    return entry


def apply_dataset_config(
    args: argparse.Namespace,
    *,
    required: Optional[Iterable[str]] = None,
) -> argparse.Namespace:
    """Fill args from --dataset; CLI flags for the same field take precedence."""
    if getattr(args, "list_datasets", False):
        print(f"Datasets in {args.datasets}:")
        data = load_datasets_config(args.datasets)
        for full_name, entry in iter_named_datasets(args.datasets):
            desc = entry.get("description", "")
            if not desc:
                date_key, capture_key = _split_dataset_name(
                    full_name, data["datasets"].keys()
                )
                day_desc = data["datasets"].get(date_key, {}).get("description", "")
                if day_desc and capture_key:
                    desc = day_desc
            suffix = f" — {desc}" if desc else ""
            print(f"  {full_name}{suffix}")
        raise SystemExit(0)

    if not args.dataset:
        missing = [
            f
            for f in (required or ())
            if not getattr(args, f, None)
        ]
        if missing:
            raise SystemExit(
                f"Missing required argument(s): {', '.join('--' + m for m in missing)}. "
                "Pass paths explicitly or use --dataset NAME (see --list_datasets)."
            )
        return args

    entry = get_dataset_entry(args.dataset, args.datasets)
    data_root = entry.pop("data_root", None)
    if data_root is not None:
        data_root = _resolve_path(str(data_root))

    flag_map = {
        "radar_h5": "--radar_h5",
        "lidar_pcap": "--lidar_pcap",
        "lidar_metadata": "--lidar_metadata",
        "camera_bag": "--camera_bag",
        "camera_index": "--camera_index",
        "cfg_file": "--cfg_file",
        "sync_csv": "--sync_csv",
        "sync_summary": "--sync_summary",
        "out_video": "--out_video",
        "output_csv": "--output_csv",
        "output_json": "--output_json",
        "cache_dir": "--cache_dir",
    }

    applied: List[str] = []
    for key in DATASET_FIELDS:
        if key not in entry or entry[key] is None:
            continue
        flag = flag_map.get(key)
        if flag and _cli_flag_was_set(flag):
            continue
        if not hasattr(args, key):
            continue
        resolved = _resolve_path(str(entry[key]), data_root=data_root, field=key)
        setattr(args, key, resolved)
        applied.append(key)

    if applied:
        print(f"Dataset {args.dataset!r}: using {', '.join(applied)} from {args.datasets}")

    still_missing = [
        f for f in (required or ()) if not getattr(args, f, None)
    ]
    if still_missing:
        raise SystemExit(
            f"Dataset {args.dataset!r} did not set: {', '.join(still_missing)}. "
            f"Add them to datasets.json or pass on CLI."
        )
    return args
