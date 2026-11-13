"""Shared infrastructure for the radar-first proposal build.

Centralizes paths, the sync-pair table, and the lidar/radar loaders so the phase
modules (odometry, ego-Doppler, tracker, cross-check) don't each re-glue the IO.
"""
import os, sys, csv, functools
import numpy as np

ROOT = "/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT)
sys.path.insert(0, "sync/src")   # lib.*
sys.path.insert(0, ROOT)          # utils.* (repo root)

LOCAL = "/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11"
RES   = "sync/res/2026.05.11/21.37.07"
PCAP  = f"{LOCAL}/lidar/20260511_2137_35_OS-0-128_122519000479.pcap"
META  = f"{LOCAL}/lidar/20260511_2137_35_OS-0-128_122519000479.json"
H5    = f"{LOCAL}/radar/2026.05.11-21.37.07.h5"
CFG   = f"{ROOT}/mmWaveStudio/mobile_collection/rng40_res0004_tx3_rx4.lua"
CSV   = f"{RES}/camera_sync_pairs.csv"
CALIB = f"{RES}/calibration.json"
OUTDIR = f"{RES}/radar_first"
os.makedirs(OUTDIR, exist_ok=True)


def load_pairs():
    """pair_idx -> dict(lidar_idx, radar_idx, color_idx, depth_idx, radar_t)."""
    out = {}
    for r in csv.DictReader(open(CSV)):
        try:
            out[int(r["pair_idx"])] = dict(
                lidar_idx=int(r["lidar_idx"]), radar_idx=int(r["radar_idx"]),
                color_idx=int(r["camera_color_idx"]), depth_idx=int(r["camera_depth_idx"]),
                radar_t=float(r["radar_t"]))
        except (KeyError, ValueError):
            pass
    return out


class LidarSource:
    """Opens the 70GB pcap once (indexed), returns filtered XYZ per scan index."""
    def __init__(self, min_range_mm=500.0):
        from lib.lidar_io import (open_pcap_scan_source, sensor_info_from_source)
        self.src = open_pcap_scan_source(PCAP, META, index=True)
        self.meta = sensor_info_from_source(self.src)
        self.min_range_mm = float(min_range_mm)

    def xyz(self, scan_idx):
        from lib.lidar_io import scan_range_flat, scan_to_xyz, _scan_from_scan_set
        scan = _scan_from_scan_set(self.src[int(scan_idx)])
        xyz = scan_to_xyz(self.meta, scan).reshape(-1, 3)
        rng = scan_range_flat(scan).reshape(-1)
        v = np.isfinite(xyz).all(1) & (rng >= self.min_range_mm)
        return xyz[v]


@functools.lru_cache(maxsize=1)
def radar_config():
    from utils.parse_config import radarConfig
    import io, contextlib
    r = radarConfig()
    with contextlib.redirect_stdout(io.StringIO()):
        r.parse_radar(cfg_file=CFG)
    return r


def load_calib():
    import json
    return json.load(open(CALIB))


def pose_apply(T, xy_or_xyz):
    """Apply a 4x4 pose to Nx2 (z=0) or Nx3 points -> Nx(same)."""
    a = np.asarray(xy_or_xyz, dtype=np.float64)
    if a.shape[1] == 2:
        h = np.c_[a[:, 0], a[:, 1], np.zeros(len(a)), np.ones(len(a))]
        return (T @ h.T).T[:, :2]
    h = np.c_[a, np.ones(len(a))]
    return (T @ h.T).T[:, :3]
