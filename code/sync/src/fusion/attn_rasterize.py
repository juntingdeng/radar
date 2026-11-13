"""Rasterize real benchmark_v1 samples into BEV tensors for attn_fusion_proto, and cache them.

BEV grid (lidar frame): forward x in [0,FWD], lateral y in [-LAT,LAT], cell RES m.  row=forward, col=lateral.
  lidar_bev [3,H,W] = occupancy, density, max-height
  radar_bev [3,H,W] = occupancy, doppler(vel), power
  relfeat   [3,H,W] = lidar-density, radar-power, range
  target    [1,H,W] = car-center heatmap (3x3 blob)

  python3 sync/src/fusion/attn_rasterize.py <split> <frame> <frame> ...   # 05.11 frames -> cache .pt
"""
import os, sys, json
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "sync/res/benchmark_v1"))
sys.path.insert(0, os.path.join(REPO, "sync/src/fusion/radar_first"))

FWD, LAT, RES = 45.0, 24.0, 1.5
H, W = int(FWD / RES), int(2 * LAT / RES)                 # 30 x 32


def _cell(x_fwd, y_lat):
    r = np.clip((x_fwd / RES).astype(int), 0, H - 1)
    c = np.clip(((y_lat + LAT) / RES).astype(int), 0, W - 1)
    return r, c


def rasterize_lidar(pts):
    g = np.zeros((3, H, W), np.float32)
    m = (pts[:, 0] >= 0) & (pts[:, 0] < FWD) & (np.abs(pts[:, 1]) < LAT)
    p = pts[m]
    if len(p):
        r, c = _cell(p[:, 0], p[:, 1])
        np.add.at(g[1], (r, c), 1.0)                      # density (count)
        np.maximum.at(g[2], (r, c), np.clip((p[:, 2] + 0.5) / 2.5, 0, 1))  # max height
        g[0, r, c] = 1.0                                   # occupancy
    g[1] = np.clip(g[1] / 8.0, 0, 1)
    return g


def rasterize_radar(xy, vel, power):
    g = np.zeros((3, H, W), np.float32)
    m = (xy[:, 0] >= 0) & (xy[:, 0] < FWD) & (np.abs(xy[:, 1]) < LAT)
    xy, vel, power = xy[m], vel[m], power[m]
    if len(xy):
        r, c = _cell(xy[:, 0], xy[:, 1])
        g[0, r, c] = 1.0
        g[1, r, c] = np.clip(vel / 5.0, -1, 1)             # doppler (normalized ~max_doppler)
        pn = (power - power.min()) / (np.ptp(power) + 1e-9)
        g[2, r, c] = pn
    return g


def relfeat(lidar_bev, radar_bev):
    import torch.nn.functional as F
    ld = F.avg_pool2d(torch.tensor(lidar_bev[1:2])[None], 5, 1, 2)[0].numpy()
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(-1, 1, W), indexing="ij")
    rng = np.hypot(yy, xx)[None].astype(np.float32)
    return np.concatenate([ld, radar_bev[2:3], rng], 0)


def rasterize_target(boxes):
    import torch.nn.functional as F
    g = np.zeros((1, H, W), np.float32)
    for b in boxes:
        if b.get("label") != "car":
            continue
        xf = 0.5 * (b["forward"][0] + b["forward"][1]); yl = 0.5 * (b["lateral"][0] + b["lateral"][1])
        if not (0 <= xf < FWD and abs(yl) < LAT):
            continue
        r, c = _cell(np.array([xf]), np.array([yl]))
        g[0, r[0], c[0]] = 1.0
    return F.max_pool2d(torch.tensor(g)[None], 3, 1, 1)[0].numpy()


def build_cache(split, frames, out):
    from dataset import Benchmark
    from detector import RadarDetector
    ds = Benchmark(os.path.join(REPO, "sync/res/benchmark_v1"), split)
    by_id = {s["sample_id"]: s for s in ds.samples}
    L, Rr, Rf, T = [], [], [], []
    for n, f in enumerate(frames):
        sid = f"0511_{f:06d}"
        if sid not in by_id:
            continue
        s = ds[[i for i, x in enumerate(ds.samples) if x["sample_id"] == sid][0]]
        R2L = np.array(s["calib"]["radar_to_lidar_auto"])
        pts = ds.lidar(s)
        d = ds.radar(s); Xd, Yd = RadarDetector.to_xy(d)
        rf = R2L[0, 0] * Xd + R2L[0, 1] * Yd + R2L[0, 3]; rl = R2L[1, 0] * Xd + R2L[1, 1] * Yd + R2L[1, 3]
        lb = rasterize_lidar(pts)
        rb = rasterize_radar(np.c_[rf, rl], np.asarray(d["vel"], float), np.asarray(d["power"], float))
        L.append(lb); Rr.append(rb); Rf.append(relfeat(lb, rb)); T.append(rasterize_target(s["boxes"]))
        print(f"  {n+1}/{len(frames)} {sid}: lidar_occ {lb[0].sum():.0f}  radar_occ {rb[0].sum():.0f}  gt_cars {int(T[-1].sum()>0)}", flush=True)
    torch.save({k: torch.tensor(np.stack(v)) for k, v in [("lidar", L), ("radar", Rr), ("relf", Rf), ("target", T)]}, out)
    print(f"wrote {out}  ({len(L)} frames, grid {H}x{W})")


if __name__ == "__main__":
    split = sys.argv[1]; frames = [int(x) for x in sys.argv[2:]]
    build_cache(split, frames, os.path.join(REPO, "sync/res/benchmark_v1/bev_cache.pt"))
