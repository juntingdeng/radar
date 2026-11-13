import sys, os, csv, time
sys.path.insert(0,'sync/src')
import numpy as np
from lib.lidar_io import open_pcap_scan_source, sensor_info_from_source, scan_range_flat, scan_to_xyz, _scan_from_scan_set

# One-time: pre-extract the filtered XYZ for every lidar scan in the render range to
# small per-scan .npy (float16). ONE pcap open (~66s index build), then ~0.02s/scan.
# This lets the parallel render skip the 70GB pcap entirely (no per-process index
# rebuild, no concurrent-seek thrash) -- the thing that made multi-worker rendering
# stall. Mirrors _lidar_points_at exactly (min_range 500mm, finite xyz).
ROOT="/Users/juntingdeng/Library/CloudStorage/GoogleDrive-juntingd@andrew.cmu.edu/My Drive/CMU/Research/3DImage/sensor/TI/setup_test/code"
os.chdir(ROOT)
RES="sync/res/2026.05.11/21.37.07"
LOC="/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11/lidar"
PCAP=f"{LOC}/20260511_2137_35_OS-0-128_122519000479.pcap"
META=f"{LOC}/20260511_2137_35_OS-0-128_122519000479.json"
CSV=f"{RES}/camera_sync_pairs.csv"
OUT=os.path.expanduser("~/.cache/ti_setup_frames/lidar_scan_cache")
LO,HI=13000,30000
os.makedirs(OUT, exist_ok=True)

rows={int(r['pair_idx']):r for r in csv.DictReader(open(CSV))}
need=sorted({int(rows[p]['lidar_idx']) for p in range(LO,HI) if p in rows})
todo=[i for i in need if not os.path.exists(f"{OUT}/l_{i}.npy")]
print(f"lidar scans needed: {len(need)}  already cached: {len(need)-len(todo)}  to extract: {len(todo)}", flush=True)
if not todo:
    print("all cached; nothing to do", flush=True); sys.exit(0)

t0=time.time()
src=open_pcap_scan_source(PCAP, META, index=True)
meta=sensor_info_from_source(src)
print(f"pcap index built in {time.time()-t0:.0f}s; extracting...", flush=True)
t0=time.time(); done=0; fail=0
for i in todo:
    try:
        s=_scan_from_scan_set(src[int(i)])
        xyz=scan_to_xyz(meta,s).reshape(-1,3); rr=scan_range_flat(s).reshape(-1)
        v=np.isfinite(xyz).all(1)&(rr>=500.0)
        np.save(f"{OUT}/l_{i}.npy", xyz[v].astype(np.float16))
    except Exception as exc:
        fail+=1; print(f"  scan {i} FAILED: {exc}", flush=True); continue
    done+=1
    if done%1000==0:
        el=time.time()-t0; print(f"  {done}/{len(todo)}  {el:.0f}s  {el/done:.3f}s/scan  ETA {(len(todo)-done)*el/done/60:.1f}min", flush=True)
print(f"DONE: {done} scans cached, {fail} failed, {time.time()-t0:.0f}s total -> {OUT}", flush=True)
