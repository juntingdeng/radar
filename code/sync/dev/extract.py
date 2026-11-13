from __future__ import annotations
"""Extraction utilities (merged). Subcommands: multiview_cam | timestamps | precache_lidar"""
import sys

# ======== dev/extract_multiview_cam.py ========
# Multi-view dataset v1 -- CAMERA-ONLY extraction for the 12 selected 0508 windows.
# 
# Windows = 4 visually-approved (aspect + motion picks) + 8 top census tiles (350-pid tiles over the event bands,
# scored by vehicle-aspect event count + type diversity + close-range bonus, non-overlapping).
# 
# For every pid in a window, copies the already-extracted colour jpg + aligned depth png into a self-contained
# local dataset (NOT Google Drive), named by pid:
#     ~/CMU/Research/mobile_data_local/multiview_v1/0508/win_<pid0>-<pid1>/{color,depth}/p<pid>.<ext>
# and writes manifests that make the later lidar/radar extraction a pure lookup:
#     per-window manifest.csv   pid, lidar_idx, radar_idx, camera_color_idx, camera_depth_idx, lidar_t, radar_t, camera_t
#     dataset-root manifest.json  windows + census events inside each + extraction status {cam: done, lidar/radar: pending}
# (a copy of manifest.json also lands in late_fusion/ for the pipeline side)
# Run: python sync/dev/extract_multiview_cam.py
def multiview_cam_main():
    import os, csv, json, shutil, collections
    CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RES = os.path.join(CODE, "sync", "res", "2026.05.08", "17-34-27")
    CAM = os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.08/camera_extracted")
    OUT = os.path.expanduser("~/CMU/Research/mobile_data_local/multiview_v1/0508")
    CENSUS = os.path.join(CODE, "sync", "src", "fusion", "late_fusion", "analysis", "aspect_census_0508.json")
    APPROVED = [(6545, 6890, "aspect: front+back of parked vehicles (ego passes around)"),
                (14365, 14710, "aspect: parking-lot sweep, 92 deg ego turn"),
                (11670, 11910, "lateral pass baseline (single-aspect control)"),
                (4350, 4590, "patio->plaza approach, movers")]
    TILE = 350; NWIN = 12
    TILE = 350; NWIN = 12
    def main():
        pairs = {}
        with open(os.path.join(RES, "camera_sync_pairs.csv")) as fh:
            for r in csv.DictReader(fh):
                pairs[int(r["pair_idx"])] = r
        events = json.load(open(CENSUS))

        # ---- choose 8 census tiles, non-overlapping with approved + each other ----
        chosen = [dict(pid0=a, pid1=b, why=w) for a, b, w in APPROVED]
        tiles = []
        pmax = max(pairs)
        for a in range(2000, pmax - TILE, TILE // 2):
            b = a + TILE
            ev = [e for e in events if e["pid0"] < b and e["pid1"] > a]
            if not ev: continue
            kinds = set(k for e in ev for k in e["kind"].split("+"))
            close = sum(1 for e in ev if e.get("rng_min", 99) < 8)
            tiles.append(dict(pid0=a, pid1=b, score=len(ev) + 2 * len(kinds) + 2 * close,
                              why=f"census: {len(ev)} events, kinds={sorted(kinds)}, close={close}"))
        tiles.sort(key=lambda t: -t["score"])
        for t in tiles:
            if len(chosen) >= NWIN: break
            if all(t["pid0"] >= c["pid1"] + 50 or t["pid1"] <= c["pid0"] - 50 for c in chosen):
                chosen.append(dict(pid0=t["pid0"], pid1=t["pid1"], why=t["why"]))
        chosen.sort(key=lambda c: c["pid0"])

        # ---- extract camera frames + per-window manifest ----
        man = dict(seq="0508", capture="2026.05.08/17-34-27", status=dict(cam="done", lidar="pending", radar="pending"),
                   windows=[])
        tot = 0; bytes_ = 0
        for c in chosen:
            wdir = os.path.join(OUT, f"win_{c['pid0']}-{c['pid1']}")
            os.makedirs(os.path.join(wdir, "color"), exist_ok=True)
            os.makedirs(os.path.join(wdir, "depth"), exist_ok=True)
            rows = []
            for p in range(c["pid0"], c["pid1"] + 1):
                r = pairs.get(p)
                if r is None: continue
                ci, di = int(r["camera_color_idx"]), int(r["camera_depth_idx"])
                srcC = None
                for ext in ("jpg", "png", "jpeg"):
                    q = os.path.join(CAM, "color", f"{ci:06d}.{ext}")
                    if os.path.exists(q): srcC = q; break
                srcD = os.path.join(CAM, "depth", f"{di:06d}.png")
                if srcC is None or not os.path.exists(srcD): continue
                dstC = os.path.join(wdir, "color", f"p{p:06d}" + os.path.splitext(srcC)[1])
                dstD = os.path.join(wdir, "depth", f"p{p:06d}.png")
                if not os.path.exists(dstC): shutil.copy2(srcC, dstC)
                if not os.path.exists(dstD): shutil.copy2(srcD, dstD)
                bytes_ += os.path.getsize(dstC) + os.path.getsize(dstD); tot += 1
                rows.append(dict(pid=p, lidar_idx=int(r["lidar_idx"]), radar_idx=int(r["radar_idx"]),
                                 camera_color_idx=ci, camera_depth_idx=di,
                                 lidar_t=r["lidar_t"], radar_t=r["radar_t"], camera_t=r["camera_t"]))
            with open(os.path.join(wdir, "manifest.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
            ev = [e for e in events if e["pid0"] < c["pid1"] and e["pid1"] > c["pid0"]]
            man["windows"].append(dict(pid0=c["pid0"], pid1=c["pid1"], frames=len(rows), why=c["why"],
                                       dir=wdir, lidar_idx=[r["lidar_idx"] for r in rows],
                                       radar_idx=[r["radar_idx"] for r in rows], events=ev))
            print(f"win {c['pid0']}-{c['pid1']}: {len(rows)} frames  [{c['why']}]", flush=True)
        for dst in (os.path.join(OUT, "manifest.json"),
                    os.path.join(CODE, "sync", "src", "fusion", "late_fusion", "analysis", "multiview_v1_manifest.json")):
            json.dump(man, open(dst, "w"), indent=1)
        print(f"\nDONE: {len(man['windows'])} windows, {tot} frames (colour+depth), {bytes_/1e9:.2f} GB -> {OUT}")
        print("manifest copies:", os.path.join(OUT, "manifest.json"), "and late_fusion/multiview_v1_manifest.json")
    if True:
        main()

# ======== dev/extract_timestamps.py ========
# Extract radar/lidar timestamps for quick sanity checks.
def timestamps_main():
    import argparse
    import numpy as np
    import sys
    from pathlib import Path
    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from lib.dataset_config import add_dataset_arguments, apply_dataset_config
    from lib.sync_utils import (
        lidar_packets_to_frames,
        radar_packets_to_frames,
        read_lidar_packet_timestamps_from_pcap,
        read_radar_packet_timestamps,
        try_read_ouster_scan_timestamps,
    )
    def build_argparser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Extract timestamp arrays from radar/lidar captures.")
        p.add_argument("--radar_h5", default=None)
        p.add_argument("--lidar_pcap", default=None)
        p.add_argument("--lidar_metadata", default=None)
        p.add_argument("--radar_packets_per_frame", type=int, default=128)
        p.add_argument("--lidar_packets_per_frame", type=int, default=128)
        p.add_argument("--lidar_udp_port", type=int, default=7502)
        p.add_argument(
            "--output_npz",
            default="./code/sync/timestamps.npz",
            help="Output npz containing packet/frame timestamps.",
        )
        add_dataset_arguments(p)
        return p
    def main() -> None:
        args = apply_dataset_config(
            build_argparser().parse_args(),
            required=("radar_h5", "lidar_pcap"),
        )

        radar_packet_ts = read_radar_packet_timestamps(args.radar_h5)
        radar_frame_ts = radar_packets_to_frames(
            radar_packet_ts, packets_per_frame=args.radar_packets_per_frame
        )

        lidar_frame_ts_sdk = try_read_ouster_scan_timestamps(args.lidar_pcap, args.lidar_metadata)
        using_sdk = lidar_frame_ts_sdk is not None
        if using_sdk:
            lidar_packet_ts = np.empty((0,), dtype=np.float64)
            lidar_frame_ts = lidar_frame_ts_sdk
        else:
            udp_port = None if args.lidar_udp_port < 0 else args.lidar_udp_port
            lidar_packet_ts = read_lidar_packet_timestamps_from_pcap(
                args.lidar_pcap, udp_port=udp_port
            )
            lidar_frame_ts = lidar_packets_to_frames(
                lidar_packet_ts, packets_per_frame=args.lidar_packets_per_frame
            )

        np.savez(
            args.output_npz,
            radar_packet_t=radar_packet_ts,
            radar_frame_t=radar_frame_ts,
            lidar_packet_t=lidar_packet_ts,
            lidar_frame_t=lidar_frame_ts,
            lidar_from_sdk=np.array([int(using_sdk)], dtype=np.int32),
        )

        print(f"Radar packets: {radar_packet_ts.size}")
        print(f"Radar frames: {radar_frame_ts.size}")
        print(f"Lidar packets: {lidar_packet_ts.size}")
        print(f"Lidar frames: {lidar_frame_ts.size}")
        print(f"Lidar timestamps source: {'Ouster SDK' if using_sdk else 'PCAP fallback'}")
        print(f"Wrote: {args.output_npz}")
    if True:
        main()

# ======== dev/precache_lidar.py ========
def precache_lidar_main():
    import sys, os, csv, time
    sys.path.insert(0,'sync/src')
    import numpy as np
    from lib.lidar_io import open_pcap_scan_source, sensor_info_from_source, scan_range_flat, scan_to_xyz, _scan_from_scan_set
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
    t0=time.time(); done=0; fail=0
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

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'multiview_cam': multiview_cam_main, 'timestamps': timestamps_main, 'precache_lidar': precache_lidar_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
