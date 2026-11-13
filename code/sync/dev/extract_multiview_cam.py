"""Multi-view dataset v1 -- CAMERA-ONLY extraction for the 12 selected 0508 windows.

Windows = 4 visually-approved (aspect + motion picks) + 8 top census tiles (350-pid tiles over the event bands,
scored by vehicle-aspect event count + type diversity + close-range bonus, non-overlapping).

For every pid in a window, copies the already-extracted colour jpg + aligned depth png into a self-contained
local dataset (NOT Google Drive), named by pid:
    ~/CMU/Research/mobile_data_local/multiview_v1/0508/win_<pid0>-<pid1>/{color,depth}/p<pid>.<ext>
and writes manifests that make the later lidar/radar extraction a pure lookup:
    per-window manifest.csv   pid, lidar_idx, radar_idx, camera_color_idx, camera_depth_idx, lidar_t, radar_t, camera_t
    dataset-root manifest.json  windows + census events inside each + extraction status {cam: done, lidar/radar: pending}
(a copy of manifest.json also lands in late_fusion/ for the pipeline side)
Run: python sync/dev/extract_multiview_cam.py
"""
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

if __name__ == "__main__":
    main()
