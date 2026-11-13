"""radar_first one-off diagnostics (merged diag_*). Subcommands: d14260_proc|d14280|dblobs|dgates|dglare_lidar|dglare_motion|dmoving_match|dradar_lidar|drange|dreject|dvscan"""
import sys

# ======== src/fusion/radar_first/diag_14260_proc.py ========
# Diagnose WHERE the process loses the car at 14260 (data is fine per user).
# 
# Shows, in the LiDAR frame:
#   - radar detections colored by the EGO-DOPPLER decision: kept-as-mover vs dropped-as-static
#   - strong radar BEV returns (the actual strong areas)
#   - LiDAR points
# so we can see if the car's strong radar is being dropped by the ego-Doppler gate or
# lost in clustering/density.
def d14260_proc_main():
    import os, sys
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import load_pairs, load_calib, LidarSource, RES, H5, CFG
    from detector import RadarDetector
    from ego_doppler import fit_ego_velocity
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import _radar_setup, _radar_panel, _peaks_from_bev, _extract_frames, DEFAULT_COLOR_TOPIC
    PAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 14260
    cal = load_calib(); pairs = load_pairs()
    cal = load_calib(); pairs = load_pairs()
    ci = cal["color_intrinsics"]; R2L = np.array(cal["radar_to_lidar_auto"])
    ci = cal["color_intrinsics"]; R2L = np.array(cal["radar_to_lidar_auto"])
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAM = "/Users/juntingdeng/CMU/Research/mobile_data_local/2026.05.11/camera/20260511_213724.bag"
    CIDX = f"{RES}/camera_index.npz"
    det = RadarDetector(processing="legacy")
    d, _ = det.detect(pairs[PAIR]["radar_idx"])
    Xd, Yd = RadarDetector.to_xy(d)
    lf = R2L[0,0]*Xd+R2L[0,1]*Yd+R2L[0,3]; ll = R2L[1,0]*Xd+R2L[1,1]*Yd+R2L[1,3]   # lidar frame
    lf = R2L[0,0]*Xd+R2L[0,1]*Yd+R2L[0,3]; ll = R2L[1,0]*Xd+R2L[1,1]*Yd+R2L[1,3]   # lidar frame
    ab, inl = fit_ego_velocity(d["az"], d["vel"])
    azr = np.radians(-d["az"]); pred = np.cos(azr)*ab[0] + np.sin(azr)*ab[1]
    azr = np.radians(-d["az"]); pred = np.cos(azr)*ab[0] + np.sin(azr)*ab[1]
    mover = (~inl) & (np.abs(d["vel"] - pred) > 0.8)
    print(f"detections {len(d)}  ego-inliers(static,dropped) {inl.sum()}  movers(kept) {mover.sum()}", flush=True)
    args = SimpleNamespace(radar_h5=H5, cfg_file=CFG, sync_csv=f"{RES}/sync_pairs.csv",
                           sync_summary=f"{RES}/sync_summary.json", lidar_pcap=None, lidar_metadata=None, radar_bev_res=0.1)
    loader, Rmat = _radar_setup(args, cal, doppler_mode="all")
    bev, la, fa = _radar_panel(loader, PAIR); pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.5)
    bev, la, fa = _radar_panel(loader, PAIR); pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.5)
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); ml = np.linalg.norm(P[:, :2], axis=1) < 28
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); ml = np.linalg.norm(P[:, :2], axis=1) < 28
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); ml = np.linalg.norm(P[:, :2], axis=1) < 28
    bag_ts = np.load(CIDX)["color_bag_ts_ns"]
    img = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[PAIR]["color_idx"]], Path(CACHE), "color", bag_ts).get(pairs[PAIR]["color_idx"])
    if img is not None and img.ndim == 1: img = img.reshape(ci["height"], ci["width"], 3)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    if img is not None: axs[0].imshow(img)
    axs[0].axis("off"); axs[0].set_title(f"CAMERA {PAIR} (where is the car?)")
    axs[0].axis("off"); axs[0].set_title(f"CAMERA {PAIR} (where is the car?)")
    axs[1].scatter(P[ml, 1], P[ml, 0], s=0.4, c="0.85")
    axs[1].scatter(pl, pf, s=70, marker="*", c="gold", edgecolors="orange", label="radar STRONG return", zorder=4)
    axs[1].scatter(ll[inl], lf[inl], s=40, marker="x", c="silver", label="radar det: DROPPED as static (ego-inlier)", zorder=3)
    axs[1].scatter(ll[mover], lf[mover], s=90, facecolors="none", edgecolors="red", linewidths=2,
                   label="radar det: KEPT as mover", zorder=5)
    other = (~inl) & (~mover)
    axs[1].scatter(ll[other], lf[other], s=40, marker="+", c="steelblue", label="outlier<margin (dropped)", zorder=3)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 26)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 26)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 26)
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
    axs[1].set_title(f"{PAIR}: radar strong returns + ego-Doppler keep/drop\n(is the car's strong return kept or dropped?)")
    axs[1].legend(loc="upper right", fontsize=8)
    out = f"{RES}/fusion/diag_" + str(PAIR) + "_proc.png"
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}", flush=True)

# ======== src/fusion/radar_first/diag_14280.py ========
# Diagnose pair 14280: glare-blind camera, LiDAR sees a dense FRONT face (no L-shape
# -> fails sizeok), radar has a strong arc. Show exactly why the car is missed and what
# would recover it (relaxed size + radar-range support).
def d14280_main():
    import os, sys
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import load_pairs, load_calib, LidarSource, LOCAL, RES, H5, CFG
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import (_cluster_bev, _drop_ground, _ground_level,
                               _radar_setup, _radar_panel, _peaks_from_bev,
                               _extract_frames, DEFAULT_COLOR_TOPIC)
    PAIR = 14280
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAMERA_BAG = f"{LOCAL}/camera/20260511_213724.bag"; CAMERA_INDEX = f"{RES}/camera_index.npz"
    CAMERA_BAG = f"{LOCAL}/camera/20260511_213724.bag"; CAMERA_INDEX = f"{RES}/camera_index.npz"
    ROI = (0.0, 30.0, -16.0, 16.0); GPCT, MINH = 10.0, 0.3
    ROI = (0.0, 30.0, -16.0, 16.0); GPCT, MINH = 10.0, 0.3
    MAJ = (2.0, 6.5); MIN = (1.2, 3.5); HGT = (0.5, 2.4)
    MAJ = (2.0, 6.5); MIN = (1.2, 3.5); HGT = (0.5, 2.4)
    MAJ = (2.0, 6.5); MIN = (1.2, 3.5); HGT = (0.5, 2.4)
    pairs = load_pairs(); cal = load_calib()
    pairs = load_pairs(); cal = load_calib()
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"])
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"])
    x0, x1, y0, y1 = ROI
    reg = P[(P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)]
    gz = _ground_level(reg, GPCT)
    obj = _drop_ground(reg, GPCT, MINH)
    labels = _cluster_bev(obj[:, :2], ROI)
    args = SimpleNamespace(radar_h5=H5, cfg_file=CFG, sync_csv=f"{RES}/sync_pairs.csv",
                           sync_summary=f"{RES}/sync_summary.json", lidar_pcap=None, lidar_metadata=None,
                           radar_bev_res=0.1)
    loader, Rmat = _radar_setup(args, cal, doppler_mode="all", processing="legacy")
    bev, la, fa = _radar_panel(loader, PAIR)
    pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.0)
    rows = []
    clusters = []
    for lb in np.unique(labels):
        cl = obj[labels == lb]
        if cl.shape[0] < 8:
            continue
        L = float(np.ptp(cl[:, 0])); W = float(np.ptp(cl[:, 1])); H = float(np.ptp(cl[:, 2]))
        mx, mn = max(L, W), min(L, W)
        sizeok = (MAJ[0] <= mx <= MAJ[1] and MIN[0] <= mn <= MIN[1] and HGT[0] <= H <= HGT[1])
        cen = cl[:, :2].mean(0)
        rng = float(np.hypot(*cen))
        radar_xy = bool(len(pf) and (np.hypot(pf - cen[0], pl - cen[1]) < 2.5 + mx/2).any())
        radar_rng = bool(len(pf) and (np.abs(np.hypot(pf, pl) - rng) < 1.5).any())  # radar at same RANGE
        clusters.append((cl, sizeok, cen, L, W, H, radar_xy, radar_rng))
        if cl.shape[0] >= 12:
            rows.append((int(cl.shape[0]), cen[0], cen[1], L, W, H, sizeok, radar_xy, radar_rng))
    print(f"pair {PAIR}: {len(clusters)} clusters; radar peaks {len(pf)}", flush=True)
    print(" n   fwd    lat    L    W    H   sizeok radarXY radarRNG", flush=True)
    for r in sorted(rows, key=lambda z: z[1]):
        print(f"{r[0]:4d} {r[1]:5.1f} {r[2]:6.1f} {r[3]:4.1f} {r[4]:4.1f} {r[5]:4.1f}   "
              f"{str(r[6]):5s}  {str(r[7]):5s}  {str(r[8]):5s}", flush=True)
    bag_ts = np.load(CAMERA_INDEX)["color_bag_ts_ns"]; cidx = pairs[PAIR]["color_idx"]
    bag_ts = np.load(CAMERA_INDEX)["color_bag_ts_ns"]; cidx = pairs[PAIR]["color_idx"]
    img = _extract_frames(CAMERA_BAG, DEFAULT_COLOR_TOPIC, [cidx], Path(CACHE), "color", bag_ts).get(cidx)
    if img is not None and img.ndim == 1:
        img = img.reshape(ci["height"], ci["width"], 3)
    fig, axs = plt.subplots(1, 2, figsize=(19, 8))
    if img is not None:
        axs[0].imshow(img)
    axs[0].axis("off"); axs[0].set_title(f"CAMERA pair {PAIR} (glare / night)")
    axs[0].axis("off"); axs[0].set_title(f"CAMERA pair {PAIR} (glare / night)")
    ml = np.linalg.norm(P[:, :2], axis=1) < 30
    axs[1].scatter(P[ml, 1], P[ml, 0], s=0.4, c="0.85")
    for cl, sizeok, cen, L, W, H, rxy, rrng in clusters:
        if cl.shape[0] < 12:
            continue
        col = "limegreen" if sizeok else "red"
        axs[1].scatter(cl[:, 1], cl[:, 0], s=6, c=col)
        tag = f"L{L:.1f} W{W:.1f}" + ("  R-arc" if rrng else "")
        axs[1].text(cen[1], cen[0] + 0.4, tag, fontsize=7, color=col)
    axs[1].scatter(pl, pf, s=22, marker="x", c="purple", label="radar peak (arc)")
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 28)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 28)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 28)
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
    axs[1].set_title("LiDAR clusters: green=sizeok PASS, red=FAIL (missed)\n+ radar peaks (x)")
    axs[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out = f"{RES}/fusion/diag_14280.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}", flush=True)

# ======== src/fusion/radar_first/diag_blobs.py ========
# Show what the glare-blob detector sees on a frame: bright mask + every connected component
# (size) + merged glare regions. Reveals which cars/glare are missed and why (threshold/VBAND/size).
#     python3 diag_blobs.py 14570 14600
def dblobs_main():
    import os, sys
    from pathlib import Path
    import numpy as np
    from scipy import ndimage
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from common import load_pairs, load_calib, LOCAL, RES
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
    PAIRS = [int(a) for a in sys.argv[1:]] or [14570, 14600]
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    BRIGHT = 205; VBAND = (0.20, 0.85); MIN_BLOB = 60
    BRIGHT = 205; VBAND = (0.20, 0.85); MIN_BLOB = 60
    BRIGHT = 205; VBAND = (0.20, 0.85); MIN_BLOB = 60
    cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
    cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
    cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
    pairs = load_pairs(); bag_ts = np.load(CIDX)["color_bag_ts_ns"]
    pairs = load_pairs(); bag_ts = np.load(CIDX)["color_bag_ts_ns"]
    imgs = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[p]["color_idx"] for p in PAIRS], Path(CACHE), "color", bag_ts)
    for P in PAIRS:
        img = imgs[pairs[P]["color_idx"]]
        if img.ndim == 1: img = img.reshape(H, W, 3)
        # try several brightness thresholds to see what's recoverable
        print(f"\npair {P}:", flush=True)
        for TH in (180, 205, 225):
            bright = (img.min(2) > TH).copy()
            bright[:int(H*VBAND[0])] = False; bright[int(H*VBAND[1]):] = False
            lab, n = ndimage.label(bright)
            comps = []
            for i in range(1, n+1):
                ys, xs = np.where(lab == i)
                if len(xs) >= MIN_BLOB: comps.append((xs.min(), xs.max(), ys.mean(), len(xs)))
            print(f"  BRIGHT>{TH}: {len(comps)} blobs (>= {MIN_BLOB}px): " +
                  ", ".join(f"[u{a}-{b} v{v:.0f} n{s}]" for a, b, v, s in sorted(comps)), flush=True)

        fig, axs = plt.subplots(1, 2, figsize=(18, 7))
        axs[0].imshow(img); axs[0].set_title(f"{P}: camera"); axs[0].axis("off")
        axs[1].imshow(img)
        bright = (img.min(2) > BRIGHT).copy(); bright[:int(H*VBAND[0])] = False; bright[int(H*VBAND[1]):] = False
        ov = np.zeros((*bright.shape, 4)); ov[bright] = [1, 0, 0, 0.5]; axs[1].imshow(ov)
        lab, n = ndimage.label(bright)
        for i in range(1, n+1):
            ys, xs = np.where(lab == i)
            if len(xs) < MIN_BLOB: continue
            axs[1].add_patch(Rectangle((xs.min(), ys.min()), xs.max()-xs.min(), ys.max()-ys.min(),
                             fill=False, ec="lime", lw=2))
            axs[1].text(xs.min(), ys.min()-4, f"n{len(xs)}", color="lime", fontsize=9)
        axs[1].axhline(H*VBAND[0], color="cyan", lw=0.8); axs[1].axhline(H*VBAND[1], color="cyan", lw=0.8)
        axs[1].set_title(f"{P}: BRIGHT>{BRIGHT} mask (red) + blobs (lime), VBAND (cyan)"); axs[1].axis("off")
        out = f"{RES}/fusion/night_label/diag_blobs_{P}.png"
        plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close(); print(f"  saved {out}", flush=True)

# ======== src/fusion/radar_first/diag_gates.py ========
# Per-cluster verdict of each generalizable gate (model/plane/tall/isolation) for a pair.
#     python3 diag_gates.py 14630 14520
def dgates_main():
    import os, sys
    from pathlib import Path
    import numpy as np
    from scipy import ndimage
    from sklearn.cluster import DBSCAN
    from common import load_pairs, load_calib, LidarSource, LOCAL, RES
    import glare_bearing as G   # reuse constants + fit_ground_plane + ray + car_glare_blobs
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
    PAIRS = [int(a) for a in sys.argv[1:]] or [14630]
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
    bag_ts = np.load(CIDX)["color_bag_ts_ns"]; src = LidarSource()
    bag_ts = np.load(CIDX)["color_bag_ts_ns"]; src = LidarSource()
    imgs = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[p]["color_idx"] for p in PAIRS], Path(CACHE), "color", bag_ts)
    for PAIR in PAIRS:
        img = imgs[pairs[PAIR]["color_idx"]]
        if img.ndim == 1: img = img.reshape(H, W, 3)
        blobs = G.car_glare_blobs(img)
        P = src.xyz(pairs[PAIR]["lidar_idx"]); P = P[np.linalg.norm(P[:, :2], axis=1) < G.RMAX]
        coef = G.fit_ground_plane(P); Pha = G.height_above(P, coef)
        slab = P[(Pha > 0.3) & (Pha < 2.2)]; sb = np.degrees(np.arctan2(slab[:, 1], slab[:, 0]))
        print(f"\npair {PAIR}: plane coef a={coef[0]:+.3f} b={coef[1]:+.3f} c={coef[2]:+.2f}, {len(blobs)} blobs", flush=True)
        for b in blobs:
            b_lo, b_hi = sorted((G.ray(b["u0"], b["vb"])[0], G.ray(b["u1"], b["vb"])[0]))
            onb = (sb >= b_lo-G.BEARING_TOL) & (sb <= b_hi+G.BEARING_TOL); pts = slab[onb]
            if len(pts) < G.CAR_MIN_PTS: continue
            lb = DBSCAN(eps=1.0, min_samples=4).fit_predict(pts[:, :2])
            for c in set(lb.tolist()):
                if c == -1: continue
                cl = pts[lb == c]
                if len(cl) < G.CAR_MIN_PTS: continue
                cen = cl[:, :2].mean(0); X = cl[:, :2]-cen
                _, _, vt = np.linalg.svd(X, full_matrices=False); proj = X @ vt.T
                mj, mnr = float(np.ptp(proj[:, 0])), float(np.ptp(proj[:, 1]))
                hmin = float(G.height_above(cl, coef).min())
                x0, x1 = cl[:, 0].min()-0.4, cl[:, 0].max()+0.4; y0, y1 = cl[:, 1].min()-0.4, cl[:, 1].max()+0.4
                box = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
                tall = int((box & (Pha > G.TALL_H)).sum())
                m = 1.2; big = (P[:, 0] >= x0-m) & (P[:, 0] <= x1+m) & (P[:, 1] >= y0-m) & (P[:, 1] <= y1+m)
                ring = big & ~box & (Pha > 0.3) & (Pha < 2.2)
                a_cl = max((x1-x0)*(y1-y0), 1.0); a_ring = max((x1-x0+2*m)*(y1-y0+2*m)-a_cl, 1.0)
                iso = (ring.sum()/a_ring) / (len(cl)/a_cl)
                r = float(np.hypot(*cen))
                v = []
                if not (G.CAR_LEN[0] <= mj <= G.CAR_LEN[1] and G.CAR_WID[0] <= mnr <= G.CAR_WID[1]): v.append("MODEL")
                if hmin > G.GROUND_CLEAR: v.append("PLANE")
                if tall >= 8: v.append("TALL")
                if iso > G.ISO_RATIO: v.append("ISO")
                print(f"  r{r:5.1f} ({cen[0]:5.1f},{cen[1]:5.1f}) {mj:.1f}x{mnr:.1f} n{len(cl):3d} hmin{hmin:+.2f} tall{tall:3d} iso{iso:.2f} -> {'KEEP' if not v else 'reject:'+','.join(v)}", flush=True)

# ======== src/fusion/radar_first/diag_glare_lidar.py ========
# Where is the car's LiDAR along the glare bearing? (why the glare+radar box is empty)
# 
# Shows the LiDAR car-height clusters along the glare bearing with their size/density, so
# we can see if the car has a dense cluster (and at what range) vs the box's radar range.
def dglare_lidar_main():
    import os, sys
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    from scipy import ndimage
    from sklearn.cluster import DBSCAN
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import load_pairs, load_calib, LidarSource, LOCAL, RES, H5, CFG
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import _radar_setup, _radar_panel, _peaks_from_bev, _extract_frames, DEFAULT_COLOR_TOPIC
    PAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 14560
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    BRIGHT = 205; VBAND = (0.30, 0.78); MIN_BLOB = 60
    BRIGHT = 205; VBAND = (0.30, 0.78); MIN_BLOB = 60
    BRIGHT = 205; VBAND = (0.30, 0.78); MIN_BLOB = 60
    pairs = load_pairs(); cal = load_calib()
    pairs = load_pairs(); cal = load_calib()
    ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
    ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
    ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; Hh, Wd = ci["height"], ci["width"]
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; Hh, Wd = ci["height"], ci["width"]
    R2L = np.array(cal["radar_to_lidar_auto"])
    bag_ts = np.load(CIDX)["color_bag_ts_ns"]
    img = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[PAIR]["color_idx"]], Path(CACHE), "color", bag_ts).get(pairs[PAIR]["color_idx"])
    if img.ndim == 1: img = img.reshape(Hh, Wd, 3)
    bright = (img.min(2) > BRIGHT); bright[:int(Hh*VBAND[0])] = False; bright[int(Hh*VBAND[1]):] = False
    bright = (img.min(2) > BRIGHT); bright[:int(Hh*VBAND[0])] = False; bright[int(Hh*VBAND[1]):] = False
    bright = (img.min(2) > BRIGHT); bright[:int(Hh*VBAND[0])] = False; bright[int(Hh*VBAND[1]):] = False
    lab, nl = ndimage.label(bright)
    cols = [np.where(lab == i)[1] for i in range(1, nl+1)]; cols = [c for c in cols if len(c) >= MIN_BLOB]
    cols = [np.where(lab == i)[1] for i in range(1, nl+1)]; cols = [c for c in cols if len(c) >= MIN_BLOB]
    allc = np.concatenate(cols) if cols else np.array([cx])
    umin, umax, vrow = allc.min(), allc.max(), Hh*0.55
    def bof(u): d = np.array([(u-cx)/fx, 0.0, 1.0]); d /= np.linalg.norm(d); dl = Rc2l @ d; return np.degrees(np.arctan2(dl[1], dl[0]))
    b_lo, b_hi = sorted((bof(umin), bof(umax))); bearing = 0.5*(b_lo+b_hi)
    b_lo, b_hi = sorted((bof(umin), bof(umax))); bearing = 0.5*(b_lo+b_hi)
    print(f"pair {PAIR}: glare cols [{umin},{umax}] -> bearing window [{b_lo:+.1f},{b_hi:+.1f}] deg", flush=True)
    args = SimpleNamespace(radar_h5=H5, cfg_file=CFG, sync_csv=f"{RES}/sync_pairs.csv",
                           sync_summary=f"{RES}/sync_summary.json", lidar_pcap=None, lidar_metadata=None, radar_bev_res=0.1)
    loader, Rmat = _radar_setup(args, cal, doppler_mode="all"); bev, la, fa = _radar_panel(loader, PAIR)
    loader, Rmat = _radar_setup(args, cal, doppler_mode="all"); bev, la, fa = _radar_panel(loader, PAIR)
    pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.0); pbear = np.degrees(np.arctan2(pl, pf)); prng = np.hypot(pf, pl)
    pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.0); pbear = np.degrees(np.arctan2(pl, pf)); prng = np.hypot(pf, pl)
    pf, pl = _peaks_from_bev(bev, la, fa, Rmat, 99.0); pbear = np.degrees(np.arctan2(pl, pf)); prng = np.hypot(pf, pl)
    onr = (pbear >= b_lo-6) & (pbear <= b_hi+6)
    rr = np.sort(prng[onr]) if onr.any() else np.array([])
    print(f"  radar ranges in window: {np.round(rr, 1)[:20]}", flush=True)
    RMAX = 45.0
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < RMAX
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < RMAX
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < RMAX
    gz = np.percentile(P[nm][:, 2], 10)
    slab = P[nm & (P[:, 2] > gz+0.3) & (P[:, 2] < gz+2.2)]
    sb = np.degrees(np.arctan2(slab[:, 1], slab[:, 0])); onb = (sb >= b_lo-6) & (sb <= b_hi+6)
    sb = np.degrees(np.arctan2(slab[:, 1], slab[:, 0])); onb = (sb >= b_lo-6) & (sb <= b_hi+6)
    lb = slab[onb]
    print(f"  lidar car-height pts in window (<{RMAX:.0f}m): {len(lb)}", flush=True)
    clusters = []
    if len(lb) >= 6:
        L = DBSCAN(eps=1.2, min_samples=4).fit_predict(lb[:, :2])
        for c in set(L.tolist()):
            if c == -1: continue
            cl = lb[L == c]; cen = cl[:, :2].mean(0); r = float(np.hypot(*cen))
            fpr = np.ptp(cl[:, :2], axis=0); mx, mn = max(fpr), min(fpr)
            carish = 1.5 <= mx <= 6.5 and mn <= 3.5
            clusters.append((r, len(cl), cen, fpr))
            print(f"    lidar cluster r={r:4.1f}m n={len(cl):3d} footprint {fpr[0]:.1f}x{fpr[1]:.1f} {'CAR' if carish else ''}", flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    axs[0].imshow(img); axs[0].axvline(umin, color="lime", lw=1); axs[0].axvline(umax, color="lime", lw=1)
    axs[0].imshow(img); axs[0].axvline(umin, color="lime", lw=1); axs[0].axvline(umax, color="lime", lw=1)
    axs[0].imshow(img); axs[0].axvline(umin, color="lime", lw=1); axs[0].axvline(umax, color="lime", lw=1)
    axs[0].axis("off"); axs[0].set_title(f"CAMERA {PAIR}: glare window (lime)")
    axs[0].axis("off"); axs[0].set_title(f"CAMERA {PAIR}: glare window (lime)")
    axs[1].scatter(P[nm, 1], P[nm, 0], s=0.5, c="0.8")
    axs[1].scatter(pl, pf, s=12, marker="x", c="purple", label="radar strong")
    axs[1].scatter(lb[:, 1], lb[:, 0], s=10, c="orange", label="lidar car-ht in window")
    for bb in (b_lo, b_hi):
        r = np.linspace(0, RMAX, 2); axs[1].plot(r*np.sin(np.radians(bb)), r*np.cos(np.radians(bb)), "lime", lw=1, ls="--")
    for rr_, n, cen, fpr in clusters:
        axs[1].text(cen[1], cen[0], f"r{rr_:.0f} n{n}", color="red", fontsize=9)
    axs[1].set_aspect("equal"); axs[1].set_xlim(22, -8); axs[1].set_ylim(0, RMAX)
    axs[1].set_aspect("equal"); axs[1].set_xlim(22, -8); axs[1].set_ylim(0, RMAX)
    axs[1].set_aspect("equal"); axs[1].set_xlim(22, -8); axs[1].set_ylim(0, RMAX)
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)"); axs[1].legend(loc="upper right", fontsize=8)
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)"); axs[1].legend(loc="upper right", fontsize=8)
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)"); axs[1].legend(loc="upper right", fontsize=8)
    axs[1].set_title(f"{PAIR}: lidar clusters in glare window (how many cars?)")
    out = f"{RES}/fusion/glare/diag_glare_lidar_{PAIR}.png"
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); print(f"saved {out}", flush=True)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); print(f"saved {out}", flush=True)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); print(f"saved {out}", flush=True)

# ======== src/fusion/radar_first/diag_glare_motion.py ========
# Does ego-compensated radar DOPPLER isolate the moving glare-blind cars?
# 
# Overlays, per pair: ego-Doppler MOVING radar dets (red) vs STATIC (gray) + the glare bearing
# cone + the current glare_bearing boxes. If the real cars = moving radar and the wrong boxes sit
# on static/nothing, then a motion gate fixes it. Runs several pairs in one radar open.
#     python3 diag_glare_motion.py 15510 15590
def dglare_motion_main():
    import os, sys, json
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    from scipy import ndimage
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import load_pairs, load_calib, LidarSource, LOCAL, RES, H5, CFG
    from detector import RadarDetector
    from ego_doppler import fit_ego_velocity
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
    PAIRS = [int(a) for a in sys.argv[1:]] or [15510, 15590]
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    BRIGHT = 205; VBAND = (0.30, 0.78); MIN_BLOB = 60; MOVER_MARGIN = 0.8; RMAX = 45.0
    BRIGHT = 205; VBAND = (0.30, 0.78); MIN_BLOB = 60; MOVER_MARGIN = 0.8; RMAX = 45.0
    BRIGHT = 205; VBAND = (0.30, 0.78); MIN_BLOB = 60; MOVER_MARGIN = 0.8; RMAX = 45.0
    BRIGHT = 205; VBAND = (0.30, 0.78); MIN_BLOB = 60; MOVER_MARGIN = 0.8; RMAX = 45.0
    BRIGHT = 205; VBAND = (0.30, 0.78); MIN_BLOB = 60; MOVER_MARGIN = 0.8; RMAX = 45.0
    pairs = load_pairs(); cal = load_calib()
    pairs = load_pairs(); cal = load_calib()
    ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
    ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
    ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
    R2L = np.array(cal["radar_to_lidar_auto"])
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; H, W = ci["height"], ci["width"]
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; H, W = ci["height"], ci["width"]
    bag_ts = np.load(CIDX)["color_bag_ts_ns"]
    imgs = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[p]["color_idx"] for p in PAIRS], Path(CACHE), "color", bag_ts)
    det = RadarDetector(processing="legacy"); src = LidarSource()
    det = RadarDetector(processing="legacy"); src = LidarSource()
    ann = json.load(open(f"{RES}/annotations/glare_win{PAIRS[0]}_{PAIRS[-1]}.json")) if len(PAIRS) > 1 else {"objects": {}}
    def glare_windows(img):
        bright = (img.min(axis=2) > BRIGHT); bright[: int(H*VBAND[0])] = False; bright[int(H*VBAND[1]):] = False
        lab, nlab = ndimage.label(bright); blobs = []
        for i in range(1, nlab+1):
            xs = np.where(lab == i)[1]
            if len(xs) >= MIN_BLOB: blobs.append([xs.min(), xs.max()])
        blobs.sort(); merged = []
        for u0, u1 in blobs:
            if merged and u0 - merged[-1][1] < 100: merged[-1][1] = max(merged[-1][1], u1)
            else: merged.append([u0, u1])
        def bof(u):
            d = np.array([(u-cx)/fx, 0.0, 1.0]); d /= np.linalg.norm(d); dl = Rc2l @ d
            return np.degrees(np.arctan2(dl[1], dl[0]))
        return [sorted((bof(a), bof(b))) for a, b in merged]
    for PAIR in PAIRS:
        img = imgs.get(pairs[PAIR]["color_idx"])
        if img is not None and img.ndim == 1: img = img.reshape(H, W, 3)
        wins = glare_windows(img)
        d, _ = det.detect(pairs[PAIR]["radar_idx"])
        Xd, Yd = RadarDetector.to_xy(d)
        lf = R2L[0,0]*Xd + R2L[0,1]*Yd + R2L[0,3]; ll = R2L[1,0]*Xd + R2L[1,1]*Yd + R2L[1,3]
        ab, inl = fit_ego_velocity(d["az"], d["vel"])
        if ab is not None:
            azr = np.radians(-d["az"]); pred = np.cos(azr)*ab[0] + np.sin(azr)*ab[1]
            mover = (~inl) & (np.abs(d["vel"] - pred) > MOVER_MARGIN)
        else:
            mover = np.zeros(len(d), bool)
        P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < RMAX
        print(f"pair {PAIR}: {len(d)} dets, {mover.sum()} MOVING (ego-Doppler), {len(ann['objects'].get(str(PAIR),[]))} current boxes", flush=True)

        fig, axs = plt.subplots(1, 2, figsize=(17, 8))
        axs[0].imshow(img); axs[0].axis("off"); axs[0].set_title(f"CAMERA {PAIR}: glare windows")
        for b in ann["objects"].get(str(PAIR), []):
            bb = b.get("bbox_xyxy")
            if bb: axs[0].add_patch(plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], fill=False,
                       ec=("lime" if b["color"] == "lime" else "orange"), lw=2))
        axs[1].scatter(P[nm, 1], P[nm, 0], s=0.4, c="0.85")
        axs[1].scatter(ll[~mover], lf[~mover], s=30, marker="x", c="silver", label="radar STATIC")
        axs[1].scatter(ll[mover], lf[mover], s=80, facecolors="none", edgecolors="red", linewidths=2, label="radar MOVING")
        for b in ann["objects"].get(str(PAIR), []):
            o = b["box3d"]; col = "lime" if b["color"] == "lime" else "orange"
            axs[1].add_patch(plt.Rectangle((o["cy"]-o["width"]/2, o["cx"]-o["length"]/2), o["width"], o["length"],
                             fill=False, ec=col, lw=1.5))
        for blo, bhi in wins:
            for bb in (blo, bhi):
                r = np.linspace(0, RMAX, 2); axs[1].plot(r*np.sin(np.radians(bb)), r*np.cos(np.radians(bb)), "b", lw=0.7, ls="--")
        axs[1].set_aspect("equal"); axs[1].set_xlim(24, -24); axs[1].set_ylim(0, RMAX)
        axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)"); axs[1].legend(loc="upper right", fontsize=8)
        axs[1].set_title(f"{PAIR}: MOVING radar (red) vs STATIC + current boxes (green/orange) + glare cone")
        out = f"{RES}/fusion/glare/diag_motion_{PAIR}.png"
        plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(); print(f"saved {out}", flush=True)

# ======== src/fusion/radar_first/diag_moving_match.py ========
# WHY does moving-Doppler never corroborate the glare cars?
# 
# For each kept box, find the nearest MOVING radar detection and print d_range / d_bearing.
# If movers sit at the right RANGE but wrong BEARING -> coarse radar azimuth (loosen bearing gate).
# If no movers near in range at all -> the boxed cars are stopped, or ego-Doppler misses them.
#     python3 diag_moving_match.py 15480 15580 15560
def dmoving_match_main():
    import os, sys, json
    from types import SimpleNamespace
    import numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import load_pairs, load_calib, LidarSource, RES
    from detector import RadarDetector
    from ego_doppler import fit_ego_velocity
    PAIRS = [int(a) for a in sys.argv[1:]] or [15480, 15580]
    cal = load_calib(); pairs = load_pairs()
    cal = load_calib(); pairs = load_pairs()
    R2L = np.array(cal["radar_to_lidar_auto"])
    det = RadarDetector(processing="legacy"); src = LidarSource()
    det = RadarDetector(processing="legacy"); src = LidarSource()
    tag = f"win{min(PAIRS)//10*10+0}" if False else None
    ANN = None
    for f in os.listdir(f"{RES}/annotations"):
        if f.startswith("glare_win"):
            o = json.load(open(f"{RES}/annotations/{f}"))["objects"]
            if all(str(p) in o for p in PAIRS): ANN = o; break
    ANN = ANN or {}
    def bear(a): return np.degrees(np.arctan2(a[1], a[0]))
    for PAIR in PAIRS:
        d, _ = det.detect(pairs[PAIR]["radar_idx"]); Xd, Yd = RadarDetector.to_xy(d)
        rf = R2L[0,0]*Xd + R2L[0,1]*Yd + R2L[0,3]; rl = R2L[1,0]*Xd + R2L[1,1]*Yd + R2L[1,3]
        rrng = np.hypot(rf, rl); rbear = np.degrees(np.arctan2(rl, rf))
        ab, inl = fit_ego_velocity(d["az"], d["vel"])
        if ab is not None:
            pr = np.cos(np.radians(-d["az"]))*ab[0] + np.sin(np.radians(-d["az"]))*ab[1]
            resid = d["vel"] - pr; mover = (~inl) & (np.abs(resid) > 0.8)
            egospd = np.hypot(ab[0], ab[1])
        else:
            mover = np.zeros(len(d), bool); egospd = float("nan")
        print(f"\npair {PAIR}: ego-speed {egospd:.2f} m/s, {mover.sum()}/{len(d)} movers", flush=True)
        mv_r, mv_b, mv_v = rrng[mover], rbear[mover], d["vel"][mover]
        for i in np.argsort(mv_r):
            print(f"   mover: range {mv_r[i]:5.1f}  bearing {mv_b[i]:+6.1f}  vrad {mv_v[i]:+5.1f}", flush=True)
        for b in ANN.get(str(PAIR), []):
            o = b["box3d"]; br, bb = np.hypot(o["cx"], o["cy"]), bear([o["cx"], o["cy"]])
            if mover.any():
                dr = np.abs(mv_r - br); dbear = np.abs(((mv_b - bb + 180) % 360) - 180)
                j = np.argmin(dr + dbear/5.0)
                print(f"   BOX r{br:5.1f} b{bb:+6.1f} [{b['source']}] -> nearest mover dR={dr[j]:+.1f} dBear={dbear[j]:+.1f}", flush=True)
            else:
                print(f"   BOX r{br:5.1f} b{bb:+6.1f} [{b['source']}] -> no movers", flush=True)

        P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < 45
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.scatter(P[nm, 1], P[nm, 0], s=0.4, c="0.85")
        ax.scatter(rl[~mover], rf[~mover], s=25, marker="x", c="silver", label="radar static")
        ax.scatter(rl[mover], rf[mover], s=90, facecolors="none", edgecolors="red", linewidths=2, label="radar MOVING")
        for b in ANN.get(str(PAIR), []):
            o = b["box3d"]; col = "lime" if b["color"] == "lime" else "orange"
            ax.add_patch(plt.Rectangle((o["cy"]-o["width"]/2, o["cx"]-o["length"]/2), o["width"], o["length"], fill=False, ec=col, lw=2))
        ax.set_aspect("equal"); ax.set_xlim(24, -24); ax.set_ylim(0, 45)
        ax.set_xlabel("lateral Y"); ax.set_ylabel("forward X"); ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{PAIR}: MOVING radar (red) vs boxes (green/orange) -- do they coincide?")
        out = f"{RES}/fusion/glare/diag_mvmatch_{PAIR}.png"
        plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close(); print(f"saved {out}", flush=True)

# ======== src/fusion/radar_first/diag_radar_lidar.py ========
# Diagnostic: do RAW radar detections land on the LiDAR car clusters?
# 
# Single frame, LiDAR frame, NO odometry / NO tracking -- isolates whether radar
# proposals actually coincide with the cars (then the fix is association+boxing) or
# not (then it's calibration/detection). This is the honest test of radar-first.
def dradar_lidar_main():
    import numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from common import load_pairs, load_calib, LidarSource, RES
    from detector import RadarDetector
    from ego_doppler import fit_ego_velocity
    PAIR = 13890
    pairs = load_pairs()
    R2L = np.array(load_calib()["radar_to_lidar_auto"])
    det = RadarDetector(processing="legacy")
    d, _ = det.detect(pairs[PAIR]["radar_idx"])
    azr = np.radians(-d["az"]); Xr = d["range"]*np.cos(azr); Yr = d["range"]*np.sin(azr)
    azr = np.radians(-d["az"]); Xr = d["range"]*np.cos(azr); Yr = d["range"]*np.sin(azr)
    azr = np.radians(-d["az"]); Xr = d["range"]*np.cos(azr); Yr = d["range"]*np.sin(azr)
    lf = R2L[0,0]*Xr + R2L[0,1]*Yr + R2L[0,3]
    ll = R2L[1,0]*Xr + R2L[1,1]*Yr + R2L[1,3]
    ab, inl = fit_ego_velocity(d["az"], d["vel"])
    if ab is not None:
        pred = np.cos(azr)*ab[0] + np.sin(azr)*ab[1]
        mover = (~inl) & (np.abs(d["vel"] - pred) > 0.8)
    else:
        mover = np.zeros(len(d), bool)
    det_ti = RadarDetector(processing="ti")
    dt, _ = det_ti.detect(pairs[PAIR]["radar_idx"], min_abs_vel=0.3)
    azt = np.radians(-dt["az"]); Xt = dt["range"]*np.cos(azt); Yt = dt["range"]*np.sin(azt)
    azt = np.radians(-dt["az"]); Xt = dt["range"]*np.cos(azt); Yt = dt["range"]*np.sin(azt)
    azt = np.radians(-dt["az"]); Xt = dt["range"]*np.cos(azt); Yt = dt["range"]*np.sin(azt)
    lft = R2L[0,0]*Xt + R2L[0,1]*Yt + R2L[0,3]
    llt = R2L[1,0]*Xt + R2L[1,1]*Yt + R2L[1,3]
    src = LidarSource()
    P = src.xyz(pairs[PAIR]["lidar_idx"])
    near = (np.linalg.norm(P[:, :2], axis=1) < 30)
    gz = np.percentile(P[near][:, 2], 3)
    obj = P[near & (P[:, 2] > gz + 0.3) & (P[:, 2] < gz + 2.2)]
    clusters = []
    if len(obj) > 30:
        lab = DBSCAN(eps=0.7, min_samples=8).fit_predict(obj[:, :2])
        for c in set(lab):
            if c == -1: continue
            pts = obj[lab == c][:, :2]
            if not (15 <= len(pts) <= 3000): continue
            fp = (pts.max(0) - pts.min(0))
            if 0.8 <= max(fp) <= 7 and min(fp) <= 4:         # car / object sized
                clusters.append(pts)
    fig, ax = plt.subplots(figsize=(11, 11))
    m = np.linalg.norm(P[:, :2], axis=1) < 30
    ax.scatter(P[m, 1], P[m, 0], s=0.4, c="0.8", label="lidar")
    for i, pts in enumerate(clusters):
        ax.scatter(pts[:, 1], pts[:, 0], s=6, label="lidar car-cluster" if i == 0 else None)
        cx, cy = pts.mean(0)
        ax.add_patch(plt.Rectangle((pts[:,1].min(), pts[:,0].min()),
                     np.ptp(pts[:,1]), np.ptp(pts[:,0]), fill=False, ec="k", lw=1.2))
    ax.scatter(ll, lf, s=18, c="silver", edgecolors="gray", label="radar det (all)", zorder=3)
    ax.scatter(llt, lft, s=55, marker="s", facecolors="none", edgecolors="blue",
               label="radar MOVING ('ti' notch)", zorder=4)
    if mover.any():
        ax.scatter(ll[mover], lf[mover], s=90, facecolors="none", edgecolors="red",
                   linewidths=2, label="radar mover (ego-Doppler)", zorder=5)
    ax.set_aspect("equal"); ax.set_xlim(15, -15); ax.set_ylim(0, 28)
    ax.set_aspect("equal"); ax.set_xlim(15, -15); ax.set_ylim(0, 28)
    ax.set_aspect("equal"); ax.set_xlim(15, -15); ax.set_ylim(0, 28)
    ax.set_xlabel("lateral Y (m)"); ax.set_ylabel("forward X (m)")
    ax.set_xlabel("lateral Y (m)"); ax.set_ylabel("forward X (m)")
    ax.set_title(f"pair {PAIR}: do radar detections land on the LiDAR cars?\n"
                 f"{len(clusters)} lidar clusters · {len(d)} radar det · {int(mover.sum())} ego-movers · {len(dt)} 'ti'-moving")
    ax.legend(loc="upper right", fontsize=8)
    out = f"{RES}/fusion/diag_radar_on_lidar.png"
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"clusters {len(clusters)}; radar det {len(d)}; ego-movers {int(mover.sum())}; ti-moving {len(dt)}", flush=True)
    print(f"saved {out}", flush=True)

# ======== src/fusion/radar_first/diag_range.py ========
# Along the glare bearing: is there a CAR in lidar, and at what range? Or is the car beyond lidar range
# (only near foliage returns)? Scatter of (range, height-above-plane) for points on the glare bearing.
#     python3 diag_range.py 15500 15480
def drange_main():
    import os, sys
    from pathlib import Path
    import numpy as np
    from scipy import ndimage
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import load_pairs, load_calib, LidarSource, LOCAL, RES
    import glare_bearing as G
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
    PAIRS = [int(a) for a in sys.argv[1:]] or [15500]
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]; H, W = ci["height"], ci["width"]
    bag_ts = np.load(CIDX)["color_bag_ts_ns"]; src = LidarSource()
    bag_ts = np.load(CIDX)["color_bag_ts_ns"]; src = LidarSource()
    imgs = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[p]["color_idx"] for p in PAIRS], Path(CACHE), "color", bag_ts)
    for PAIR in PAIRS:
        img = imgs[pairs[PAIR]["color_idx"]]
        if img.ndim == 1: img = img.reshape(H, W, 3)
        blobs = G.car_glare_blobs(img)
        P = src.xyz(pairs[PAIR]["lidar_idx"]); rng_all = np.linalg.norm(P[:, :2], axis=1)
        near = P[rng_all < G.RMAX]; coef = G.fit_ground_plane(near); ha = G.height_above(near, coef)
        nb = np.degrees(np.arctan2(near[:, 1], near[:, 0])); nr = np.hypot(near[:, 0], near[:, 1])
        print(f"\npair {PAIR}: max lidar range (any) {rng_all.max():.1f} m, {len(blobs)} glare blob(s)", flush=True)
        fig, axs = plt.subplots(1, len(blobs)+1, figsize=(7*(len(blobs)+1), 6))
        if len(blobs) == 1: axs = [axs[0], axs[1]] if hasattr(axs, '__len__') else axs
        axs = np.atleast_1d(axs)
        axs[0].imshow(img); axs[0].axis("off"); axs[0].set_title(f"{PAIR} camera")
        for i, b in enumerate(blobs, 1):
            bc = 0.5*(G.ray(b["u0"], b["vb"])[0] + G.ray(b["u1"], b["vb"])[0])
            onb = np.abs(((nb - bc + 180) % 360) - 180) < 5
            r = nr[onb]; h = ha[onb]
            carh = (h > 0.3) & (h < 2.0)
            print(f"  blob bearing {bc:+.1f}: {onb.sum()} pts on bearing; car-height(0.3-2.0m) {carh.sum()}; "
                  f"range span {r.min():.1f}-{r.max():.1f}m", flush=True)
            for lo in range(0, 45, 5):
                m = (r >= lo) & (r < lo+5); ch = m & carh
                if m.sum(): print(f"     {lo:2d}-{lo+5:2d}m: {m.sum():4d} pts, car-ht {int(ch.sum()):3d}", flush=True)
            ax = axs[min(i, len(axs)-1)]
            ax.scatter(r, h, s=3, c=np.where(carh, "red", "0.6"))
            ax.axhline(0.3, color="g", ls=":"); ax.axhline(2.0, color="g", ls=":")
            ax.set_xlim(0, 45); ax.set_ylim(-1, 8); ax.set_xlabel("range (m)"); ax.set_ylabel("height above plane (m)")
            ax.set_title(f"{PAIR} bearing{bc:+.0f}: red=car-height")
        out = f"{RES}/fusion/night_label/diag_range_{PAIR}.png"
        plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close(); print(f"  saved {out}", flush=True)

# ======== src/fusion/radar_first/diag_reject.py ========
# Why is the clear L-cluster rejected? For a pair, show lidar car-height clusters in a WIDE cone
# around the glare bearing and each gate's verdict (footprint / ground-touch / tall-above / range).
#     python3 diag_reject.py 14610
def dreject_main():
    import os, sys
    from pathlib import Path
    import numpy as np
    from scipy import ndimage
    from sklearn.cluster import DBSCAN
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import load_pairs, load_calib, LidarSource, LOCAL, RES
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
    PAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 14610
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
    BRIGHT = 205; VBAND = (0.20, 0.85); MIN_BLOB = 60; RMAX = 45.0; CONE = 15.0
    BRIGHT = 205; VBAND = (0.20, 0.85); MIN_BLOB = 60; RMAX = 45.0; CONE = 15.0
    BRIGHT = 205; VBAND = (0.20, 0.85); MIN_BLOB = 60; RMAX = 45.0; CONE = 15.0
    BRIGHT = 205; VBAND = (0.20, 0.85); MIN_BLOB = 60; RMAX = 45.0; CONE = 15.0
    BRIGHT = 205; VBAND = (0.20, 0.85); MIN_BLOB = 60; RMAX = 45.0; CONE = 15.0
    cal = load_calib(); ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
    cal = load_calib(); ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
    cal = load_calib(); ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
    cal = load_calib(); ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; H, W = ci["height"], ci["width"]
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; H, W = ci["height"], ci["width"]
    pairs = load_pairs(); bag_ts = np.load(CIDX)["color_bag_ts_ns"]
    pairs = load_pairs(); bag_ts = np.load(CIDX)["color_bag_ts_ns"]
    img = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[PAIR]["color_idx"]], Path(CACHE), "color", bag_ts)[pairs[PAIR]["color_idx"]]
    if img.ndim == 1: img = img.reshape(H, W, 3)
    bright = (img.min(2) > BRIGHT); bright[:int(H*VBAND[0])] = False; bright[int(H*VBAND[1]):] = False
    bright = (img.min(2) > BRIGHT); bright[:int(H*VBAND[0])] = False; bright[int(H*VBAND[1]):] = False
    bright = (img.min(2) > BRIGHT); bright[:int(H*VBAND[0])] = False; bright[int(H*VBAND[1]):] = False
    lab, n = ndimage.label(bright); bl = [np.where(lab == i)[1] for i in range(1, n+1)]
    lab, n = ndimage.label(bright); bl = [np.where(lab == i)[1] for i in range(1, n+1)]
    bl = [c for c in bl if len(c) >= MIN_BLOB]
    u = float(np.concatenate(bl).mean()) if bl else cx
    d = np.array([(u-cx)/fx, 0.0, 1.0]); d /= np.linalg.norm(d); dl = Rc2l @ d
    d = np.array([(u-cx)/fx, 0.0, 1.0]); d /= np.linalg.norm(d); dl = Rc2l @ d
    d = np.array([(u-cx)/fx, 0.0, 1.0]); d /= np.linalg.norm(d); dl = Rc2l @ d
    bearing = np.degrees(np.arctan2(dl[1], dl[0]))
    print(f"pair {PAIR}: glare bearing {bearing:+.1f} deg (search cone +-{CONE})", flush=True)
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < RMAX
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < RMAX
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < RMAX
    gz = float(np.percentile(P[nm][:, 2], 10))
    slab = P[nm & (P[:, 2] > gz+0.3) & (P[:, 2] < gz+2.2)]
    sb = np.degrees(np.arctan2(slab[:, 1], slab[:, 0]))
    onc = np.abs(((sb - bearing + 180) % 360) - 180) < CONE
    pts = slab[onc]
    print(f"  car-height pts in cone: {len(pts)}", flush=True)
    clusters = []
    if len(pts) >= 6:
        L = DBSCAN(eps=1.0, min_samples=4).fit_predict(pts[:, :2])
        for c in set(L.tolist()):
            if c == -1: continue
            cl = pts[L == c]
            if len(cl) < 6: continue
            cen = cl[:, :2].mean(0); X = cl[:, :2]-cen
            _, _, vt = np.linalg.svd(X, full_matrices=False); proj = X @ vt.T
            mj, mnr = float(np.ptp(proj[:, 0])), float(np.ptp(proj[:, 1]))
            r = float(np.hypot(*cen)); be = np.degrees(np.arctan2(cen[1], cen[0]))
            x0, x1 = cl[:, 0].min()-0.4, cl[:, 0].max()+0.4; y0, y1 = cl[:, 1].min()-0.4, cl[:, 1].max()+0.4
            above = int(((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1) & (P[:, 2] > gz+2.6)).sum())
            shape_ok = (mnr <= 3.5) and (1.0 <= mj <= 7.5)
            grnd_ok = float(cl[:, 2].min()) <= gz+0.8
            tall_ok = above < 8
            inwin = abs(((be-bearing+180) % 360)-180) < 5.0
            verdict = "KEPT" if (shape_ok and grnd_ok and tall_ok and inwin) else "reject:" + ",".join(
                ([] if shape_ok else ["shape"]) + ([] if grnd_ok else ["ground"]) + ([] if tall_ok else ["tall"]) + ([] if inwin else ["offbearing(>5deg)"]))
            clusters.append((cen, r, be, cl, verdict))
            print(f"    cl r{r:5.1f} bearing{be:+6.1f} n{len(cl):3d} fp {mj:.1f}x{mnr:.1f} z0{cl[:,2].min()-gz:+.2f} above{above:3d} -> {verdict}", flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(17, 8))
    axs[0].imshow(img); axs[0].axvline(u, color="lime"); axs[0].axis("off"); axs[0].set_title(f"{PAIR} glare u={u:.0f}")
    axs[0].imshow(img); axs[0].axvline(u, color="lime"); axs[0].axis("off"); axs[0].set_title(f"{PAIR} glare u={u:.0f}")
    axs[0].imshow(img); axs[0].axvline(u, color="lime"); axs[0].axis("off"); axs[0].set_title(f"{PAIR} glare u={u:.0f}")
    axs[0].imshow(img); axs[0].axvline(u, color="lime"); axs[0].axis("off"); axs[0].set_title(f"{PAIR} glare u={u:.0f}")
    axs[1].scatter(P[nm, 1], P[nm, 0], s=0.5, c="0.8")
    for bb, st in ((bearing-5, "lime"), (bearing+5, "lime"), (bearing-CONE, "orange"), (bearing+CONE, "orange")):
        rr = np.linspace(0, RMAX, 2); axs[1].plot(rr*np.sin(np.radians(bb)), rr*np.cos(np.radians(bb)), st, lw=0.8, ls="--")
    for cen, r, be, cl, verdict in clusters:
        col = "green" if verdict == "KEPT" else "red"
        axs[1].scatter(cl[:, 1], cl[:, 0], s=8, c=col)
        axs[1].text(cen[1], cen[0], verdict.split(":")[0], color=col, fontsize=8)
    axs[1].set_aspect("equal"); axs[1].set_xlim(24, -8); axs[1].set_ylim(0, RMAX)
    axs[1].set_aspect("equal"); axs[1].set_xlim(24, -8); axs[1].set_ylim(0, RMAX)
    axs[1].set_aspect("equal"); axs[1].set_xlim(24, -8); axs[1].set_ylim(0, RMAX)
    axs[1].set_xlabel("lateral Y"); axs[1].set_ylabel("forward X")
    axs[1].set_xlabel("lateral Y"); axs[1].set_ylabel("forward X")
    axs[1].set_title(f"{PAIR}: clusters in cone (green=kept, red=rejected); lime=+-5 window, orange=+-{CONE}")
    out = f"{RES}/fusion/night_label/diag_reject_{PAIR}.png"
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); print(f"saved {out}", flush=True)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); print(f"saved {out}", flush=True)
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches="tight"); print(f"saved {out}", flush=True)

# ======== src/fusion/radar_first/diag_vscan.py ========
# Lidar VERTICAL scan profile across a window: range (x) vs height-above-ground (y) for the forward
# glare-side cone. Shows the beam fan -- how vertical sampling falls off with range, the overhanging
# canopy, the road, and whether anything lands at CAR height at distance.
#     python3 diag_vscan.py 15450 15460 ...
def dvscan_main():
    import os, sys
    import numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from common import load_pairs, load_calib, LidarSource, RES
    import glare_bearing as G
    PAIRS = [int(a) for a in sys.argv[1:]] or [15450, 15480, 15500, 15520, 15560, 15600]
    pairs = load_pairs(); src = LidarSource()
    pairs = load_pairs(); src = LidarSource()
    CONE = 25.0                                        # deg half-cone toward the forward/glare side (+bearing)
    n = len(PAIRS); cols = 4; rows = int(np.ceil(n/cols))
    n = len(PAIRS); cols = 4; rows = int(np.ceil(n/cols))
    n = len(PAIRS); cols = 4; rows = int(np.ceil(n/cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows)); axs = np.atleast_1d(axs).ravel()
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows)); axs = np.atleast_1d(axs).ravel()
    for i, P in enumerate(PAIRS):
        xyz = src.xyz(pairs[P]["lidar_idx"]); rng = np.hypot(xyz[:, 0], xyz[:, 1])
        near = xyz[rng < G.RMAX]; coef = G.fit_ground_plane(near)
        bearing = np.degrees(np.arctan2(near[:, 1], near[:, 0]))
        cone = (bearing > 0) & (bearing < CONE)        # forward + left (glare side in this window)
        r = np.hypot(near[cone, 0], near[cone, 1]); h = G.height_above(near[cone], coef)
        ax = axs[i]
        carh = (h > 0.3) & (h < 2.0)
        ax.scatter(r[~carh], h[~carh], s=2, c="0.65")
        ax.scatter(r[carh], h[carh], s=4, c="red")     # red = car-height band
        ax.axhline(0.3, color="g", ls=":", lw=0.8); ax.axhline(2.0, color="g", ls=":", lw=0.8)
        ax.set_xlim(0, 45); ax.set_ylim(-1, 9)
        ax.set_title(f"{P}  (car-ht band red: {int(carh.sum())} pts)", fontsize=10)
        ax.set_xlabel("range (m)"); ax.set_ylabel("height above ground (m)")
    for j in range(n, len(axs)): axs[j].axis("off")
    out = f"{RES}/fusion/night_label/diag_vscan.png"
    plt.suptitle("W3 lidar vertical scan (forward glare-side cone): canopy above, road below, car-height band (red)", fontsize=12)
    plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight"); print(f"saved {out}", flush=True)
    plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight"); print(f"saved {out}", flush=True)
    plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight"); print(f"saved {out}", flush=True)

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'d14260_proc': d14260_proc_main, 'd14280': d14280_main, 'dblobs': dblobs_main, 'dgates': dgates_main, 'dglare_lidar': dglare_lidar_main, 'dglare_motion': dglare_motion_main, 'dmoving_match': dmoving_match_main, 'dradar_lidar': dradar_lidar_main, 'drange': drange_main, 'dreject': dreject_main, 'dvscan': dvscan_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
