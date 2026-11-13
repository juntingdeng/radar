"""Radar-first box proposal (merged). Subcommands: boxes | rv"""
import sys

# ======== src/fusion/radar_first/propose_boxes.py ========
# Radar-FIRST object proposals -> LiDAR cross-check -> oriented boxes.
# 
# Fix for the mover-only mistake: PROPOSE from radar range-azimuth STRONG RETURNS
# (static + moving objects, via the production BEV-peak extractor), then snap each
# proposal to the nearest LiDAR object cluster and raise an oriented box from it.
# Parked cars are now proposable (they have no Doppler but do have a strong return).
# If LiDAR is also degraded, the proposal still yields a radar-sized box.
# 
# Verifies on pair 13890 by projecting boxes onto camera + LiDAR BEV.
def boxes_main():
    import os, sys
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from common import load_pairs, load_calib, LidarSource, LOCAL, RES, H5, CFG
    from odometry import odometry_scan2map  # not used here but keeps module import stable
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import _radar_setup, _radar_panel, _peaks_from_bev, _extract_frames, DEFAULT_COLOR_TOPIC
    PAIR = 13890
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAMERA_BAG = f"{LOCAL}/camera/20260511_213724.bag"
    CAMERA_INDEX = f"{RES}/camera_index.npz"
    SNAP = 2.0           # radar peak must be within this of a LiDAR cluster centroid to confirm
    PCTL = 99.0          # radar BEV strong-return percentile
    pairs = load_pairs()
    cal = load_calib()
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    args = SimpleNamespace(radar_h5=H5, cfg_file=CFG, sync_csv=f"{RES}/sync_pairs.csv",
                           sync_summary=f"{RES}/sync_summary.json", lidar_pcap=None, lidar_metadata=None,
                           radar_bev_res=0.1)
    loader, Rmat = _radar_setup(args, cal, doppler_mode="all", processing="legacy")
    bev, la, fa = _radar_panel(loader, PAIR)
    pf, pl = _peaks_from_bev(bev, la, fa, Rmat, PCTL)          # lidar-frame (forward, lateral)
    peaks = np.c_[pf, pl]
    print(f"radar proposals (BEV peaks): {len(peaks)}", flush=True)
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"])
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"])
    near = np.linalg.norm(P[:, :2], axis=1) < 30
    gz = np.percentile(P[near][:, 2], 3)
    obj = P[near & (P[:, 2] > gz + 0.3) & (P[:, 2] < gz + 2.2)]
    clusters = []
    if len(obj) > 30:
        lab = DBSCAN(eps=0.7, min_samples=8).fit_predict(obj[:, :2])
        for c in set(lab):
            if c == -1: continue
            pts = obj[lab == c]
            if not (15 <= len(pts) <= 3000): continue
            fp = np.ptp(pts[:, :2], axis=0)
            if 0.8 <= max(fp) <= 7 and min(fp) <= 4:
                clusters.append(pts)
    print(f"lidar clusters: {len(clusters)}", flush=True)
    def fit_box(pts):
        xy = pts[:, :2]; c = xy.mean(0); X = xy - c
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        yaw = np.arctan2(vt[0, 1], vt[0, 0])
        proj = X @ vt.T
        L = max(float(np.ptp(proj[:, 0])), 1.0); Wd = max(float(np.ptp(proj[:, 1])), 0.8)
        z0, z1 = float(pts[:, 2].min()), float(pts[:, 2].max())
        return dict(cx=c[0], cy=c[1], yaw=yaw, L=L, W=max(Wd, 1.4), z0=z0, z1=max(z1, z0 + 1.3))
    boxes = []; used = set()
    boxes = []; used = set()
    for p in peaks:
        if not clusters:
            break
        cents = np.array([cl[:, :2].mean(0) for cl in clusters])
        dd = np.hypot(cents[:, 0] - p[0], cents[:, 1] - p[1])
        j = int(np.argmin(dd))
        if dd[j] <= SNAP and j not in used:
            used.add(j); b = fit_box(clusters[j]); b["src"] = "radar->lidar"; boxes.append(b)
    print(f"confirmed boxes (radar peak snapped to lidar cluster): {len(boxes)}", flush=True)
    def corners3d(b):
        dx = np.array([b["L"]/2, b["L"]/2, -b["L"]/2, -b["L"]/2])
        dy = np.array([b["W"]/2, -b["W"]/2, -b["W"]/2, b["W"]/2])
        cz, s = np.cos(b["yaw"]), np.sin(b["yaw"])
        fx = b["cx"] + dx*cz - dy*s; fy = b["cy"] + dx*s + dy*cz
        bot = np.c_[fx, fy, np.full(4, b["z0"])]; top = np.c_[fx, fy, np.full(4, b["z1"])]
        return bot, top
    bag_ts = np.load(CAMERA_INDEX)["color_bag_ts_ns"]; cidx = pairs[PAIR]["color_idx"]
    bag_ts = np.load(CAMERA_INDEX)["color_bag_ts_ns"]; cidx = pairs[PAIR]["color_idx"]
    img = _extract_frames(CAMERA_BAG, DEFAULT_COLOR_TOPIC, [cidx], Path(CACHE), "color", bag_ts).get(cidx)
    if img is not None and img.ndim == 1:
        img = img.reshape(ci["height"], ci["width"], 3)
    def to_px(xyz):
        opt = (L2C @ np.r_[xyz, 1])[:3]; Z = opt[2]
        if Z <= 0.2: return None
        return np.array([ci["fx"]*opt[0]/Z + ci["cx"], ci["fy"]*opt[1]/Z + ci["cy"]])
    fig, axs = plt.subplots(1, 2, figsize=(19, 7.5))
    if img is not None:
        axs[0].imshow(img)
    for b in boxes:
        bot, top = corners3d(b)
        pb = [to_px(c) for c in bot]; pt = [to_px(c) for c in top]
        if any(x is None for x in pb + pt):
            continue
        pb = np.array(pb); pt = np.array(pt)
        for a in (pb, pt):
            axs[0].plot(np.r_[a[:, 0], a[0, 0]], np.r_[a[:, 1], a[0, 1]], "lime", lw=2)
        for k in range(4):
            axs[0].plot([pb[k, 0], pt[k, 0]], [pb[k, 1], pt[k, 1]], "lime", lw=2)
    axs[0].set_xlim(0, ci["width"]); axs[0].set_ylim(ci["height"], 0); axs[0].axis("off")
    axs[0].set_xlim(0, ci["width"]); axs[0].set_ylim(ci["height"], 0); axs[0].axis("off")
    axs[0].set_xlim(0, ci["width"]); axs[0].set_ylim(ci["height"], 0); axs[0].axis("off")
    axs[0].set_title(f"Radar-first proposals -> LiDAR boxes on CAMERA (pair {PAIR})\n{len(boxes)} boxes")
    ml = np.linalg.norm(P[:, :2], axis=1) < 30
    axs[1].scatter(P[ml, 1], P[ml, 0], s=0.5, c="0.8")
    axs[1].scatter(peaks[:, 1], peaks[:, 0], s=40, marker="x", c="purple", label="radar proposal (BEV peak)")
    for b in boxes:
        bot, _ = corners3d(b)
        axs[1].plot(np.r_[bot[:, 1], bot[0, 1]], np.r_[bot[:, 0], bot[0, 0]], "lime", lw=2)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 28)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 28)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 28)
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
    axs[1].set_title(f"BEV: radar proposals (x) + raised boxes\n{len(peaks)} peaks, {len(boxes)} boxes"); axs[1].legend()
    axs[1].set_title(f"BEV: radar proposals (x) + raised boxes\n{len(peaks)} peaks, {len(boxes)} boxes"); axs[1].legend()
    plt.tight_layout()
    out = f"{RES}/fusion/radar_first_boxes.png"
    plt.savefig(out, dpi=115, bbox_inches="tight")
    print(f"saved {out}", flush=True)

# ======== src/fusion/radar_first/propose_boxes_rv.py ========
# Radar-first proposals via RANGE + VELOCITY association (not XY snap).
# 
# Radar is precise in RANGE and VELOCITY but coarse in azimuth. So match each LiDAR
# cluster to a radar detection by: |range| tight, azimuth LOOSE -> uses radar's strong
# axes and LiDAR's bearing. Velocity (from Doppler) is attached + used to tag moving.
# Raises an oriented box on every LiDAR cluster a radar detection corroborates.
# 
# Verifies on pair 13890: camera + BEV with boxes.
def rv_main():
    import os, sys
    from pathlib import Path
    import numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN
    from common import load_pairs, load_calib, LidarSource, LOCAL, RES
    from detector import RadarDetector
    sys.path.insert(0, "sync/src/camera_projection")
    from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
    PAIR = 13890
    CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
    CAMERA_BAG = f"{LOCAL}/camera/20260511_213724.bag"; CAMERA_INDEX = f"{RES}/camera_index.npz"
    CAMERA_BAG = f"{LOCAL}/camera/20260511_213724.bag"; CAMERA_INDEX = f"{RES}/camera_index.npz"
    RANGE_TOL = 1.5      # m  -- radar range is precise, so tight
    AZ_TOL = 20.0        # deg -- radar azimuth is coarse, so loose
    pairs = load_pairs()
    cal = load_calib()
    R2L = np.array(cal["radar_to_lidar_auto"])
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    C2L = np.array(cal["camera_to_lidar_auto"]); L2C = np.linalg.inv(C2L); ci = cal["color_intrinsics"]
    A = R2L[:2, :2]; t = R2L[:2, 3]; A_inv = np.linalg.inv(A)   # lidar-xy -> radar-xy
    A = R2L[:2, :2]; t = R2L[:2, 3]; A_inv = np.linalg.inv(A)   # lidar-xy -> radar-xy
    A = R2L[:2, :2]; t = R2L[:2, 3]; A_inv = np.linalg.inv(A)   # lidar-xy -> radar-xy
    det = RadarDetector(processing="legacy")
    d, _ = det.detect(pairs[PAIR]["radar_idx"])
    Xd, Yd = RadarDetector.to_xy(d)
    det_rng = np.hypot(Xd, Yd); det_ang = np.degrees(np.arctan2(Yd, Xd))
    det_rng = np.hypot(Xd, Yd); det_ang = np.degrees(np.arctan2(Yd, Xd))
    print(f"radar detections: {len(d)}", flush=True)
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"])
    src = LidarSource(); P = src.xyz(pairs[PAIR]["lidar_idx"])
    near = np.linalg.norm(P[:, :2], axis=1) < 30
    gz = np.percentile(P[near][:, 2], 3)
    obj = P[near & (P[:, 2] > gz + 0.3) & (P[:, 2] < gz + 2.2)]
    clusters = []
    lab = DBSCAN(eps=0.7, min_samples=8).fit_predict(obj[:, :2]) if len(obj) > 30 else np.array([])
    for c in set(lab.tolist()):
        if c == -1: continue
        pts = obj[lab == c]
        if not (15 <= len(pts) <= 3000): continue
        fp = np.ptp(pts[:, :2], axis=0)
        if 0.8 <= max(fp) <= 7 and min(fp) <= 4:
            clusters.append(pts)
    print(f"lidar clusters: {len(clusters)}", flush=True)
    def fit_box(pts):
        xy = pts[:, :2]; c = xy.mean(0); X = xy - c
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        yaw = np.arctan2(vt[0, 1], vt[0, 0]); proj = X @ vt.T
        return dict(cx=c[0], cy=c[1], yaw=yaw,
                    L=max(float(np.ptp(proj[:, 0])), 1.0), W=max(float(np.ptp(proj[:, 1])), 1.4),
                    z0=float(pts[:, 2].min()), z1=max(float(pts[:, 2].max()), float(pts[:, 2].min()) + 1.3))
    boxes = []
    for cl in clusters:
        cxy = cl[:, :2].mean(0)
        rxy = A_inv @ (cxy - t)                      # cluster centroid in radar frame
        crng = np.hypot(*rxy); cang = np.degrees(np.arctan2(rxy[1], rxy[0]))
        dr = np.abs(det_rng - crng); da = np.abs(((det_ang - cang + 180) % 360) - 180)
        hit = np.where((dr < RANGE_TOL) & (da < AZ_TOL))[0]
        if not len(hit):
            continue
        j = hit[np.argmax(np.abs(d["vel"][hit]))]    # prefer the moving match
        b = fit_box(cl); b["vel"] = float(d["vel"][j]); b["moving"] = abs(b["vel"]) > 0.3
        boxes.append(b)
    print(f"radar-confirmed boxes (range+vel assoc): {len(boxes)}  "
          f"(moving {sum(b['moving'] for b in boxes)})", flush=True)
    def corners3d(b):
        dx = np.array([b["L"]/2, b["L"]/2, -b["L"]/2, -b["L"]/2]); dy = np.array([b["W"]/2, -b["W"]/2, -b["W"]/2, b["W"]/2])
        cs, sn = np.cos(b["yaw"]), np.sin(b["yaw"])
        fx = b["cx"] + dx*cs - dy*sn; fy = b["cy"] + dx*sn + dy*cs
        return np.c_[fx, fy, np.full(4, b["z0"])], np.c_[fx, fy, np.full(4, b["z1"])]
    def to_px(xyz):
        opt = (L2C @ np.r_[xyz, 1])[:3]
        if opt[2] <= 0.2: return None
        return np.array([ci["fx"]*opt[0]/opt[2] + ci["cx"], ci["fy"]*opt[1]/opt[2] + ci["cy"]])
    bag_ts = np.load(CAMERA_INDEX)["color_bag_ts_ns"]; cidx = pairs[PAIR]["color_idx"]
    bag_ts = np.load(CAMERA_INDEX)["color_bag_ts_ns"]; cidx = pairs[PAIR]["color_idx"]
    img = _extract_frames(CAMERA_BAG, DEFAULT_COLOR_TOPIC, [cidx], Path(CACHE), "color", bag_ts).get(cidx)
    if img is not None and img.ndim == 1:
        img = img.reshape(ci["height"], ci["width"], 3)
    fig, axs = plt.subplots(1, 2, figsize=(19, 7.5))
    if img is not None:
        axs[0].imshow(img)
    for b in boxes:
        bot, top = corners3d(b); pb = [to_px(c) for c in bot]; pt = [to_px(c) for c in top]
        if any(x is None for x in pb + pt): continue
        pb = np.array(pb); pt = np.array(pt); col = "lime" if b["moving"] else "cyan"
        for a in (pb, pt): axs[0].plot(np.r_[a[:, 0], a[0, 0]], np.r_[a[:, 1], a[0, 1]], col, lw=2)
        for k in range(4): axs[0].plot([pb[k, 0], pt[k, 0]], [pb[k, 1], pt[k, 1]], col, lw=2)
    axs[0].set_xlim(0, ci["width"]); axs[0].set_ylim(ci["height"], 0); axs[0].axis("off")
    axs[0].set_xlim(0, ci["width"]); axs[0].set_ylim(ci["height"], 0); axs[0].axis("off")
    axs[0].set_xlim(0, ci["width"]); axs[0].set_ylim(ci["height"], 0); axs[0].axis("off")
    axs[0].set_title(f"Radar-first (range+vel) -> LiDAR boxes · CAMERA (pair {PAIR})\n"
                     f"{len(boxes)} boxes (lime=moving, cyan=static)")
    ml = np.linalg.norm(P[:, :2], axis=1) < 30
    axs[1].scatter(P[ml, 1], P[ml, 0], s=0.5, c="0.8")
    axs[1].scatter(Yd, Xd, s=16, marker="x", c="purple", label="radar det")
    for b in boxes:
        bot, _ = corners3d(b); col = "lime" if b["moving"] else "cyan"
        axs[1].plot(np.r_[bot[:, 1], bot[0, 1]], np.r_[bot[:, 0], bot[0, 0]], col, lw=2)
        axs[1].text(b["cy"], b["cx"], f"{b['vel']:+.1f}", fontsize=8, color=col)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 28)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 28)
    axs[1].set_aspect("equal"); axs[1].set_xlim(15, -15); axs[1].set_ylim(0, 28)
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
    axs[1].set_xlabel("lateral Y (m)"); axs[1].set_ylabel("forward X (m)")
    axs[1].set_title(f"BEV: radar det (x) + range+vel-confirmed boxes ({len(boxes)})"); axs[1].legend()
    axs[1].set_title(f"BEV: radar det (x) + range+vel-confirmed boxes ({len(boxes)})"); axs[1].legend()
    plt.tight_layout()
    out = f"{RES}/fusion/radar_first_boxes_rv.png"
    plt.savefig(out, dpi=115, bbox_inches="tight")
    print(f"saved {out}", flush=True)

if __name__ == '__main__':
    _c = sys.argv[1] if len(sys.argv) > 1 else ''
    _m = {'boxes': boxes_main, 'rv': rv_main}
    if _c not in _m:
        print(__doc__); sys.exit(1)
    sys.argv = [sys.argv[0] + ' ' + _c] + sys.argv[2:]
    _m[_c]()
