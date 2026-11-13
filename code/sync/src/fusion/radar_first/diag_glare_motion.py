"""Does ego-compensated radar DOPPLER isolate the moving glare-blind cars?

Overlays, per pair: ego-Doppler MOVING radar dets (red) vs STATIC (gray) + the glare bearing
cone + the current glare_bearing boxes. If the real cars = moving radar and the wrong boxes sit
on static/nothing, then a motion gate fixes it. Runs several pairs in one radar open.
    python3 diag_glare_motion.py 15510 15590
"""
import os, sys, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load_pairs, load_calib, LidarSource, LOCAL, RES, H5, CFG
from detector import RadarDetector
from ego_doppler import fit_ego_velocity
sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC

PAIRS = [int(a) for a in sys.argv[1:]] or [15510, 15590]
CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
BRIGHT = 205; VBAND = (0.30, 0.78); MIN_BLOB = 60; MOVER_MARGIN = 0.8; RMAX = 45.0

pairs = load_pairs(); cal = load_calib()
ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
R2L = np.array(cal["radar_to_lidar_auto"])
fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; H, W = ci["height"], ci["width"]
bag_ts = np.load(CIDX)["color_bag_ts_ns"]
imgs = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[p]["color_idx"] for p in PAIRS], Path(CACHE), "color", bag_ts)
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
