#!/usr/bin/env python3
"""Depth-FREE camera<->lidar calibration: pick in the RGB image + lidar range-view, solve by PnP.

Why: D435 depth is dead beyond ~6 m, so the depth-based picker can't use far features (stop signs,
poles) even though the camera AND lidar see them clearly. PnP needs only a 2D pixel + a 3D lidar point,
no depth -- so you pick the feature in the RGB image (left) and the SAME feature in the lidar range-view
(right), across several frames, and it solves the camera extrinsic.

Run (from repo root):
    python3 sync/src/calibration/pnp_pick_rgb.py --pairs 41023 39015 26874 16936 38990 ...

Controls:
    - click the feature in the RGB (left) THEN the same feature in the lidar range-view (right)  = one pick
    - n / p : next / previous frame        u : undo last pick        f : flip lidar left-right
    - s     : solve PnP + preview (writes nothing)      w : solve + WRITE camera_to_lidar_auto      q : quit
Pick ~8-12 features spread across azimuth, elevation, range (the round sign panel is ideal).
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "sync/src/camera_projection"))
sys.path.insert(0, os.path.join(REPO, "sync/src/fusion/radar_first"))
from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC
from common import load_pairs, load_calib, LidarSource, LOCAL, RES

CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
CALJSON = f"{RES}/calibration.json"
CORR = f"{RES}/correspondences/pnp_camera_lidar.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", required=True)
    ap.add_argument("--fov", type=float, default=40.0, help="lidar range-view half-FOV (deg) shown")
    args = ap.parse_args()

    pairs = load_pairs(); cal = load_calib(); ci = cal["color_intrinsics"]
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; H, W = ci["height"], ci["width"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    C2L_cur = np.array(cal["camera_to_lidar_auto"])
    bag_ts = np.load(CIDX)["color_bag_ts_ns"]
    src = LidarSource()
    corr = json.load(open(CORR))["points"] if os.path.exists(CORR) else []

    state = {"i": 0, "pending": None, "flip": True, "img": None, "P": None, "az": None, "el": None, "rng": None}

    def load(i):
        F = args.pairs[i]
        im = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[F]["color_idx"]], Path(CACHE), "color", bag_ts).get(pairs[F]["color_idx"])
        if im is not None and im.ndim == 1: im = im.reshape(H, W, 3)
        P = src.xyz(pairs[F]["lidar_idx"]); P = P[np.hypot(P[:, 0], P[:, 1]) > 1.0]
        az = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
        el = np.degrees(np.arctan2(P[:, 2], np.hypot(P[:, 0], P[:, 1])))
        rng = np.hypot(P[:, 0], P[:, 1])
        m = np.abs(az) < args.fov
        state.update(img=im, P=P[m], az=az[m], el=el[m], rng=rng[m])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))

    def redraw():
        F = args.pairs[state["i"]]
        axL.clear(); axR.clear()
        if state["img"] is not None: axL.imshow(state["img"])
        axL.set_xlim(0, W); axL.set_ylim(H, 0)
        axL.set_title(f"[{state['i']+1}/{len(args.pairs)}] pair {F} -- RGB: click the feature")
        sgn = -1 if state["flip"] else 1
        axR.scatter(sgn * state["az"], state["el"], c=state["rng"], s=6, cmap="jet", vmin=0, vmax=40)
        axR.set_xlim(sgn * args.fov, -sgn * args.fov); axR.set_ylim(-25, 30)
        axR.set_xlabel("azimuth (deg)"); axR.set_ylabel("elevation (deg)")
        axR.set_title("LIDAR range-view: click the SAME feature  ('f' flips L-R)")
        for c in corr:
            if c["pair"] != F: continue
            axL.plot(*c["pixel"], "w+", ms=12, mew=2)
            a = np.degrees(np.arctan2(c["lidar"][1], c["lidar"][0])); e = np.degrees(np.arctan2(c["lidar"][2], np.hypot(c["lidar"][0], c["lidar"][1])))
            axR.plot(sgn * a, e, "w+", ms=12, mew=2)
        n_here = sum(c["pair"] == F for c in corr)
        fig.suptitle(f"total picks: {len(corr)}  (this frame: {n_here})   pending-pixel: {'yes' if state['pending'] else 'no'}")
        fig.canvas.draw_idle()

    def onclick(ev):
        if ev.inaxes is axL and ev.xdata is not None:
            state["pending"] = [float(ev.xdata), float(ev.ydata)]
            print(f"  RGB pixel ({ev.xdata:.0f},{ev.ydata:.0f}) -> now click the same feature in the lidar panel", flush=True)
            redraw()
        elif ev.inaxes is axR and ev.xdata is not None and state["pending"] is not None:
            sgn = -1 if state["flip"] else 1
            d = np.hypot(sgn * state["az"] - ev.xdata, state["el"] - ev.ydata)
            j = int(np.argmin(d)); p3 = state["P"][j]
            corr.append({"pixel": state["pending"], "lidar": [float(p3[0]), float(p3[1]), float(p3[2])], "pair": args.pairs[state["i"]]})
            print(f"  + pick: px{[round(x) for x in state['pending']]} <-> lidar({p3[0]:.1f},{p3[1]:.1f},{p3[2]:.1f})  range {np.hypot(p3[0],p3[1]):.1f}m   [total {len(corr)}]", flush=True)
            state["pending"] = None; redraw()

    def solve(write):
        if len(corr) < 6:
            print(f"need >=6 picks (have {len(corr)})"); return
        obj = np.array([c["lidar"] for c in corr], float); img = np.array([c["pixel"] for c in corr], float)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=5.0, iterationsCount=3000, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None: print("PnP failed"); return
        inl = inl.ravel(); Rm, _ = cv2.Rodrigues(rvec); L2C = np.eye(4); L2C[:3, :3] = Rm; L2C[:3, 3] = tvec[:, 0]; C2L = np.linalg.inv(L2C)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None); proj = proj.reshape(-1, 2)
        rmse = float(np.sqrt(((proj[inl] - img[inl]) ** 2).sum(1).mean()))
        rot = float(np.degrees(np.arccos(np.clip((np.trace(C2L[:3, :3] @ C2L_cur[:3, :3].T) - 1) / 2, -1, 1))))
        t = np.linalg.norm(C2L[:3, 3])
        print(f"\nPnP: {len(inl)}/{len(corr)} inliers  reproj RMSE {rmse:.1f}px  |t|={t*100:.1f}cm  rot-vs-current {rot:.2f}deg")
        json.dump({"kind": "pnp_camera_lidar", "points": corr}, open(CORR, "w"), indent=1)
        healthy = rmse < 6 and 0.05 < t < 0.8 and rot < 20
        if not write:
            print("preview only (press 'w' to WRITE). health:", "OK" if healthy else "FAIL (not writing even on w)"); return
        if not healthy:
            print("REFUSING to write: health check failed (RMSE/|t|/rotation out of range). Re-pick."); return
        c = json.load(open(CALJSON)); import shutil
        shutil.copy(CALJSON, f"{RES}/backups/calibration.pre_pnp.bak.json")
        c["camera_to_lidar_auto"] = C2L.tolist()
        c["camera_to_lidar_auto_meta"] = {"solved": True, "method": "RGB-pixel<->lidar 2D-3D PnP (depth-free)", "n_inliers": int(len(inl)), "reproj_rmse_px": round(rmse, 2)}
        json.dump(c, open(CALJSON, "w"), indent=1)
        print(f"WROTE camera_to_lidar_auto (backup -> backups/calibration.pre_pnp.bak.json). Re-render overlays to verify.")

    def onkey(ev):
        if ev.key == "n": state["i"] = (state["i"] + 1) % len(args.pairs); state["pending"] = None; load(state["i"]); redraw()
        elif ev.key == "p": state["i"] = (state["i"] - 1) % len(args.pairs); state["pending"] = None; load(state["i"]); redraw()
        elif ev.key == "u" and corr: c = corr.pop(); print("undo", c["pair"]); redraw()
        elif ev.key == "f": state["flip"] = not state["flip"]; redraw()
        elif ev.key == "s": solve(False)
        elif ev.key == "w": solve(True)
        elif ev.key == "q": plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    print(__doc__)
    load(0); redraw(); plt.show()


if __name__ == "__main__":
    main()
