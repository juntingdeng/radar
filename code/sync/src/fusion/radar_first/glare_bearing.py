"""Glare-blind night-car detector (generalizable, moving + stopped).

PRINCIPLE: GLARE is the discriminator (car vs background -- clutter emits no glare); radar/lidar
only LOCALIZE. Every box is anchored to a discrete glare source; nothing is raised from lidar-shape
in a cone. The only non-car glare is lights -> rejected by TWO above-ground gates:
  (1) glare ray elevation <= ELEV_MAX deg above horizon (headlight is ~level, a traffic/street light
      points up), AND
  (2) a car-height ground object (LOCAL-ground slab, 0.3-2.2 m) exists along the glare bearing.
Range/geometry from that lidar cluster; confidence by corroboration:
  moving (ego-Doppler) > strong radar > lidar-only.
Works for moving AND stopped cars (headlights on). One mechanism, no per-window recipe.
    python3 glare_bearing.py 15450 15460 ...        # many pairs, one radar/lidar/camera open
"""
import os, sys, json
from pathlib import Path
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
from common import load_pairs, load_calib, LidarSource, LOCAL, RES
from detector import RadarDetector
from ego_doppler import fit_ego_velocity
sys.path.insert(0, "sync/src/camera_projection")
from yolo_to_lidar import _extract_frames, DEFAULT_COLOR_TOPIC

PAIRS = [int(a) for a in sys.argv[1:]] or [15450, 15460]
LOW_SLICE = os.environ.get("GLARE_LOWSLICE", "0") == "1"   # dense-foliage regime: trust glare bearing, drop the
# tall/isolation gates (a car under a tree has canopy above + foliage around), and require RADAR to pick the
# car (metal -> strong return) out of the car+foliage merge. Keeps car-footprint + sits-on-plane.
OUT_TAG = str(PAIRS[0]) if len(PAIRS) == 1 else f"win{PAIRS[0]}_{PAIRS[-1]}"
CACHE = os.path.expanduser("~/.cache/ti_setup_frames/f0fff15992")
CAM = f"{LOCAL}/camera/20260511_213724.bag"; CIDX = f"{RES}/camera_index.npz"
BRIGHT = 205; VBAND = (0.20, 0.85); MIN_BLOB = 60     # glare detection
RMAX = 45.0                                           # lidar near mask (sensor range), not a car threshold
BEARING_TOL = 10.0                                    # deg: glare bloom centroid is offset from the car -> wide cone;
#                                                      the car MODEL + ground plane + isolation reject clutter, not bearing
DEDUP_M = 3.0                                         # ~one car length: merge boxes of the same car
PAIR_DU, PAIR_DV = 70, 30                             # px: merge the two headlights of ONE car
MOVER_MARGIN = 0.8                                    # m/s off the ego-static line = a true mover (radar tier only)
CAR_MIN_PTS = 6
# --- physical passenger-vehicle model: fixed real-world dimensions, NOT per-frame tuning. Clutter is
#     rejected because it does not MATCH this model + does not SIT on the self-fit ground plane + is not
#     ISOLATED -- no range/count cutoffs. ---
CAR_LEN = (2.0, 6.0)   # OBSERVED footprint major (m): >=2 (partial/far), <=6 (large car + margin). A pole/hydrant
#                        has major < 2 -> this length gate rejects it, so the width bound below can stay loose.
CAR_WID = (0.4, 2.8)   # OBSERVED footprint minor (m): as low as ~0.4 when a car is seen edge-on (one face only),
#                        up to ~2.8 with mirrors/noise. (Real width is 1.6-2.1; the observed cluster is noisier.)
GROUND_CLEAR = 1.0     # lowest return within this of the fitted ROAD PLANE. A far car's lower body isn't hit, so
#                        its lowest return sits ~0.6-0.9 m up; a tall bank/foliage still separates via TALL/ISO.
TALL_H = 2.6           # returns above this (over the plane) = tree / pole / streetlight, not a car
SPLIT_LEN = 6.0        # footprint longer than one car -> adjacent cars fused -> split along the major axis
MAX_LEN = 13.0         # longer than this -> wall / guardrail, drop
ISO_RATIO = 0.6        # car is ISOLATED: surrounding car-height density < this x the cluster's (a bank is continuous)
DENSE_MIN = 60         # car-height returns in a 2.5 m range window on the glare bearing = a SOLID object (car).
#                        Fallback for the CLOSE regime where a car blends with a roadside wall/canopy and so fails
#                        TALL/ISO: glare says a car is there, density (not isolation) localizes it. Background on a
#                        bearing is sparse (< this) so it never fires -- the discriminator is glare + density.
DENSE_RMAX = 20.0      # only the close/blended regime; far cars go through the gated isolation path above
DENSE_RMIN = 6.0       # below this is EGO-ADJACENT curb/wall/embankment (always dense on the right) -> not a car
DENSE_WID = 3.0        # the dense window must have a CAR-sized minor extent; a wall swath sprawls wider -> reject

pairs = load_pairs(); cal = load_calib()
ci = cal["color_intrinsics"]; C2L = np.array(cal["camera_to_lidar_auto"]); Rc2l = C2L[:3, :3]
R2L = np.array(cal["radar_to_lidar_auto"])
fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]; H, W = ci["height"], ci["width"]
bag_ts = np.load(CIDX)["color_bag_ts_ns"]
imgs = _extract_frames(CAM, DEFAULT_COLOR_TOPIC, [pairs[p]["color_idx"] for p in PAIRS], Path(CACHE), "color", bag_ts)
det = RadarDetector(processing="legacy")             # all returns; ego-Doppler splits moving/static below
src = LidarSource()


def ray(u, v):
    """camera pixel -> (bearing_deg, elevation_deg) in the lidar frame (x fwd, y lat, z up)."""
    d = np.array([(u-cx)/fx, (v-cy)/fy, 1.0]); d /= np.linalg.norm(d); dl = Rc2l @ d
    return np.degrees(np.arctan2(dl[1], dl[0])), np.degrees(np.arctan2(dl[2], np.hypot(dl[0], dl[1])))


def car_glare_blobs(img):
    """Distinct glare sources; merge ONLY the two headlights of one car (tight), not across cars."""
    bright = (img.min(axis=2) > BRIGHT)
    bright[: int(H*VBAND[0])] = False; bright[int(H*VBAND[1]):] = False
    lab, nlab = ndimage.label(bright)
    bl = []
    for i in range(1, nlab+1):
        ys, xs = np.where(lab == i)
        if len(xs) < MIN_BLOB: continue
        # vb = BOTTOM edge of the glare (near the true light; the bloom/halo extends UPWARD, so the
        # centroid sits above the source and over-estimates elevation -> use the bottom instead)
        bl.append(dict(u=xs.mean(), v=ys.mean(), vb=float(np.percentile(ys, 85)), u0=xs.min(), u1=xs.max(), n=len(xs)))
    bl.sort(key=lambda b: b["u"]); merged = []
    for b in bl:
        m = merged[-1] if merged else None
        if m and abs(b["v"]-m["v"]) < PAIR_DV and b["u0"]-m["u1"] < PAIR_DU:   # same car's two headlights
            w = m["n"]+b["n"]
            m["u"] = (m["u"]*m["n"]+b["u"]*b["n"])/w; m["v"] = (m["v"]*m["n"]+b["v"]*b["n"])/w
            m["vb"] = max(m["vb"], b["vb"]); m["u1"] = max(m["u1"], b["u1"]); m["n"] = w
        else:
            merged.append(b)
    return merged


def fit_ground_plane(pts):
    """Robust ground PLANE  z = a*x + b*y + c  (self-calibrating; replaces a flat constant).

    Seeded from the lower half, then iterated keeping only points near the plane -> converges to the
    dominant flat road surface. An embankment/slope RISES above this plane, so its chunks fail the
    'sits on the plane' test by GEOMETRY -- no distance cutoff needed, works on slopes too.
    """
    lo = pts[pts[:, 2] < np.median(pts[:, 2])]
    A = np.c_[lo[:, 0], lo[:, 1], np.ones(len(lo))]
    coef, *_ = np.linalg.lstsq(A, lo[:, 2], rcond=None)
    for _ in range(4):
        resid = pts[:, 2] - (pts[:, 0]*coef[0] + pts[:, 1]*coef[1] + coef[2])
        inl = np.abs(resid) < 0.3                                   # 0.3 m = lidar/road roughness band
        if inl.sum() < 50: break
        A = np.c_[pts[inl, 0], pts[inl, 1], np.ones(int(inl.sum()))]
        coef, *_ = np.linalg.lstsq(A, pts[inl, 2], rcond=None)
    return coef


def height_above(xyz, coef):
    return xyz[:, 2] - (xyz[:, 0]*coef[0] + xyz[:, 1]*coef[1] + coef[2])


def detect(PAIR):
    img = imgs.get(pairs[PAIR]["color_idx"])
    if img is None: print(f"pair {PAIR}: no camera frame", flush=True); return []
    if img.ndim == 1: img = img.reshape(H, W, 3)
    blobs = car_glare_blobs(img)
    # radar detections -> lidar frame + ego-Doppler moving mask
    d, _ = det.detect(pairs[PAIR]["radar_idx"]); Xd, Yd = RadarDetector.to_xy(d)
    rf = R2L[0,0]*Xd + R2L[0,1]*Yd + R2L[0,3]; rl = R2L[1,0]*Xd + R2L[1,1]*Yd + R2L[1,3]
    rrng = np.hypot(rf, rl); rbear = np.degrees(np.arctan2(rl, rf)); rpow = d["power"]
    ab, inl = fit_ego_velocity(d["az"], d["vel"])
    if ab is not None:
        pr = np.cos(np.radians(-d["az"]))*ab[0] + np.sin(np.radians(-d["az"]))*ab[1]
        mover = (~inl) & (np.abs(d["vel"] - pr) > MOVER_MARGIN)
    else:
        mover = np.zeros(len(d), bool)
    powhi = np.percentile(rpow, 75) if len(rpow) else 0.0
    # self-calibrating ground plane -> car-height slab defined RELATIVE to it (handles slope/embankment)
    P = src.xyz(pairs[PAIR]["lidar_idx"]); nm = np.linalg.norm(P[:, :2], axis=1) < RMAX
    P = P[nm]; coef = fit_ground_plane(P); Pha = height_above(P, coef)
    slab = P[(Pha > 0.3) & (Pha < 2.2)]; sha = Pha[(Pha > 0.3) & (Pha < 2.2)]
    sbear = np.degrees(np.arctan2(slab[:, 1], slab[:, 0]))
    print(f"pair {PAIR}: {len(blobs)} glare source(s), {mover.sum()}/{len(d)} moving radar", flush=True)

    cars = []

    def emit_cluster(cl):
        """Validate a cluster against the physical car model + ground plane + isolation. Return 1 if a car."""
        if len(cl) < CAR_MIN_PTS: return 0
        cen = cl[:, :2].mean(0); X = cl[:, :2]-cen
        _, _, vt = np.linalg.svd(X, full_matrices=False); proj = X @ vt.T
        mj, mnr = float(np.ptp(proj[:, 0])), float(np.ptp(proj[:, 1]))
        if not (CAR_LEN[0] <= mj <= CAR_LEN[1] and CAR_WID[0] <= mnr <= CAR_WID[1]):
            return 0                                                  # MODEL: footprint must match a car
        if float(height_above(cl, coef).min()) > GROUND_CLEAR:
            return 0                                                  # PLANE: must sit on the road (a bank is above it)
        x0, x1 = cl[:, 0].min()-0.4, cl[:, 0].max()+0.4; y0, y1 = cl[:, 1].min()-0.4, cl[:, 1].max()+0.4
        box = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
        if not LOW_SLICE:
            if int((box & (Pha > TALL_H)).sum()) >= 8:
                return 0                                              # TALL: returns overhead -> tree/pole/light
            m = 1.2; big = (P[:, 0] >= x0-m) & (P[:, 0] <= x1+m) & (P[:, 1] >= y0-m) & (P[:, 1] <= y1+m)
            ring = big & ~box & (Pha > 0.3) & (Pha < 2.2)            # ISOLATION: a car has empty road around it
            a_cl = max((x1-x0)*(y1-y0), 1.0); a_ring = max((x1-x0+2*m)*(y1-y0+2*m) - a_cl, 1.0)
            if ring.sum()/a_ring > ISO_RATIO * (len(cl)/a_cl):
                return 0                                              # continuous surface (embankment), not a car
        r = float(np.hypot(*cen)); cb = np.degrees(np.arctan2(cen[1], cen[0]))
        nr = (np.abs(rrng - r) < 2.5) & (np.abs(((rbear - cb + 180) % 360) - 180) < 10)
        moving = bool((nr & mover).any()); strong = bool(nr.any() and rpow[nr].max() >= powhi)
        if LOW_SLICE and not (moving or strong):
            return 0                                                  # dense foliage: RADAR (metal car) picks the car
        tier = "moving" if moving else ("radar" if strong else ("lidar_dense" if len(cl) >= 40 else "lidar"))
        print(f"    -> CAR {r:.1f}m ({cen[0]:.1f},{cen[1]:.1f}) {mj:.1f}x{mnr:.1f} n{len(cl)} [{tier}]", flush=True)
        cars.append(dict(cxy=cen, pts=cl, tier=tier)); return 1

    for b in blobs:
        # wide cone over the glare blob's column span (bloom centroid is offset from the car); the car
        # MODEL + ground plane + isolation reject clutter, not the bearing.
        b_lo, b_hi = sorted((ray(b["u0"], b["vb"])[0], ray(b["u1"], b["vb"])[0]))
        onb = (sbear >= b_lo - BEARING_TOL) & (sbear <= b_hi + BEARING_TOL)
        pts = slab[onb]; found = 0
        if len(pts) >= CAR_MIN_PTS:
            lb = DBSCAN(eps=1.0, min_samples=4).fit_predict(pts[:, :2])
            for c in set(lb.tolist()):
                if c == -1: continue
                cl = pts[lb == c]
                if len(cl) < CAR_MIN_PTS: continue
                X = cl[:, :2] - cl[:, :2].mean(0)
                _, _, vt = np.linalg.svd(X, full_matrices=False); proj = X @ vt.T
                mj = float(np.ptp(proj[:, 0]))
                if mj <= SPLIT_LEN:
                    found += emit_cluster(cl)                         # one car
                elif mj <= MAX_LEN:                                   # longer than a car -> adjacent cars fused: SPLIT
                    sub = DBSCAN(eps=0.6, min_samples=3).fit_predict(cl[:, :2])
                    labs = [x for x in set(sub.tolist()) if x != -1]
                    if len(labs) >= 2:
                        for x in labs: found += emit_cluster(cl[sub == x])
                    else:
                        ns = max(2, int(round(mj/4.5))); edg = np.linspace(proj[:, 0].min(), proj[:, 0].max(), ns+1)
                        for i in range(ns): found += emit_cluster(cl[(proj[:, 0] >= edg[i]) & (proj[:, 0] <= edg[i+1])])
                # mj > MAX_LEN -> wall/guardrail, drop
        if not found and len(pts):
            # glare says a car is here but the gated path found none -> it BLENDS with structure (close car
            # against a wall/under canopy, fails TALL/ISO). Localize by the NEAREST dense car-height window;
            # density (not isolation) is the discriminator -- background on a bearing is too sparse to trigger.
            rr = np.hypot(pts[:, 0], pts[:, 1])
            for idx in np.argsort(rr):
                r0 = rr[idx]
                if r0 < DENSE_RMIN: continue                             # skip ego-adjacent curb/wall
                if r0 >= DENSE_RMAX: break
                win = pts[(rr >= r0) & (rr < r0 + 2.5)]
                if len(win) < DENSE_MIN or float(height_above(win, coef).min()) > GROUND_CLEAR:
                    continue
                X = win[:, :2] - win[:, :2].mean(0)                      # CAR-footprint check: a wall swath sprawls
                _, _, vt = np.linalg.svd(X, full_matrices=False); proj = X @ vt.T
                if float(np.ptp(proj[:, 1])) > DENSE_WID: continue       # too wide -> continuous structure, not a car
                cen = win[:, :2].mean(0)
                rc = float(np.hypot(*cen)); cbb = np.degrees(np.arctan2(cen[1], cen[0]))
                nrd = (np.abs(rrng - rc) < 2.5) & (np.abs(((rbear - cbb + 180) % 360) - 180) < 10)
                if not bool((nrd & mover).any()):
                    continue                                             # require MOVING radar: a static embankment /
                #                                                          guardrail is dense + radar-bright too, but a
                #                                                          MOVING return at this spot can only be a car
                print(f"    -> CAR(dense) {rc:.1f}m ({cen[0]:.1f},{cen[1]:.1f}) n{len(win)} [glare_dense]", flush=True)
                cars.append(dict(cxy=cen, pts=win, tier="glare_dense")); found += 1; break
        if not found:
            print(f"  blob u[{b['u0']:.0f},{b['u1']:.0f}] bearing[{b_lo:+.1f},{b_hi:+.1f}] -> no car", flush=True)
    # dedup: several glare blobs of ONE car land on the SAME lidar cluster -> keep the densest per location
    order = {"moving": 4, "radar": 3, "glare_dense": 2, "lidar_dense": 1, "lidar": 0}
    cars.sort(key=lambda c: (order[c["tier"]], len(c["pts"])), reverse=True)
    kept = []
    for c in cars:
        if any(np.hypot(c["cxy"][0]-k["cxy"][0], c["cxy"][1]-k["cxy"][1]) < DEDUP_M for k in kept):
            continue
        kept.append(c)
    return kept


def to_boxes(cars):
    out = []
    TIER = {"moving": (0.75, "lime", True), "radar": (0.6, "lime", True),
            "glare_dense": (0.55, "lime", True),
            "lidar_dense": (0.5, "lime", True), "lidar": (0.35, "orange", False)}
    for c in cars:
        cl = c["pts"]; fwd, lat = float(c["cxy"][0]), float(c["cxy"][1])
        X = cl[:, :2] - c["cxy"]; _, _, vt = np.linalg.svd(X, full_matrices=False)
        yaw = float(np.arctan2(vt[0, 1], vt[0, 0])); proj = X @ vt.T
        L = float(np.clip(np.ptp(proj[:, 0]), 1.5, 6.0)); Wd = float(np.clip(np.ptp(proj[:, 1]), 1.6, 2.6))
        if L < 3.0: L = 4.3; yaw = float(np.arctan2(lat, fwd))
        z0 = float(cl[:, 2].min()); Hh = float(np.clip(np.ptp(cl[:, 2]), 1.3, 2.0)); cz = z0 + Hh/2
        dx = np.array([L/2]*2 + [-L/2]*2); dy = np.array([Wd/2, -Wd/2, -Wd/2, Wd/2])
        cs, sn = np.cos(yaw), np.sin(yaw); fxx = fwd+dx*cs-dy*sn; fyy = lat+dx*sn+dy*cs
        pxs = []
        for z in (z0, z0+Hh):
            for k in range(4):
                opt = (np.linalg.inv(C2L) @ np.r_[fxx[k], fyy[k], z, 1])[:3]
                if opt[2] > 0.2: pxs.append([fx*opt[0]/opt[2]+cx, fy*opt[1]/opt[2]+cy])
        bbox = None
        if len(pxs) >= 4:
            a = np.array(pxs); bbox = [float(a[:,0].min()), float(a[:,1].min()), float(a[:,0].max()), float(a[:,1].max())]
        score, col, rel = TIER[c["tier"]]
        box3d = dict(cx=fwd, cy=lat, cz=cz, yaw=yaw, length=L, width=Wd, height=Hh,
                     reliable=rel, size_plausible=True, length_fit=0.0, n_points=int(len(cl)))
        out.append(dict(label="car", source=f"glare_{c['tier']}", score=score, color=col,
            bbox_xyxy=bbox, box3d=box3d, radar_confirmed=(c["tier"] in ("moving", "radar")),
            forward=[fwd-L/2, fwd+L/2], lateral=[lat-Wd/2, lat+Wd/2]))
    return out


if __name__ == "__main__":
    objects = {str(p): to_boxes(detect(p)) for p in PAIRS}
    os.makedirs(f"{RES}/annotations", exist_ok=True)
    ANN = f"{RES}/annotations/glare_{OUT_TAG}.json"
    json.dump(dict(objects=objects), open(ANN, "w"), indent=1)
    print(f"wrote {ANN} ({sum(len(v) for v in objects.values())} boxes / {len(objects)} pairs)", flush=True)
