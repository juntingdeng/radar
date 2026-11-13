"""Label the 10.24 green-pants SMOKE window: DETR (person/car) + lidar FRUSTUM -> 3D boxes (forward/lateral).

For each window frame: DETR 2D boxes -> project the frame's lidar into the camera (inv camera_to_lidar + colour
intrinsics) -> the box's frustum lidar points -> median (fwd,lat) = object position + a size prior (person/car).
Outputs benchmark-format label entries (forward, lateral, bbox_xyxy, label, source, score, median_depth, n_lidar)
and a verification preview montage. Smoke makes this unreliable -> the output is AUTO labels to be hand-verified."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(CODE, "sync", "src")); sys.path.insert(0, os.path.join(CODE, "sync", "src", "fusion", "late_fusion"))
import camera_proposals as CP
RES = os.path.join(CODE, "sync", "res", "2026.10.24", "15.02.30")
CALIB = json.load(open(os.path.join(RES, "calibration.json")))
NPZ = os.path.expanduser("~/.cache/ti_setup_frames/smoke_window.npz")
LIDC = os.path.expanduser("~/.cache/ti_setup_frames/smoke1024_lidar")
OUT = os.path.join(RES, "smoke_labels_auto.json")
T = np.array(CALIB["camera_to_lidar"], float); Tinv = np.linalg.inv(T)
ci = CALIB["color_intrinsics"]; FX, FY, CX, CY = ci["fx"], ci["fy"], ci["cx"], ci["cy"]
SIZE = {1: (0.5, 0.5), 2: (1.0, 0.6), 3: (2.1, 0.9), 4: (1.0, 0.6), 6: (6.0, 1.4), 8: (6.0, 1.4)}  # half-extent fwd,lat by COCO class

def frustum(bbox2d, lid):
    """DETR 2D box + lidar (N,>=3 fwd,lat,z) -> (fwd,lat, median_depth, n_pts) or None."""
    P = lid[:, :3].astype(float); Pc = (Tinv @ np.c_[P, np.ones(len(P))].T).T[:, :3]   # lidar -> camera frame
    Z = Pc[:, 2]; front = Z > 0.5
    u = FX * Pc[:, 0] / np.where(front, Z, 1) + CX; v = FY * Pc[:, 1] / np.where(front, Z, 1) + CY
    x0, y0, x1, y1 = bbox2d
    inb = front & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
    if inb.sum() < 3: return None
    sel = P[inb]; rng = np.linalg.norm(sel[:, :2], axis=1)
    rmin = np.percentile(rng, 10)                                                       # robust nearest range (ignore a stray near point)
    near = sel[rng <= rmin + 2.5]                                                        # FOREGROUND cluster only (person, not the building behind)
    if len(near) < 3: near = sel[rng <= np.percentile(rng, 30)]
    if len(near) < 3: near = sel
    return float(np.median(near[:, 0])), float(np.median(near[:, 1])), float(np.median(np.linalg.norm(near[:, :2], axis=1))), int(len(near))

def main():
    S = np.load(NPZ, allow_pickle=True); cams = S["cams"]; meta = json.loads(str(S["meta"]))
    detr = CP._detector("detr"); labels = {}; nlab = 0
    for k, mt in enumerate(meta):
        cam = cams[k]
        if cam.max() == 0: continue
        lid = np.load(f"{LIDC}/l_{mt['lidar_idx']}.npy").astype(float)
        sid = f"1024_{mt['pair']:06d}"; boxes = []
        for bbox2d, score, label in detr(cam):
            fr = frustum(bbox2d, lid)
            if fr is None: continue
            fwd, lat, md, npts = fr; hf, hl = SIZE.get(label, (0.6, 0.6))
            boxes.append(dict(forward=[fwd - hf, fwd + hf], lateral=[lat - hl, lat + hl], bbox_xyxy=[round(x, 1) for x in bbox2d],
                              label="person" if label == 1 else "car", source="detr_lidar_frustum", score=round(score, 3),
                              median_depth_m=round(md, 2), n_depth_points=npts, reliable=False))
        if boxes: labels[sid] = boxes; nlab += len(boxes)
        if k % 40 == 0: print(f"  frame {k+1}/{len(meta)}: {len(boxes)} boxes", flush=True)
    json.dump(labels, open(OUT, "w"), indent=1)
    import collections; lc = collections.Counter(b["label"] for v in labels.values() for b in v)
    print(f"\nsaved {OUT}: {len(labels)} frames labeled, {nlab} boxes {dict(lc)}")

if __name__ == "__main__":
    main()
