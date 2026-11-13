"""Camera proposals: a 2D detector (DETR, or local YOLO fallback) -> BEV boxes via the D435 depth-lift.

The camera subsystem is independent of the Ouster lidar: it detects in RGB and localises each box in BEV using the
D435's OWN IR depth (depth-pixel back-projection through the depth intrinsics, then camera_to_lidar). In smoke the
D435 IR depth scatters just like the camera, so a box may fail to localise (no valid depth) -> dropped. That's
realistic, and keeps "did the camera see it" separate from "did the lidar see it".

propose(frame) -> list of dict(bev_box=[fmin,lmin,fmax,lmax], score, label, bbox2d). Backend chosen by LF_CAM_BACKEND
env ("detr" default, "yolo" fallback)."""
import os, json, numpy as np
import boxes as BX
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
# REAL per-capture calibration (the benchmark_v1 calib is a generic same-rig placeholder with a bad extrinsic; the
# per-capture calibration.json carries the actual camera_to_lidar used to make the GT). 05.11 is good, 05.08 is ~45 off.
REAL_CALIB = {"2026.05.11": os.path.join(CODE, "sync", "res", "2026.05.11", "21.37.07", "calibration.json"),
              "2026.05.08": os.path.join(CODE, "sync", "res", "2026.05.08", "17-34-27", "calibration.json")}
# Which extrinsic key to use per capture, from a depth-lift A/B against union GT (hit = nearest proposal <= 2 m):
#   05.08: manual 16% (median 5.9 m, R ~45deg off) vs ICP auto 48% (2.2 m)      -> auto
#   05.11: manual 48% (2.0 m) vs PnP auto 24% (3.3 m)                           -> manual
# 05.11's hand-picked fit absorbed the D435 depth bias (it was solved THROUGH depth), so it beats the geometrically
# correct PnP auto on the depth-lift path; 05.08's hand-pick is simply wrong (4 correspondences, RMS 1.3 m).
CALIB_KEY = {"2026.05.11": "camera_to_lidar", "2026.05.08": "camera_to_lidar_auto"}
EXTRACT = {"2026.05.11": os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.11/camera_extracted"),
           "2026.05.08": os.path.expanduser("~/CMU/Research/mobile_data_local/2026.05.08/camera_extracted")}
VEHICLE_PERSON_COCO = {1, 2, 3, 4, 6, 8}   # person, bicycle, car, motorcycle, bus, truck
_calib = {}; _camidx = {}

def calib(seq):
    if seq not in _calib:
        cb = json.load(open(REAL_CALIB[seq]))
        cb["camera_to_lidar"] = cb[CALIB_KEY[seq]]
        _calib[seq] = cb
    return _calib[seq]

def _cam_index(seq):
    if seq not in _camidx:
        _camidx[seq] = np.load(os.path.join(EXTRACT[seq], "camera_index.npz"), allow_pickle=True)
    return _camidx[seq]

def aligned_depth(seq, cam_color_idx):
    """Depth frame (848,480 uint16 mm) temporally aligned to the given COLOR index, via the extracted index."""
    import cv2
    ix = _cam_index(seq); ct, dt = ix["color_ts"], ix["depth_ts"]
    if cam_color_idx < 0 or cam_color_idx >= len(ct) or len(dt) == 0: return None
    di = int(np.argmin(np.abs(dt - ct[cam_color_idx])))
    p = os.path.join(EXTRACT[seq], "depth", f"{di:06d}.png")
    if not os.path.exists(p): return None
    return cv2.imread(p, cv2.IMREAD_UNCHANGED)

def lift_box(bbox2d, depth, cb, win=9):
    """COLOR box -> BEV box using D435 depth. Returns [fmin,lmin,fmax,lmax] or None if no valid depth."""
    ci = cb["color_intrinsics"]; di = cb["intrinsics"]; T = np.array(cb["camera_to_lidar"], float)
    cx_c, cy_c, fx_c, fy_c = ci["cx"], ci["cy"], ci["fx"], ci["fy"]
    cx_d, cy_d, fx_d, fy_d = di["cx"], di["cy"], di["fx"], di["fy"]; sc = cb.get("depth_scale", 0.001)
    x0, y0, x1, y1 = bbox2d; uc = (x0 + x1) / 2; vc = (y0 + y1) / 2
    ud = int(round((uc - cx_c) / fx_c * fx_d + cx_d)); vd = int(round((vc - cy_c) / fy_c * fy_d + cy_d))
    H, W = depth.shape
    if not (0 <= vd < H and 0 <= ud < W): return None
    patch = depth[max(0, vd - win):vd + win, max(0, ud - win):ud + win].astype(float) * sc
    valid = patch[(patch > 0.3) & (patch < 40)]
    if valid.size < 5: return None
    d = float(np.median(valid))
    def to_lidar(u, v):                                   # depth pixel + d -> lidar (fwd,lat)
        X = (u - cx_d) / fx_d * d; Y = (v - cy_d) / fy_d * d; Z = d
        p = T @ np.array([X, Y, Z, 1.0]); return p[0], p[1]
    fwd, lat = to_lidar(ud, vd)
    # BEV extent from the 2D box width/height at depth d (metric), mapped through fx_d scale
    wpx = (x1 - x0) / fx_c * fx_d; wm = wpx / fx_d * d                     # lateral extent (m)
    wm = float(np.clip(wm, 1.0, 6.0)); ext = wm / 2
    return [fwd - ext, lat - ext, fwd + ext, lat + ext]

def lift_box_frustum(bbox2d, lidar_pts, cb):
    """COLOR box -> BEV box via LIDAR FRUSTUM: project the frame's lidar into the camera (inv camera_to_lidar +
    colour intrinsics), keep points inside the 2D box, take the NEAREST-cluster median (fwd,lat). Robust to the
    extrinsic's translation error (uses real lidar positions) and consistent with how the benchmark GT was made.
    Returns [fmin,lmin,fmax,lmax] or None if <3 lidar points fall in the box (e.g. smoke-scattered lidar)."""
    if lidar_pts is None or len(lidar_pts) == 0: return None
    ci = cb["color_intrinsics"]; T = np.array(cb["camera_to_lidar"], float); Tinv = np.linalg.inv(T)
    fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]
    P = np.asarray(lidar_pts, float)[:, :3]                               # lidar frame: x=fwd, y=lat, z=up
    Pc = (Tinv @ np.c_[P, np.ones(len(P))].T).T[:, :3]                    # -> camera frame (X right, Y down, Z fwd)
    Z = Pc[:, 2]; front = Z > 0.5
    u = fx * Pc[:, 0] / np.where(front, Z, 1.0) + cx
    v = fy * Pc[:, 1] / np.where(front, Z, 1.0) + cy
    x0, y0, x1, y1 = bbox2d
    inb = front & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
    if inb.sum() < 3: return None
    sel = P[inb]; rng = np.linalg.norm(sel[:, :2], axis=1)
    rmin = np.percentile(rng, 10); near = sel[rng <= rmin + 2.5]          # FOREGROUND cluster (object, not background through the box)
    if len(near) < 3: near = sel[rng <= np.percentile(rng, 30)]
    if len(near) < 3: near = sel
    fwd = float(np.median(near[:, 0])); lat = float(np.median(near[:, 1])); ext = 1.0
    return [fwd - ext, lat - ext, fwd + ext, lat + ext]

# ---------------- 2D detector backends ----------------
_det = {}
def _detector(backend):
    if backend in _det: return _det[backend]
    if backend == "detr":
        import torch
        from transformers import DetrForObjectDetection, DetrImageProcessor
        proc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()
        def run(img):
            inp = proc(images=img, return_tensors="pt")
            with torch.no_grad(): out = model(**inp)
            tgt = torch.tensor([img.shape[:2]])
            r = proc.post_process_object_detection(out, target_sizes=tgt, threshold=0.5)[0]
            return [(b.tolist(), float(s), int(l)) for b, s, l in zip(r["boxes"], r["scores"], r["labels"])
                    if int(l) in VEHICLE_PERSON_COCO]
        _det[backend] = run
    elif backend == "yolo":
        from ultralytics import YOLO
        y = YOLO(os.path.join(CODE, "yolov8n.pt")) if os.path.exists(os.path.join(CODE, "yolov8n.pt")) else YOLO("yolov8n.pt")
        def run(img):
            res = y.predict(img[:, :, ::-1], verbose=False, conf=0.3)[0]  # RGB->BGR for cv-based YOLO
            out = []
            for b in res.boxes:
                cls = int(b.cls);
                if cls in {0, 1, 2, 3, 5, 7}:  # coco person/bicycle/car/motorcycle/bus/truck
                    out.append((b.xyxy[0].tolist(), float(b.conf), cls))
            return out
        _det[backend] = run
    else:
        raise ValueError(backend)
    return _det[backend]

def propose(frame, backend=None, localize=None):
    """localize: 'frustum' (lidar frustum, default -- robust to the bad extrinsic, GT-consistent) or 'depth' (D435)."""
    backend = backend or os.environ.get("LF_CAM_BACKEND", "detr")
    # depth-lift is the default: it keeps the camera subsystem self-contained (D435's own depth), which the per-modality
    # invisibility analysis needs. With the per-capture CALIB_KEY extrinsics it hits ~48% of union GT on both day and
    # night. 'frustum' localises better (~65%, lidar has range the D435 lacks beyond ~15-20 m) but borrows lidar
    # geometry, entangling "camera saw it" with "lidar saw it" exactly in the glare/smoke conditions under study.
    localize = localize or os.environ.get("LF_CAM_LOCALIZE", "depth")
    run = _detector(backend); cb = calib(frame["seq"])
    depth = aligned_depth(frame["seq"], frame["cam_idx"]) if localize == "depth" else None
    dets = run(frame["cam"])
    out = []
    for bbox2d, score, label in dets:
        if localize == "frustum":
            bev = lift_box_frustum(bbox2d, frame.get("lidar_pts"), cb)
        else:
            bev = lift_box(bbox2d, depth, cb) if depth is not None else None
        if bev is not None:
            out.append(dict(bev_box=bev, score=score, label=label, bbox2d=bbox2d))
    return out

if __name__ == "__main__":
    import data_l2
    be = os.environ.get("LF_CAM_BACKEND", "detr")
    fr = data_l2.load_frames(("day_clean",))[:3]
    for f in fr:
        p = propose(f, be)
        G = [u["box"] for u in f["union_gt"]]
        print(f"{f['fid']}: {len(p)} cam proposals (backend={be}); union-GT {len(G)}")
        for d in p[:4]:
            fc, lc = BX.center(np.array(d["bev_box"]))
            print(f"    score {d['score']:.2f} label {d['label']} -> BEV fwd {fc:.1f} lat {lc:.1f}")
