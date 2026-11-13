"""Run all three detectors on every frame ONCE and cache per-modality BEV proposals + union GT (DETR is slow).
Also caches per-frame RAW-EVIDENCE rasters (lidar occupancy count, radar BEV power) on the detector GRID so the
fusion head can score cross-modal agreement with the raw sensors, not just with the other detectors.
Downstream (fusion_head, eval_l2) read proposals_l2.pkl so they don't re-run detectors each time."""
import os, pickle, numpy as np, torch
import data_l2, camera_proposals as CP
from voxel_detector import VoxelDetector, decode, GRID, CELL
HERE = os.path.dirname(os.path.abspath(__file__))

def evidence_grids(f):
    """-> lid (64,64) uint16 point count per cell, rad (64,64) float16 max BEV power per cell, on GRID/CELL."""
    fmin, fmax, lmin, lmax = GRID
    H = int(round((fmax - fmin) / CELL)); W = int(round((lmax - lmin) / CELL))
    lid = np.zeros((H, W), np.uint16)
    P = f["lidar_pts"]
    if len(P):
        m = (P[:, 0] >= fmin) & (P[:, 0] < fmax) & (P[:, 1] >= lmin) & (P[:, 1] < lmax)
        px = ((P[m, 0] - fmin) / CELL).astype(int); py = ((P[m, 1] - lmin) / CELL).astype(int)
        np.add.at(lid, (px, py), 1)
    rad = np.zeros((H, W), np.float32)
    bev = f["radar_bev"]; fa = np.asarray(f["fa"], float); la = np.asarray(f["la"], float)
    fi, li = np.meshgrid(np.arange(len(fa)), np.arange(len(la)), indexing="ij")
    mm = (fa[fi] >= fmin) & (fa[fi] < fmax) & (la[li] >= lmin) & (la[li] < lmax)
    gx = ((fa[fi[mm]] - fmin) / CELL).astype(int); gy = ((la[li[mm]] - lmin) / CELL).astype(int)
    np.maximum.at(rad, (gx, gy), bev[fi[mm], li[mm]])
    return lid, rad.astype(np.float16)

def load_detectors():
    ck = torch.load(f"{HERE}/detectors_l2.pt", map_location="cpu")
    lidar = VoxelDetector(in_ch=4); lidar.load_state_dict(ck["lidar"]); lidar.eval()
    radar = VoxelDetector(in_ch=3); radar.load_state_dict(ck["radar"]); radar.eval()
    return lidar, radar

def voxel_props(model, pts):
    if len(pts) == 0: return []
    b, s = decode(model, model(torch.tensor(pts).float()), score_thr=0.3, topk=20, nms_thr=0.2)
    return [(bb.tolist(), float(ss)) for bb, ss in zip(b, s)]

def main():
    # LF_CAM_BACKEND: 'detr'/'yolo' = zero-shot 2D detector + geometric D435 depth-lift (nothing trained);
    #                 'mono'        = the TRAINED camera detector (camera_detector.py: crop classifier +
    #                                 monocular range regression, no depth).
    backend = os.environ.get("LF_CAM_BACKEND", "detr")
    frames = data_l2.load_frames(); lidar, radar = load_detectors()
    train_pids = data_l2.load_split()                      # canonical split (split.json)
    camdet = None
    if backend == "mono":
        import camera_detector as CD
        camdet = CD.load_nets()
    out = []
    for i, f in enumerate(frames):
        if camdet is not None:
            import camera_detector as CD
            cam_props = [(d["bev_box"], d["score"]) for d in CD.propose(f, camdet)]
        else:
            cam_props = [(d["bev_box"], d["score"]) for d in CP.propose(f, backend)]
        props = {"cam": cam_props,
                 "lidar": voxel_props(lidar, f["lidar_pts"]),
                 "radar": voxel_props(radar, f["radar_pts"])}
        lid_ev, rad_ev = evidence_grids(f)
        out.append(dict(fid=f["fid"], cls=f["cls"], pid=f["pid"], seq=f["seq"],
                        props=props, ev=dict(lid=lid_ev, rad=rad_ev),
                        union_gt=[dict(box=u["box"].tolist(), provs=list(u["provs"]), label=u["label"])
                                  for u in f["union_gt"]]))
        if i % 20 == 0:
            print(f"{i+1}/{len(frames)} {f['fid']}: cam {len(props['cam'])} lid {len(props['lidar'])} rad {len(props['radar'])}", flush=True)
    # train_pids here is a provenance copy only -- every reader uses data_l2.load_split() (split.json)
    pickle.dump(dict(frames=out, train_pids=sorted(train_pids), backend=backend),
                open(f"{HERE}/proposals_l2.pkl", "wb"))
    tot = {m: sum(len(fr["props"][m]) for fr in out) for m in ["cam", "lidar", "radar"]}
    print(f"saved proposals_l2.pkl ({len(out)} frames). total proposals: {tot}")

if __name__ == "__main__":
    main()
