"""BEV (top-down, forward x lateral) axis-aligned box geometry for Level-2 late fusion.

A box is [fwd_min, lat_min, fwd_max, lat_max] in metres (forward = range X, lateral = Y). The benchmark GT already
comes as forward=(f0,f1) / lateral=(l0,l1) ranges, so GT and proposals share this representation. Axis-aligned IoU
keeps the prototype simple (oriented boxes can come later). Used by camera/lidar/radar proposals, cross-modal
association, and the eval (IoU-matched TP/FP/FN, miss rate, recoverability)."""
import numpy as np

def to_bev(fwd, lat):
    """(forward=(f0,f1), lateral=(l0,l1)) -> [fmin, lmin, fmax, lmax]."""
    return np.array([min(fwd), min(lat), max(fwd), max(lat)], float)

def iou_matrix(A, B):
    """Pairwise axis-aligned BEV IoU. A:(n,4), B:(m,4) -> (n,m)."""
    A = np.asarray(A, float).reshape(-1, 4); B = np.asarray(B, float).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    f0 = np.maximum(A[:, None, 0], B[None, :, 0]); l0 = np.maximum(A[:, None, 1], B[None, :, 1])
    f1 = np.minimum(A[:, None, 2], B[None, :, 2]); l1 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(f1 - f0, 0, None) * np.clip(l1 - l0, 0, None)
    aa = ((A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]))[:, None]
    ab = ((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]))[None, :]
    u = aa + ab - inter
    return np.where(u > 0, inter / u, 0.0)

def iou(a, b):
    """Scalar IoU of two boxes."""
    return float(iou_matrix(np.asarray(a)[None], np.asarray(b)[None])[0, 0])

def nms(boxes, scores, iou_thr=0.3):
    """Greedy BEV NMS -> kept indices (descending score)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4); scores = np.asarray(scores, float)
    idx = np.argsort(scores)[::-1]; keep = []
    while len(idx):
        i = idx[0]; keep.append(int(i))
        if len(idx) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[idx[1:]])[0]
        idx = idx[1:][ious < iou_thr]
    return np.array(keep, int)

def greedy_match(A, B, iou_thr=0.1):
    """Greedy one-to-one IoU matching of rows of A to rows of B. -> list of (i, j, iou), descending IoU."""
    A = np.asarray(A, float).reshape(-1, 4); B = np.asarray(B, float).reshape(-1, 4)
    M = iou_matrix(A, B)
    if M.size == 0:
        return []
    order = np.dstack(np.unravel_index(np.argsort(M, axis=None)[::-1], M.shape))[0]
    pairs = []; usedA = set(); usedB = set()
    for i, j in order:
        if M[i, j] < iou_thr:
            break
        if i in usedA or j in usedB:
            continue
        pairs.append((int(i), int(j), float(M[i, j]))); usedA.add(i); usedB.add(j)
    return pairs

def center(box):
    """[fmin,lmin,fmax,lmax] -> (fwd_c, lat_c)."""
    b = np.asarray(box, float)
    return (b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2

def center_dist_matrix(A, B):
    """Pairwise BEV centre distance (m). A:(n,4), B:(m,4) -> (n,m). Use this (not IoU) to match detections to GT:
    BEV IoU is too sensitive to the ~1.3 m camera_to_lidar error at range; centre distance (nuScenes-style) is the
    right metric for coarse multi-sensor localisation."""
    A = np.asarray(A, float).reshape(-1, 4); B = np.asarray(B, float).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0: return np.full((len(A), len(B)), 1e9)
    ca = np.stack([(A[:, 0] + A[:, 2]) / 2, (A[:, 1] + A[:, 3]) / 2], 1)
    cb = np.stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2], 1)
    return np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
