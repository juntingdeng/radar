"""Cross-modal association: per-modality BEV proposals -> detection GROUPS by BEV IoU, with cross-modal features.

Each group is a candidate detection that aggregates whichever modalities localise the same place. A single-modality
group (e.g. camera only) is an uncorroborated detection = likely false positive; a multi-modality group is corroborated.
Greedy: proposals processed by descending score; each joins an existing group it overlaps (IoU>=thr), else starts one;
best score per modality is kept.

group_features(g) -> the per-detection input for the fusion head / eval:
  scores   [s_cam, s_lid, s_rad]   (0 if that modality is absent)
  present  [0/1, 0/1, 0/1]
  pair_iou [IoU_cam-lid, IoU_cam-rad, IoU_lid-rad]   (0 if a modality absent)  <- SPATIAL agreement
  box      the group's BEV box (highest-score member)
"""
import numpy as np
import boxes as BX
MODS = ["cam", "lidar", "radar"]

DIST_LINK = 1.5   # BEV centre distance (m) to link proposals into one group. Swept 1.5-5 m (nuScenes-style):
                  # learned-fusion F1 0.49/0.47/0.43/0.42/0.43/0.41 at 1.5/2/2.5/3/4/5 -- tighter wins because a
                  # wide radius swallows distinct neighbours (plaza car rows) into one group, and a group can only
                  # match one GT. 1.5 m still links cross-modal pairs (detector ATEs are 0.6-1.1 m).

def _cdist(a, b):
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2); cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

def associate(props_by_mod, dist_thr=DIST_LINK):
    """props_by_mod: {'cam':[(box,score)], 'lidar':[...], 'radar':[...]} -> list of groups (linked by centre distance)."""
    items = []
    for m in MODS:
        for box, score in props_by_mod.get(m, []):
            items.append((m, np.asarray(box, float), float(score)))
    items.sort(key=lambda x: -x[2])                      # high score anchors groups first
    groups = []
    for m, box, score in items:
        g = next((g for g in groups if any(_cdist(box, b) <= dist_thr for b in g["boxes"].values())), None)
        if g is None:
            groups.append({"boxes": {m: box}, "scores": {m: score}})
        elif m not in g["scores"] or score > g["scores"][m]:
            g["boxes"][m] = box; g["scores"][m] = score
    return groups

def group_features(g):
    """Cross-modal detection features. pair_close in [0,1] = centre agreement (1 - dist/DIST_LINK) -- the SPATIAL
    agreement signal, robust to the ~1.3 m localisation error (IoU would read ~0 for the same object).
    pair_size in (0,1] = BEV-area agreement min/max for present pairs (same object -> similar extent)."""
    scores = [g["scores"].get(m, 0.0) for m in MODS]
    present = [int(m in g["scores"]) for m in MODS]
    def close(a, b):
        if a in g["boxes"] and b in g["boxes"]:
            return max(0.0, 1.0 - _cdist(g["boxes"][a], g["boxes"][b]) / DIST_LINK)
        return 0.0
    def area(b): return max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    def size_agree(a, b):
        if a in g["boxes"] and b in g["boxes"]:
            aa, ab = area(g["boxes"][a]), area(g["boxes"][b])
            return min(aa, ab) / max(aa, ab)
        return 0.0
    pairs = [("cam", "lidar"), ("cam", "radar"), ("lidar", "radar")]
    pair_close = [close(a, b) for a, b in pairs]
    pair_size = [size_agree(a, b) for a, b in pairs]
    ref = max(g["scores"], key=g["scores"].get)
    return dict(scores=scores, present=present, pair_iou=pair_close, pair_size=pair_size,
                box=np.asarray(g["boxes"][ref], float), nmod=sum(present))

if __name__ == "__main__":
    # synthetic: a corroborated car (all 3 overlap) + a camera-only ghost + a lidar-only object
    props = {
        "cam":   [([10, -2, 14, 2], 0.9), ([25, 5, 29, 9], 0.6)],         # car@12, ghost@27
        "lidar": [([10.2, -1.8, 14.2, 2.2], 0.8), ([18, -6, 22, -2], 0.7)],  # car@12, object@20
        "radar": [([9.8, -2.1, 13.9, 1.9], 0.5)],                          # car@12
    }
    gs = associate(props)
    print(f"{len(gs)} groups (expect 3: corroborated car, camera-only ghost, lidar-only object)")
    for g in gs:
        f = group_features(g)
        print(f"  present(c,l,r)={f['present']} scores={[round(s,2) for s in f['scores']]} "
              f"pair_iou={[round(i,2) for i in f['pair_iou']]} nmod={f['nmod']}")
