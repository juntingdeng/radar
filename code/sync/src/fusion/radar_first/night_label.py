"""Night car-labeling ROUTER: per frame, detect the regime and apply the strategy that fits it.

No single cue labels a glare-blind car, and the two working strategies are complementary:
  - MOVING cars  -> rescue_v3  (ego-Doppler gate -> moving-car rescue)      [works on W1 14260]
  - STOPPED cars -> glare_bearing (glare bearing + ground-level lidar cluster) [works on W2 14500]
Regime signal = ego-Doppler MOVER COUNT, which cleanly separates the two (W1: 12-20 movers/frame;
W2: 0-9). Per frame:  movers > MOVER_TH -> MOVING -> rescue_v3 boxes;  else -> STOPPED -> glare boxes.
One entry point that reproduces both windows.
    python3 night_label.py 14260 14270 14280 14290 14300
"""
import os, sys, json, subprocess
from common import RES

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
PAIRS = sys.argv[1:] or ["14280"]
MOVER_TH = float(os.environ.get("NIGHT_MOVER_TH", 10))   # >TH movers = busy moving scene -> rescue; else glare


def run(script):
    print(f"--- running {script} on {len(PAIRS)} pair(s) ---", flush=True)
    subprocess.run([PY, "-u", os.path.join(HERE, script), *PAIRS], check=True)


run("rescue_v3.py")
run("glare_bearing.py")

rescue = json.load(open(f"{RES}/annotations/rescue_v3.json"))
movers = rescue.get("movers", {}); rboxes = rescue["objects"]
gtag = PAIRS[0] if len(PAIRS) == 1 else f"win{PAIRS[0]}_{PAIRS[-1]}"
gboxes = json.load(open(f"{RES}/annotations/glare_{gtag}.json"))["objects"]

def bev(b):
    b3 = b.get("box3d") or {}
    return float(b3.get("cx", 0.0)), float(b3.get("cy", 0.0))


def merge(a, b, tol=3.0):
    """Union two box lists; drop a later box whose BEV center is within tol of one already kept."""
    out = list(a)
    for box in b:
        bx, by = bev(box)
        if any(abs(bx-ox) < tol and abs(by-oy) < tol for ox, oy in (bev(o) for o in out)):
            continue
        out.append(box)
    return out


DEDUP_M = float(os.environ.get("NIGHT_DEDUP_M", 3.0))
objects = {}
print("\n=== MERGE (rescue + glare) ===", flush=True)
for p in PAIRS:
    mv = int(movers.get(p, 0))
    rb, gb = rboxes.get(p, []), gboxes.get(p, [])
    # a mover-heavy frame is dominated by moving cars -> rescue leads and glare fills in the stopped/oncoming
    # cars it misses; a quiet frame is the reverse. Either way BOTH run and the union is deduped by BEV.
    objects[p] = merge(rb, gb, DEDUP_M) if mv > MOVER_TH else merge(gb, rb, DEDUP_M)
    print(f"pair {p}: {mv:3d} movers | rescue {len(rb)} + glare {len(gb)} -> {len(objects[p])} box(es)", flush=True)
ANN = f"{RES}/annotations/night_label.json"
json.dump(dict(objects=objects), open(ANN, "w"), indent=1)
print(f"\nwrote {ANN} ({sum(len(v) for v in objects.values())} boxes / {len(objects)} pairs)", flush=True)
