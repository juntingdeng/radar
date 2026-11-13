"""Compare gate variants (threshold baseline vs learned feature gate) on the REAL benchmark_v1 BEV cache.
Early-stopping (best val recall) + weight decay + multi-seed, because 51 train frames overfit fast.
NOTE: these are CLEAR frames (lidar dense everywhere, radar sparse) -> expect the gate to matter LITTLE here;
the gate's payoff is the GLARE/lidar-blind regime (needs verified GT). This run is the clear-condition control.
  python3 sync/src/fusion/attn_real.py
"""
import os, sys
import numpy as np, torch, torch.nn.functional as F
from scipy import ndimage

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "sync/src/fusion"))
from attn_fusion_proto import BEVAttnFusionNet, DEV


def obj_recall_prec(pred, tgt):
    tp_r = n_gt = tp_p = n_pr = 0
    for b in range(pred.shape[0]):
        gl, ng = ndimage.label(tgt[b, 0] > 0.5); pl, npr = ndimage.label(pred[b, 0] > 0.5)
        n_gt += ng; n_pr += npr
        for g in range(1, ng + 1):
            if (pred[b, 0][gl == g] > 0.5).any(): tp_r += 1
        for p in range(1, npr + 1):
            if (tgt[b, 0][pl == p] > 0.5).any(): tp_p += 1
    return tp_r / max(n_gt, 1), tp_p / max(n_pr, 1)


def train_eval(data, gate_kind, seed, steps=400):
    lidar, radar, relf, tgt = data
    n = len(lidar)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, va = perm[:int(n * 0.8)], perm[int(n * 0.8):]
    to = lambda t, idx: t[idx].to(DEV)
    Ltr, Rtr, Ftr, Ttr = to(lidar, tr), to(radar, tr), to(relf, tr), to(tgt, tr)
    Lva, Rva, Fva, Tva = to(lidar, va), to(radar, va), to(relf, va), to(tgt, va)
    torch.manual_seed(seed)
    net = BEVAttnFusionNet(c_l=3, c_r=3, relf=3, gate_kind=gate_kind).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
    best = (-1, 0, 0)  # recall, prec, mean_gate  (early stopping on val recall)
    for step in range(steps):
        net.train(); hm, g = net(Ltr, Rtr, Ftr)
        loss = F.binary_cross_entropy_with_logits(hm, Ttr, pos_weight=torch.tensor(40.0, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            net.eval()
            with torch.no_grad():
                hv, gv = net(Lva, Rva, Fva)
                rec, prec = obj_recall_prec((hv.sigmoid() > 0.5).float().cpu().numpy(), Tva.cpu().numpy())
            if rec > best[0]:
                best = (rec, prec, gv.mean().item())
    return best


def main():
    d = torch.load(os.path.join(REPO, "sync/res/benchmark_v1/bev_cache.pt"))
    data = (d["lidar"].float(), d["radar"].float(), d["relf"].float(), d["target"].float())
    print(f"device={DEV}  {len(data[0])} real clear frames  grid {tuple(data[0].shape[2:])}  "
          f"(early-stop best val recall, 3 seeds)\n")
    print(f"{'gate':10s} | {'best val recall':>16s} | {'prec@best':>10s} | {'mean gate':>9s}")
    print("-" * 56)
    for kind in ["threshold", "shallow", "feature"]:
        res = [train_eval(data, kind, s) for s in (0, 1, 2)]
        rec = np.array([r[0] for r in res]); prec = np.array([r[1] for r in res]); mg = np.array([r[2] for r in res])
        print(f"{kind:10s} | {rec.mean():6.2f} +/- {rec.std():4.2f}   | {prec.mean():10.2f} | {mg.mean():9.3f}")
    print("\nbest val recall = object recall at the early-stopping peak (avg of 3 seeds).")
    print("Clear-condition control: if gate variants tie here, that's EXPECTED (no lidar-blind cells to exploit);")
    print("the discriminating test is test_glare (lidar-blind) with verified GT.")


if __name__ == "__main__":
    main()
