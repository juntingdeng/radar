"""Reliability-gated cross-attention BEV fusion — prototype.

Idea (ties to the condition-aware story): fuse lidar + radar BEV features with cross-attention whose
contribution is modulated by a per-CELL reliability gate derived from the data (lidar density, radar
power, range). Where lidar is reliable the gate suppresses radar; where lidar drops out (far range /
glare-induced sparsity) the gate opens radar. The gate map IS a spatial 'radar-reliance' visualization.

This file is self-contained: a toy experiment synthesizes scenes where lidar is zeroed inside a random
'dropout' box (the glare/range analog) while radar still sees the objects, and shows the learned gate
opens exactly in the dropout regions. Run:  python3 sync/src/fusion/attn_fusion_proto.py
"""
import torch, torch.nn as nn, torch.nn.functional as F

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- model ----------------
class GateThreshold(nn.Module):
    """BASELINE (no learning): a fixed hand rule on reliability feats [lidar_density, radar_power, range].
    Open radar where lidar is sparse or far. No parameters -> isolates whether LEARNING the gate helps."""
    def forward(self, relfeat):
        ld, rng = relfeat[:, 0:1], relfeat[:, 2:3]
        return torch.sigmoid(6.0 * (0.10 - ld) + 2.0 * (rng - 0.6))


class GateShallow(nn.Module):
    """learned but SHALLOW: 1x1 conv (per-cell, no context) over hand-crafted relfeat = learned soft threshold."""
    def __init__(self, fin, hid=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(fin, hid, 1), nn.ReLU(), nn.Conv2d(hid, 1, 1))

    def forward(self, relfeat):
        return torch.sigmoid(self.net(relfeat))


class GateFeature(nn.Module):
    """LEARNED POLICY: gate from the ENCODER FEATURES (lidar+radar) with 3x3 spatial context.
    Learns what 'unreliable' looks like from data + neighborhood, not from 3 hand-picked scalars."""
    def __init__(self, c, hid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * c, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, 1, 1))

    def forward(self, l_feat, r_feat):
        return torch.sigmoid(self.net(torch.cat([l_feat, r_feat], 1)))


class GatedCrossAttn(nn.Module):
    """Soft per-cell modality SELECTION + attention refinement.
      base = (1-g)*lidar + g*radar         # direct gated path: radar reaches the head where lidar is blind
      query = lidar+radar (so blind cells still query via radar); keys/values = radar
      fused = base + g * crossattn(query, radar)
    """
    def __init__(self, c, heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm = nn.LayerNorm(c)

    def forward(self, l, r, gate):
        B, C, H, W = l.shape
        q = (l + r).flatten(2).transpose(1, 2)         # B, HW, C  (radar-aware query)
        rt = r.flatten(2).transpose(1, 2)
        att, _ = self.mha(self.norm(q), self.norm(rt), rt)
        att = att.transpose(1, 2).reshape(B, C, H, W)
        base = (1 - gate) * l + gate * r               # convex gate = soft modality selection
        return base + gate * att                        # attention refines, opened by the gate


class BEVAttnFusionNet(nn.Module):
    def __init__(self, c_l=2, c_r=3, c=32, relf=3, gate_kind="feature"):
        super().__init__()
        self.enc_l = nn.Sequential(nn.Conv2d(c_l, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU())
        self.enc_r = nn.Sequential(nn.Conv2d(c_r, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU())
        self.gate_kind = gate_kind
        self.gate = {"threshold": GateThreshold, "shallow": lambda: GateShallow(relf),
                     "feature": lambda: GateFeature(c)}[gate_kind]() if gate_kind != "threshold" else GateThreshold()
        self.fuse = GatedCrossAttn(c)
        self.head = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, 1, 1))

    def forward(self, lidar_bev, radar_bev, relfeat):
        l = self.enc_l(lidar_bev); r = self.enc_r(radar_bev)
        g = self.gate(l, r) if self.gate_kind == "feature" else self.gate(relfeat)
        f = self.fuse(l, r, g)
        return self.head(f), g                         # heatmap logits, gate map


# ---------------- toy data (glare/range-dropout analog) ----------------
def toy_batch(B, H, W, n_obj=5, device=DEV, gen=None):
    lidar = torch.zeros(B, 2, H, W); radar = torch.zeros(B, 3, H, W); tgt = torch.zeros(B, 1, H, W)
    dropout = torch.zeros(B, 1, H, W)                  # where lidar is blind (bookkeeping only)
    ri = torch.randint
    for b in range(B):
        # lidar-dropout box (the glare/far-range analog): lidar sees nothing inside it
        y0, x0 = int(ri(0, H // 2, (1,), generator=gen)), int(ri(0, W // 2, (1,), generator=gen))
        y1, x1 = y0 + H // 3, x0 + W // 3
        dropout[b, 0, y0:y1, x0:x1] = 1
        for _ in range(n_obj):
            oy, ox = int(ri(2, H - 2, (1,), generator=gen)), int(ri(2, W - 2, (1,), generator=gen))
            tgt[b, 0, oy, ox] = 1
            # radar ALWAYS sees the object but with COARSE localization (jitter +-2 cells = angular res)
            jy, jx = int(ri(-2, 3, (1,), generator=gen)), int(ri(-2, 3, (1,), generator=gen))
            ry, rx = min(max(oy + jy, 0), H - 1), min(max(ox + jx, 0), W - 1)
            radar[b, 0, ry, rx] = 1; radar[b, 2, ry, rx] = 1 + torch.rand(1, generator=gen).item()
            if not dropout[b, 0, oy, ox]:                # lidar sees it (EXACT) only outside dropout
                lidar[b, 0, oy, ox] = 1; lidar[b, 1, oy, ox] = 1
    tgt = F.max_pool2d(tgt, 3, 1, 1)                     # fatten targets to 3x3 blobs
    lidar += 0.02 * torch.randn_like(lidar); radar += 0.02 * torch.randn_like(radar)
    # reliability feats: lidar density, radar power, normalized range (dist from bottom-center ego)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    rng = torch.hypot(yy, xx)[None, None].expand(B, 1, H, W)
    ldens = F.avg_pool2d(lidar[:, :1], 5, 1, 2)
    relf = torch.cat([ldens, radar[:, 2:3], rng], 1)
    return (lidar.to(device), radar.to(device), relf.to(device), tgt.to(device), dropout.to(device))


# ---------------- demo ----------------
def run_gate(gate_kind, H=48, W=48, steps=400):
    torch.manual_seed(0)                                   # same init-seed & data stream per variant
    net = BEVAttnFusionNet(gate_kind=gate_kind).to(DEV)
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if any(p.requires_grad for p in net.parameters()):
        opt = torch.optim.Adam(net.parameters(), 2e-3)
    for step in range(steps):
        lidar, radar, relf, tgt, drop = toy_batch(16, H, W)
        hm, g = net(lidar, radar, relf)
        loss = F.binary_cross_entropy_with_logits(hm, tgt, pos_weight=torch.tensor(20.0, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():                                  # fixed eval set
        g_ = torch.Generator().manual_seed(123)
        lidar, radar, relf, tgt, drop = toy_batch(64, H, W, gen=g_)
        hm, g = net(lidar, radar, relf)
        pred = (hm.sigmoid() > 0.5).float(); obj = (tgt > 0.5)
        rec_in = ((pred.bool() & obj & drop.bool()).sum() / (obj & drop.bool()).sum().clamp(min=1)).item()
        rec_out = ((pred.bool() & obj & ~drop.bool()).sum() / (obj & ~drop.bool()).sum().clamp(min=1)).item()
        g_in = ((g * drop).sum() / drop.sum()).item(); g_out = ((g * (1 - drop)).sum() / (1 - drop).sum()).item()
    return dict(params=params, loss=loss.item(), rec_in=rec_in, rec_out=rec_out, g_in=g_in, g_out=g_out)


def demo():
    print(f"device={DEV}   toy: lidar zeroed inside a random dropout box (glare/range analog); radar coarse\n")
    print(f"{'gate':10s} | {'params':>7s} | {'recall-in':>9s} | {'recall-out':>10s} | {'gate sel (in-out)':>17s}")
    print("-" * 70)
    for kind in ["threshold", "shallow", "feature"]:
        m = run_gate(kind)
        sel = m["g_in"] - m["g_out"]
        print(f"{kind:10s} | {m['params']:7d} | {m['rec_in']:9.2f} | {m['rec_out']:10.2f} | "
              f"{m['g_in']:.3f}-{m['g_out']:.3f}={sel:+.3f}")
    print("\nrecall-in = objects detected in the lidar-BLIND region (radar must rescue them).")
    print("gate sel  = gate[in-dropout] - gate[outside]; >0 means it leans on radar where lidar is blind.")
    print("threshold = non-learnable baseline; feature = learned gate from encoder features + 3x3 context.")


if __name__ == "__main__":
    demo()
