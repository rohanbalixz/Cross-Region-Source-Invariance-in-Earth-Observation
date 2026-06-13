"""A geospatial foundation model (Prithvi-EO) on cross-region retention.
The frozen/fine-tuned DINOv2 control answers "does pretraining scale rescue the
imagery penalty?" but DINOv2 is natural-image pretrained -- the wrong
distribution to settle the question for satellite imagery. Prithvi-EO
(IBM/NASA, pretrained on HLS satellite imagery, the exact 6-band HLS input our
Landsat patches already use) is an EO foundation model. We freeze its encoder,
train a small decoder per source region on built-up segmentation, and read the
source-by-target retention. If Prithvi also retains ~0.8 rather than ~1.0, the
imagery penalty is a property of the input, not of using a natural-image backbone.

Usage: python -m scripts.eval.prithvi_seg_matrix --seeds 20260525 1 2
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
L8DIR = REPO / "data/raw/landsat8"; PRITHVI = REPO / "models/prithvi"
sys.path.insert(0, str(PRITHVI))
from prithvi_mae import PrithviViT

from scripts.acquire.regions import city_by_name
from scripts.common import TILE_PX, enumerate_tiles_from_grid
from scripts.eval.cross_region_eval import TARGET_EPOCH, load_city_rasters

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 20260525
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]
GRID = 64; CENTER_PX = 27; BU_THRESH = 0.1
_CFG = json.load(open(PRITHVI / "config.json"))["pretrained_cfg"]
MEAN = torch.tensor(_CFG["mean"]).view(6, 1, 1, 1); STD = torch.tensor(_CFG["std"]).view(6, 1, 1, 1)


def build_encoder():
    m = PrithviViT(img_size=224, patch_size=(1, 16, 16), num_frames=3, in_chans=6,
                   embed_dim=768, depth=12, num_heads=12, encoder_only=True)
    sd = torch.load(PRITHVI / "Prithvi_100M.pt", map_location="cpu", weights_only=False)
    sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}  # strip prefix
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert len(miss) <= 2, f"Prithvi weights did not load: {len(miss)} missing keys {miss[:5]}"
    print(f"  Prithvi loaded (missing {len(miss)}: {miss}, unexpected {len(unexp)})", flush=True)
    m.eval().to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def encode(enc, x):
    """x: (B,3,6,224,224) raw L8 DN -> spatial features (B,768,14,14)."""
    refl = x.float() * 2.75e-5 - 0.2                  # L8 C2L2 surface reflectance
    hls = (refl * 1e4).permute(0, 2, 1, 3, 4)         # (B,6,3,224,224), HLS DN scale
    xn = (hls - MEAN.to(x.device)) / STD.to(x.device)
    feat, _, _ = enc(xn, mask_ratio=0.0)              # (B, 1+588, 768)
    feat = feat[:, 1:, :].reshape(x.shape[0], 3, 14, 14, 768).mean(1)  # mean over frames
    return feat.permute(0, 3, 1, 2).contiguous()      # (B,768,14,14)


class Decoder(nn.Module):
    def __init__(self, cin=768, n=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),   # 14
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),    # 28
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),    # 56
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Upsample(size=(GRID, GRID), mode="bilinear", align_corners=False), # 64
            nn.Conv2d(32, n, 1))

    def forward(self, x):
        return self.net(x)


_CACHE = {}
def load_region(enc, region):
    if region in _CACHE:
        return _CACHE[region]
    F, Y = [], []
    for npz in glob.glob(str(L8DIR / "*.npz")):
        city = city_by_name(Path(npz).stem)
        if city is None or city.region != region:
            continue
        d = np.load(npz, allow_pickle=True)
        patches = d["patches"]; tile_ids = list(d["tile_ids"])     # (n,3,6,224,224)
        try:
            bu, _, _, transform, crs = load_city_rasters(city, PROC)
        except Exception:
            continue
        gt = bu[TARGET_EPOCH]
        refs = enumerate_tiles_from_grid(builtup_2015=gt, utm_transform=transform,
                                         city_name=city.name, region=region, utm_crs=crs)
        id2ij = {r.tile_id: (r.i, r.j) for r in refs}
        c0 = TILE_PX // 2 - CENTER_PX // 2
        xb, yb = [], []
        for k, tid in enumerate(tile_ids):
            if tid not in id2ij:
                continue
            i, j = id2ij[tid]
            box = gt[i+c0:i+c0+CENTER_PX, j+c0:j+c0+CENTER_PX]
            if box.shape != (CENTER_PX, CENTER_PX):
                continue
            tgt = (zoom(box, GRID/CENTER_PX, order=1) > BU_THRESH).astype(np.int64)[:GRID, :GRID]
            if tgt.shape != (GRID, GRID):
                continue
            xb.append(patches[k]); yb.append(tgt)
        if not xb:
            continue
        xb = torch.from_numpy(np.stack(xb))
        for s in range(0, len(xb), 16):                # encode in chunks (frozen)
            F.append(encode(enc, xb[s:s+16].to(DEV)).cpu())
        Y.extend(yb)
    if not F:
        _CACHE[region] = (None, None); return None, None
    _CACHE[region] = (torch.cat(F), torch.from_numpy(np.stack(Y)))
    return _CACHE[region]


def train(enc, region, epochs=20):
    Fe, Y = load_region(enc, region)
    if Fe is None or len(Fe) < 20:
        return None
    torch.manual_seed(SEED)
    tr = torch.randperm(len(Fe), generator=torch.Generator().manual_seed(SEED))[:int(0.85*len(Fe))].to(DEV)
    Fe, Y = Fe.to(DEV), Y.to(DEV)
    d = Decoder().to(DEV); opt = torch.optim.Adam(d.parameters(), 1e-3); ce = nn.CrossEntropyLoss()
    g = torch.Generator(device=DEV).manual_seed(SEED)
    for _ in range(epochs):
        d.train(); perm = tr[torch.randperm(len(tr), generator=g, device=DEV)]
        for k in range(0, len(perm), 16):
            b = perm[k:k+16]; opt.zero_grad(); loss = ce(d(Fe[b]), Y[b]); loss.backward(); opt.step()
    d.eval(); return d


def iou(enc, dec, region):
    Fe, Y = load_region(enc, region)
    if Fe is None:
        return None
    I = U = 0.0
    with torch.no_grad():
        for k in range(0, len(Fe), 16):
            p = dec(Fe[k:k+16].to(DEV)).argmax(1).cpu().numpy(); y = Y[k:k+16].numpy()
            I += np.logical_and(p == 1, y == 1).sum(); U += np.logical_or(p == 1, y == 1).sum()
    return float(I / U) if U > 0 else None


def main(seeds):
    global SEED
    enc = build_encoder()
    ready = [r for r in REGIONS if load_region(enc, r)[0] is not None and len(load_region(enc, r)[0]) >= 20]
    print(f"Prithvi regions={ready} tiles={[len(load_region(enc, r)[0]) for r in ready]}", flush=True)
    per = []
    for sd in seeds:
        SEED = sd
        models = {r: train(enc, r) for r in ready}; models = {r: m for r, m in models.items() if m}
        rr = list(models)
        mat = {s: {t: iou(enc, models[s], t) for t in rr} for s in models}
        diag = np.mean([mat[r][r] for r in rr if mat[r].get(r) is not None])
        off = np.mean([mat[s][t] for s in rr for t in rr if s != t and mat[s].get(t) is not None])
        srcs = [s for s in rr if all(mat[s].get(t) is not None for t in rr)]
        M = np.array([[mat[s][t] for t in rr] for s in srcs])
        inv = np.mean([spearmanr(M[a], M[b]).correlation for a in range(len(srcs))
                       for b in range(a+1, len(srcs))]) if len(srcs) > 1 else float("nan")
        per.append({"in_region": float(diag), "out_region": float(off),
                    "retention": float(off/diag) if diag else None,
                    "home_field_gap": float(diag-off), "source_inv": float(inv)})
        print(f"[seed {sd}] in={diag:.3f} out={off:.3f} retention={off/diag:.3f} "
              f"gap={diag-off:+.3f} src_inv={inv:.3f}", flush=True)
    agg = {k: {"mean": float(np.nanmean([p[k] for p in per])),
               "sd": float(np.nanstd([p[k] for p in per], ddof=1)) if len(per) > 1 else 0.0}
           for k in ("in_region", "out_region", "retention", "home_field_gap", "source_inv")}
    out = {"model": "Prithvi-EO-1.0-100M (frozen, EO foundation model)", "seeds": seeds,
           "per_seed": per, "aggregate": agg}
    fn = REPO / "results/metrics/prithvi_seg_matrix.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"\n=== Prithvi-EO frozen, {len(seeds)} seeds ===")
    print(f"  retention {agg['retention']['mean']:.3f}+/-{agg['retention']['sd']:.3f}  "
          f"gap {agg['home_field_gap']['mean']:+.3f}  src_inv {agg['source_inv']['mean']:.2f}", flush=True)
    print("  compare: from-scratch U-Net 0.82 ; frozen DINOv2 0.78 ; temporal ~1.0", flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seeds", type=int, nargs="+", default=[20260525, 1, 2])
    main(p.parse_args().seeds)
