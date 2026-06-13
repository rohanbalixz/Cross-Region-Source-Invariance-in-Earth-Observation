"""The dissociation needs a non-degenerate
uniform-MEASUREMENT input that fails, scored the same way (extent IoU) as the
harmonised-classification leg (WorldCover-withheld -> built-up). NDBI, the
normalised built-up index (SWIR1-NIR)/(SWIR1+NIR) computed identically everywhere
from raw Landsat, is exactly that: a globally-uniform formula engineered for
built-up. We predict GHSL built-up 2015 from the three-frame NDBI patch and read
the source-by-target IoU retention, reporting the in-region IoU so it can be
compared with WorldCover's 0.66 and crop's excluded 0.16.

Usage: python -m scripts.eval.ndbi_builtup_matrix --seeds 20260525 1 2 3 4
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom

from scripts.acquire.regions import city_by_name
from scripts.common import TILE_PX, enumerate_tiles_from_grid
from scripts.eval.cross_region_eval import TARGET_EPOCH, load_city_rasters
from scripts.eval.multitask_difficulty import SegUNet

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
L8DIR = REPO / "data/raw/landsat8"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 20260525; GRID = 64; CENTER_PX = 27; BU_THRESH = 0.1
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania",
           "weur", "latam", "camcar", "canada", "nordic", "central_europe",
           "china_interior", "south_asia_2", "southern_africa", "mediterranean",
           "brazil_north", "japan_korea_2"]  # 8 original + 12 expansion (n=20)


def l8_refl(dn):
    return np.clip(dn.astype(np.float32) * 2.75e-5 - 0.2, 0.0, 1.0)


def load_region(region):
    X, Y = [], []
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
            refl = l8_refl(patches[k])                          # (3,6,224,224)
            nir, swir1 = refl[:, 3], refl[:, 4]                 # B05, B06 -> (3,224,224)
            ndbi = (swir1 - nir) / (swir1 + nir + 1e-6)         # NDBI per frame
            ndbi = zoom(ndbi, (1, GRID/224, GRID/224), order=1)[:, :GRID, :GRID].astype(np.float32)
            if ndbi.shape != (3, GRID, GRID):
                continue
            X.append(ndbi); Y.append(tgt)
    if not X:
        return None, None
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))


_CACHE = {}
def region_xy(region):
    if region not in _CACHE:
        _CACHE[region] = load_region(region)
    return _CACHE[region]


def train(region, epochs=15):
    X, Y = region_xy(region)
    if X is None or len(X) < 20:
        return None
    torch.manual_seed(SEED)
    tr = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))[:int(0.85*len(X))].to(DEV)
    X, Y = X.to(DEV), Y.to(DEV)
    m = SegUNet(in_ch=3, n=2).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3); ce = nn.CrossEntropyLoss()
    g = torch.Generator(device=DEV).manual_seed(SEED)
    for _ in range(epochs):
        m.train(); perm = tr[torch.randperm(len(tr), generator=g, device=DEV)]
        for k in range(0, len(perm), 8):
            b = perm[k:k+8]; opt.zero_grad(); loss = ce(m(X[b]), Y[b]); loss.backward(); opt.step()
    m.eval(); return m


def iou(model, region):
    X, Y = region_xy(region)
    if X is None:
        return None
    I = U = 0.0
    with torch.no_grad():
        for k in range(0, len(X), 16):
            p = model(X[k:k+16].to(DEV)).argmax(1).cpu().numpy(); y = Y[k:k+16].numpy()
            I += np.logical_and(p == 1, y == 1).sum(); U += np.logical_or(p == 1, y == 1).sum()
    return float(I / U) if U > 0 else None


def main(seeds):
    global SEED
    ready = [r for r in REGIONS if region_xy(r)[0] is not None and len(region_xy(r)[0]) >= 20]
    print(f"NDBI->built-up regions={ready} tiles={[len(region_xy(r)[0]) for r in ready]}", flush=True)
    per = []
    for sd in seeds:
        SEED = sd
        models = {r: train(r) for r in ready}; models = {r: m for r, m in models.items() if m}
        rr = list(models)
        mat = {s: {t: iou(models[s], t) for t in rr} for s in models}
        diag = np.mean([mat[r][r] for r in rr if mat[r].get(r) is not None])
        off = np.mean([mat[s][t] for s in rr for t in rr if s != t and mat[s].get(t) is not None])
        per.append({"in": float(diag), "out": float(off), "retention": float(off/diag) if diag else None})
        print(f"  [seed {sd}] in-IoU={diag:.3f} out-IoU={off:.3f} retention={off/diag:.3f}", flush=True)
    agg = {k: {"mean": float(np.nanmean([p[k] for p in per])),
               "sd": float(np.nanstd([p[k] for p in per], ddof=1)) if len(per) > 1 else 0.0}
           for k in ("in", "out", "retention")}
    out = {"input": "NDBI built-up index (uniform measurement)", "target": "GHSL built-up 2015",
           "seeds": seeds, "per_seed": per, "aggregate": agg}
    json.dump(out, open(REPO / "results/metrics/ndbi_builtup_matrix.json", "w"), indent=1)
    print(f"\n=== NDBI->built-up, {len(seeds)} seeds ===")
    print(f"  in-region IoU {agg['in']['mean']:.3f}  retention {agg['retention']['mean']:.3f}+/-{agg['retention']['sd']:.3f}",
          flush=True)
    print("  compare: WorldCover-withheld in-IoU 0.66 ret 0.85 ; NDVI in-IoU 0.13 (degenerate)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seeds", type=int, nargs="+", default=[20260525, 1, 2, 3, 4])
    main(p.parse_args().seeds)
