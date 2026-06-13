"""A uniform but non-classifier input (MODIS NDVI, a
band-ratio MEASUREMENT) -> built-up, matched-input matrix. If source-invariant
like GHSL, "uniform" suffices; if source-dependent like raw sensors, GHSL's
invariance owes to its global model-normalisation. Matched-input matrix to
scripts.eval.landsat_temporal_matrix, but the swapped-in raw input is
Sentinel-1 SAR backscatter (VV+VH gamma-0, acquired by scripts.acquire.sentinel1)
instead of optical reflectance. Target held fixed: GHSL built-up 2015.

If SAR->built-up retention is source-DEPENDENT (~0.8, like optical imagery and
raw Landsat 0.56), then the imagery-side source-dependence is a property of
exogenous raw-sensor inputs in general -- not an optical quirk -- and the
uniform-product source-invariance is the genuinely special case. That is the
breadth result: the divide is provenance (uniform product vs raw sensor),
across modalities, optical AND radar.

Usage: python -m scripts.eval.sar_matrix --seeds 20260525 1 2 3 4 --grid 64
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from scipy.stats import spearmanr

from scripts.acquire.regions import city_by_name
from scripts.common import TILE_PX, enumerate_tiles_from_grid
from scripts.eval.cross_region_eval import TARGET_EPOCH, load_city_rasters
from scripts.eval.multitask_difficulty import SegUNet

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
SARDIR = REPO / "data/raw/ndvi_aligned"   # NDVI (n,3,64,64)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 20260525
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania",
           "weur", "latam", "camcar", "canada", "nordic", "central_europe",
           "china_interior", "south_asia_2", "southern_africa", "mediterranean",
           "brazil_north", "japan_korea_2"]  # 8 original + 12 expansion (n=20)
GRID = 64
CENTER_PX = 27                  # 6.72 km / 250 m, matches the SAR patch extent + the L8 control
BU_THRESH = 0.1


def load_region(region):
    X, Y = [], []
    for npz_path in glob.glob(str(SARDIR / "*.npz")):
        city = city_by_name(Path(npz_path).stem)
        if city is None or city.region != region:
            continue
        d = np.load(npz_path, allow_pickle=True)
        patches = d["patches"]                         # (n,3,64,64) float32 NDVI x3yr
        tile_ids = list(d["tile_ids"])
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
            box = gt[i + c0:i + c0 + CENTER_PX, j + c0:j + c0 + CENTER_PX]
            if box.shape != (CENTER_PX, CENTER_PX):
                continue
            tgt = (zoom(box, GRID / CENTER_PX, order=1) > BU_THRESH).astype(np.int64)[:GRID, :GRID]
            if tgt.shape != (GRID, GRID):
                continue
            sar = patches[k].astype(np.float32)        # (3,64,64)
            if sar.shape[-1] != GRID:
                sar = zoom(sar, (1, GRID / sar.shape[-2], GRID / sar.shape[-1]), order=1)
            sar = sar[:, :GRID, :GRID]
            if sar.shape != (sar.shape[0], GRID, GRID):
                continue
            X.append(sar); Y.append(tgt)
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
    tr = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))[:int(0.85 * len(X))].to(DEV)
    X, Y = X.to(DEV), Y.to(DEV)
    m = SegUNet(in_ch=X.shape[1], n=2).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    ce = nn.CrossEntropyLoss()
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


def stats_for_seed(seed, ready):
    global SEED
    SEED = seed
    models = {r: train(r) for r in ready}; models = {r: m for r, m in models.items() if m is not None}
    rr = list(models)
    mat = {s: {t: iou(models[s], t) for t in rr} for s in models}
    diag = np.mean([mat[r][r] for r in rr if mat[r].get(r) is not None])
    off = np.mean([mat[s][t] for s in rr for t in rr if s != t and mat[s].get(t) is not None])
    srcs = [s for s in rr if all(mat[s].get(t) is not None for t in rr)]
    M = np.array([[mat[s][t] for t in rr] for s in srcs])
    inv = np.mean([spearmanr(M[a], M[b]).correlation
                   for a in range(len(srcs)) for b in range(a+1, len(srcs))]) if len(srcs) > 1 else float("nan")
    return {"in_region": float(diag), "out_region": float(off), "home_field_gap": float(diag - off),
            "retention": float(off / diag) if diag else None, "source_inv": float(inv),
            "diag": {r: round(float(mat[r][r]), 3) for r in rr if mat[r].get(r) is not None}}


def main(seeds, grid):
    global GRID
    GRID = grid; _CACHE.clear()
    ready = [r for r in REGIONS if region_xy(r)[0] is not None and len(region_xy(r)[0]) >= 20]
    print(f"SAR grid={grid} regions={ready} tiles={[len(region_xy(r)[0]) for r in ready]}", flush=True)
    per = {}
    for sd in seeds:
        s = stats_for_seed(sd, ready); per[sd] = s
        print(f"[seed {sd}] in={s['in_region']:.3f} out={s['out_region']:.3f} "
              f"retention={s['retention']:.3f} gap={s['home_field_gap']:+.3f} "
              f"src_inv={s['source_inv']:.3f}", flush=True)
    agg = {}
    for k in ["in_region", "out_region", "home_field_gap", "retention", "source_inv"]:
        v = np.array([per[sd][k] for sd in seeds], dtype=float)
        agg[k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0}
    out = {"input": "MODIS NDVI 3-year (uniform measurement)", "target": "GHSL built-up 2015",
           "grid": grid, "seeds": list(seeds), "per_seed": {str(sd): per[sd] for sd in seeds},
           "aggregate": agg}
    fn = REPO / f"results/metrics/ndvi_builtup_g{grid}_multiseed.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"\n=== MODIS NDVI->built-up grid={grid}, {len(seeds)} seeds ===")
    print(f"  retention {agg['retention']['mean']:.3f} +/- {agg['retention']['sd']:.3f}  |  "
          f"gap {agg['home_field_gap']['mean']:+.3f} +/- {agg['home_field_gap']['sd']:.3f}  |  "
          f"src_inv {agg['source_inv']['mean']:.2f} +/- {agg['source_inv']['sd']:.2f}", flush=True)
    print("  compare: GHSL-history ~1.0 (uniform PRODUCT) ; raw Landsat 0.56 ; SAR 0.89", flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[20260525, 1, 2, 3, 4])
    p.add_argument("--grid", type=int, default=64)
    a = p.parse_args()
    GRID = a.grid
    main(a.seeds, a.grid)
