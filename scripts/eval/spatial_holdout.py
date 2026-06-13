"""Are the in-region (diagonal) scores inflated by within-city
spatial autocorrelation? The transfer matrix splits tiles randomly within a
region, so tiles from one city can land in both train and test. If that inflates
the in-region denominator, the no-home-field result (in approx out) could be an
artefact.

Spatial block-holdout test. For each region we split its CITIES into two
disjoint halves, train on one half, and evaluate three ways:
  * in-random   : held-out TILES from the SAME cities (the standard split)
  * in-spatial  : held-out CITIES not seen in training (no within-city leakage)
  * out         : all OTHER regions' tiles (cross-region)
If in-spatial stays close to in-random and to out (home-field gap ~0 under the
spatial split too), the no-home-field finding is not a leakage artefact.

Usage: python -m scripts.eval.spatial_holdout --seeds 20260525 1 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.acquire.regions import CITIES
from scripts.common import TILE_PX
from scripts.eval.cross_region_eval import (
    EVAL_MASK_THRESH,
    TARGET_EPOCH,
    TRAIN_EPOCHS,
    build_input_tensor,
    fom_metrics,
    load_city_rasters,
)
from scripts.eval.cross_region_train import soft_jaccard
from scripts.eval.models import SimpleCNN

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
T = 0.01; PREV = TRAIN_EPOCHS[-1]
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]


def gather_city(city, stride=64):
    try:
        bu, vol, pop, _, _ = load_city_rasters(city, PROC)
    except Exception:
        return [], [], []
    H, W = bu[TARGET_EPOCH].shape
    X, GT, PV = [], [], []
    for i in range(0, H - TILE_PX + 1, stride):
        for j in range(0, W - TILE_PX + 1, stride):
            gt = bu[TARGET_EPOCH][i:i+TILE_PX, j:j+TILE_PX]
            if gt.mean() < 0.005:
                continue
            X.append(build_input_tensor(bu, vol, pop, i, j)[0])
            GT.append(gt.astype(np.float32))
            PV.append(bu[PREV][i:i+TILE_PX, j:j+TILE_PX].astype(np.float32))
    return X, GT, PV


def region_tiles(region):
    """Per-city tile lists for a region: {city_name: (X, GT, PV)}."""
    out = {}
    for c in [c for c in CITIES if c.region == region]:
        if not (PROC / region / c.name / "eval_metrics.json").exists():
            continue
        X, GT, PV = gather_city(c)
        if X:
            out[c.name] = (X, GT, PV)
    return out


def fom_on(model, tiles, bs=32):
    """Pooled FoM over a list of (X, GT, PV) tiles."""
    X = torch.stack([t for X, _, _ in tiles for t in X]) if tiles else None
    GT = [g for _, GT, _ in tiles for g in GT]
    PV = [p for _, _, PV in tiles for p in PV]
    if X is None or len(X) == 0:
        return None
    preds = []
    with torch.no_grad():
        for b0 in range(0, len(X), bs):
            preds.append(model(X[b0:b0+bs].to(DEV)).cpu().numpy()[:, 0])
    pred = np.concatenate(preds)
    B = A = C = 0
    for k in range(len(GT)):
        mask = GT[k] > EVAL_MASK_THRESH
        if mask.sum():
            fm = fom_metrics(GT[k], PV[k], pred[k], mask, T)
            B += fm["B"]; A += fm["A"]; C += fm["C"]
    return B / (A + B + C) if (A + B + C) else 0.0


def train(tiles, seed, epochs=25, bs=16):
    X = torch.stack([t for X, _, _ in tiles for t in X])
    Y = torch.from_numpy(np.stack([g for _, GT, _ in tiles for g in GT]))[:, None]
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(X); idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr_idx = idx[:int(0.85 * n)]
    Xd, Yd = X.to(DEV), Y.to(DEV)
    m = SimpleCNN(input_channels=24).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator(device=DEV).manual_seed(seed); tr_idx = tr_idx.to(DEV)
    for _ in range(epochs):
        m.train(); perm = tr_idx[torch.randperm(len(tr_idx), generator=g, device=DEV)]
        for b0 in range(0, len(perm), bs):
            b = perm[b0:b0+bs]; opt.zero_grad(); p = m(Xd[b])
            loss = nn.functional.mse_loss(p, Yd[b]) + 0.5 * soft_jaccard(p, Yd[b])
            loss.backward(); opt.step()
    m.eval(); return m, idx[int(0.85 * n):]   # also the held-out random tiles


def main(seeds):
    data = {r: region_tiles(r) for r in REGIONS}
    data = {r: d for r, d in data.items() if len(d) >= 4}     # need >=4 cities to split
    print(f"regions usable (>=4 cities): {list(data)}", flush=True)
    per_seed = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        rows = {}
        for r, cities in data.items():
            names = sorted(cities)
            half = rng.permutation(len(names))
            train_names = [names[i] for i in half[: len(names)//2]]
            test_names = [names[i] for i in half[len(names)//2:]]
            train_tiles = [cities[n] for n in train_names]
            m, _ = train(train_tiles, sd)
            in_spatial = fom_on(m, [cities[n] for n in test_names])      # held-out CITIES
            in_random = fom_on(m, train_tiles)                          # same cities (upper bound)
            out_tiles = [data[r2][n] for r2 in data if r2 != r for n in data[r2]]
            out = fom_on(m, out_tiles)
            rows[r] = {"in_spatial": in_spatial, "in_random": in_random, "out": out}
        ins = np.mean([rows[r]["in_spatial"] for r in rows])
        inr = np.mean([rows[r]["in_random"] for r in rows])
        ot = np.mean([rows[r]["out"] for r in rows])
        per_seed.append({"in_spatial": float(ins), "in_random": float(inr), "out": float(ot),
                         "gap_spatial": float(ins - ot), "per_region": rows})
        print(f"[seed {sd}] in-spatial={ins:.3f}  in-random={inr:.3f}  out={ot:.3f}  "
              f"spatial home-field gap={ins-ot:+.3f}", flush=True)
    agg = {k: {"mean": float(np.mean([s[k] for s in per_seed])),
               "sd": float(np.std([s[k] for s in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0}
           for k in ("in_spatial", "in_random", "out", "gap_spatial")}
    out = {"seeds": seeds, "per_seed": per_seed, "aggregate": agg}
    fn = REPO / "results/metrics/spatial_holdout.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"\n=== Spatial block-holdout, {len(seeds)} seeds ===")
    print(f"  in-region (spatial holdout) {agg['in_spatial']['mean']:.3f}±{agg['in_spatial']['sd']:.3f}  "
          f"vs out-of-region {agg['out']['mean']:.3f}±{agg['out']['sd']:.3f}  "
          f"=> home-field gap {agg['gap_spatial']['mean']:+.3f}±{agg['gap_spatial']['sd']:.3f}", flush=True)
    print(f"  (in-random {agg['in_random']['mean']:.3f}: leakage inflation = "
          f"{agg['in_random']['mean']-agg['in_spatial']['mean']:+.3f})", flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seeds", type=int, nargs="+", default=[20260525, 1, 2])
    main(p.parse_args().seeds)
