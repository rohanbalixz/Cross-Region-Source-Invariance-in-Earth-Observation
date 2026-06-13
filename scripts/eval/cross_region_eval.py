"""Cross-region inference + per-tile FoM for CONUS-trained baselines.

For each city, reads the per-city UTM rasters prepared by
`scripts.preprocess.build_city_tiffs`, tiles them on the same grid the
covariate scripts use (`scripts.common.enumerate_tiles_from_grid`), runs the
three CONUS-trained baselines (CNN, U-Net, ConvLSTM) tile by tile, and writes
per-tile metrics.

This is the SCM's `Ŷ = f_CONUS(X)` and `E = ℓ(Ŷ, Y)` plumbing — nothing
more. Statistical attribution (stratification, bootstrap, Wilcoxon) lives in
a separate `attribution.py` to be written after this script's outputs land.

Run:
    python -m scripts.eval.cross_region_eval

Outputs:
    data/processed/{region}/{city}/eval_metrics.json

Each per-tile record carries `tile_id`, the per-model predicted vs ground
truth means, MSE, FoM at four thresholds, and precision/recall — enough to
reproduce stratum-matched gaps downstream.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.acquire.regions import CITIES, City
from scripts.common import (
    TILE_PX,
    enumerate_tiles_from_grid, tile_ref_to_dict, write_tile_records,
)
from scripts.eval.models import load_models

warnings.filterwarnings("ignore")

TRAIN_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH = 2015
EVAL_MASK_THRESH = 0.01
FOM_THRESHOLDS = [0.005, 0.01, 0.02, 0.05]


def fom_metrics(gt: np.ndarray, prev: np.ndarray, pred: np.ndarray,
                mask: np.ndarray, t: float) -> Dict[str, float]:
    obs_ch = (gt - prev) > t
    pred_ch = (pred - prev) > t
    obs_ch &= mask
    B = int((obs_ch & pred_ch).sum())
    A = int((obs_ch & ~pred_ch).sum())
    C = int((~obs_ch & pred_ch & mask).sum())
    denom = A + B + C
    fom = B / denom if denom > 0 else 0.0
    prec = B / (B + C) if (B + C) > 0 else 0.0
    rec  = B / (A + B) if (A + B) > 0 else 0.0
    return {
        "fom": fom, "precision": prec, "recall": rec,
        "A": A, "B": B, "C": C,
    }


def load_city_rasters(city: City, processed_root: Path):
    """Read all 9 epochs (8 training + target) into memory as float32 arrays."""
    city_dir = processed_root / city.region / city.name
    bu = {}
    vol = {}
    pop = {}
    for yr in TRAIN_EPOCHS + [TARGET_EPOCH]:
        with rasterio.open(str(city_dir / f"builtup_{yr}.tif")) as src:
            bu[yr] = src.read(1).astype(np.float32)
            transform = src.transform
            utm_crs = src.crs.to_string()
        if yr in TRAIN_EPOCHS:
            with rasterio.open(str(city_dir / f"builtvol_{yr}.tif")) as src:
                vol[yr] = src.read(1).astype(np.float32)
            with rasterio.open(str(city_dir / f"pop_{yr}.tif")) as src:
                pop[yr] = src.read(1).astype(np.float32)
    # Per-region log1p-max normalization of volume/pop to [0,1], matching the
    # parent transfer eval (scripts/eval_lagos_transfer.py). build_city_tiffs
    # stores log1p(volume/pop) but does NOT divide by the per-region max, so
    # city channels sit at ~[0,12]/[0,7] while CONUS (and the model's training
    # data) are [0,1]. Normalizing here makes every region [0,1] -> consistent
    # input scale across CONUS and all cities.
    vol_max = max((v.max() for v in vol.values()), default=1.0) or 1.0
    pop_max = max((p.max() for p in pop.values()), default=1.0) or 1.0
    vol = {yr: np.clip(v / vol_max, 0.0, 1.0) for yr, v in vol.items()}
    pop = {yr: np.clip(p / pop_max, 0.0, 1.0) for yr, p in pop.items()}
    return bu, vol, pop, transform, utm_crs


def build_input_tensor(bu, vol, pop, i: int, j: int) -> torch.Tensor:
    frames = []
    for yr in TRAIN_EPOCHS:
        frames.append(np.stack([
            bu[yr][i:i + TILE_PX, j:j + TILE_PX],
            vol[yr][i:i + TILE_PX, j:j + TILE_PX],
            pop[yr][i:i + TILE_PX, j:j + TILE_PX],
        ], axis=0))
    seq = np.stack(frames, axis=0)        # (8, 3, H, W)
    return torch.from_numpy(seq[None].astype(np.float32))  # (1, 8, 3, H, W)


def per_tile_metrics(pred: np.ndarray, gt: np.ndarray, prev: np.ndarray,
                     mask: np.ndarray) -> Dict[str, object]:
    rec: Dict[str, object] = {}
    if mask.sum() == 0:
        rec["n_eval_px"] = 0
        rec["mse"] = None
        rec["pred_mean"] = None
        rec["gt_mean"] = None
        rec["fom"] = {f"t={t}": None for t in FOM_THRESHOLDS}
        return rec
    err = (pred - gt) ** 2
    rec["n_eval_px"] = int(mask.sum())
    rec["mse"] = float(err[mask].mean())
    rec["pred_mean"] = float(pred[mask].mean())
    rec["gt_mean"]   = float(gt[mask].mean())
    rec["fom"] = {
        f"t={t}": fom_metrics(gt, prev, pred, mask, t) for t in FOM_THRESHOLDS
    }
    return rec


def process_city(city: City, processed_root: Path, models: dict,
                 out_path: Path, device: torch.device) -> None:
    city_dir = processed_root / city.region / city.name
    missing = [
        p for p in (
            [city_dir / f"builtup_{yr}.tif" for yr in TRAIN_EPOCHS + [TARGET_EPOCH]] +
            [city_dir / f"builtvol_{yr}.tif" for yr in TRAIN_EPOCHS] +
            [city_dir / f"pop_{yr}.tif" for yr in TRAIN_EPOCHS]
        ) if not p.exists()
    ]
    if missing:
        print(f"  SKIP {city.name}: missing {len(missing)} rasters; first: {missing[0].name}")
        return

    bu, vol, pop, transform, utm_crs = load_city_rasters(city, processed_root)

    refs = enumerate_tiles_from_grid(
        builtup_2015=bu[TARGET_EPOCH], utm_transform=transform,
        city_name=city.name, region=city.region, utm_crs=utm_crs,
    )

    gt_map = bu[TARGET_EPOCH]
    prev_map = bu[TRAIN_EPOCHS[-1]]

    records = []
    for ref in refs:
        i, j = ref.i, ref.j
        gt_tile = gt_map[i:i + TILE_PX, j:j + TILE_PX]
        prev_tile = prev_map[i:i + TILE_PX, j:j + TILE_PX]
        mask = gt_tile > EVAL_MASK_THRESH

        x = build_input_tensor(bu, vol, pop, i, j).to(device)

        model_outputs: Dict[str, Dict[str, object]] = {}
        for name, model in models.items():
            model.to(device)
            with torch.no_grad():
                pred = model(x).squeeze().cpu().numpy()
            model_outputs[name] = per_tile_metrics(pred, gt_tile, prev_tile, mask)

        rec = tile_ref_to_dict(ref)
        rec["n_eval_px"] = int(mask.sum())
        rec["gt_mean"] = float(gt_tile[mask].mean()) if mask.any() else None
        rec["prev_mean"] = float(prev_tile[mask].mean()) if mask.any() else None
        rec["models"] = model_outputs
        records.append(rec)

    write_tile_records(records, out_path)
    print(f"  {city.name}: {len(records)} tiles -> {out_path}")
    gc.collect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--weights", type=Path,
                   default=Path("../models"),
                   help="Path to CONUS-trained checkpoints.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu"))
    p.add_argument("--city", action="append", default=[])
    p.add_argument("--region", action="append", default=[],
                   help="Evaluate only cities in the named region(s).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    processed_root = repo_root / args.processed
    weights_root = (repo_root / args.weights).resolve()
    device = torch.device(args.device)

    print(f"loading models from {weights_root} on {device} ...")
    models = load_models(weights_root)
    print(f"loaded: {list(models)}")

    cities = list(CITIES)
    if args.city:
        cities = [c for c in cities if c.name in args.city]
    if args.region:
        cities = [c for c in cities if c.region in set(args.region)]
    for city in cities:
        out = processed_root / city.region / city.name / "eval_metrics.json"
        process_city(city, processed_root, models, out, device)
    print("done.")
