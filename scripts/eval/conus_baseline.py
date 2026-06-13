"""CONUS pooled-baseline FoM through the same inference path as other regions.

Reads the preprocessed CONUS rasters at
`data/processed/conus/CONUS_{builtup,volume,population}_{year}.tif` (EPSG:5070,
250 m), tiles them on the half-stride grid with the standard 0.01 built-up
filter, runs the three CONUS-trained baselines, and writes per-tile metrics.

This provides the *reference* FoM used in pooled-gap statements like
"CNN FoM drops from CONUS X to South Asia Y." It does NOT include CONUS
terrain/transport/settlement covariates — building those would require a
US-wide DEM and OSM extract, which is out of scope for the workshop pilot.
Within-test stratified attribution remains anchored on East Asia (the
lowest-shift, most CONUS-like region in our test set).

Run:
    python -m scripts.eval.conus_baseline --max-tiles 500 \
        --weights ../models

Outputs:
    data/processed/conus/conus/eval_metrics.json
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common import (
    TILE_BUILTUP_THRESHOLD,
    TILE_PX,
    TILE_STRIDE,
    TileRef,
    tile_ref_to_dict,
    write_tile_records,
)
from scripts.eval.cross_region_eval import (
    EVAL_MASK_THRESH,
    TARGET_EPOCH,
    TRAIN_EPOCHS,
    per_tile_metrics,
)
from scripts.eval.models import load_models


def load_conus_rasters(geotiff_root: Path):
    bu, vol, pop = {}, {}, {}
    for yr in TRAIN_EPOCHS + [TARGET_EPOCH]:
        with rasterio.open(str(geotiff_root / f"CONUS_builtup_{yr}.tif")) as src:
            bu[yr] = src.read(1).astype(np.float32)
            transform = src.transform
            utm_crs = src.crs.to_string()
        if yr in TRAIN_EPOCHS:
            with rasterio.open(str(geotiff_root / f"CONUS_volume_{yr}.tif")) as src:
                vol[yr] = src.read(1).astype(np.float32)
            with rasterio.open(str(geotiff_root / f"CONUS_population_{yr}.tif")) as src:
                pop[yr] = src.read(1).astype(np.float32)
    return bu, vol, pop, transform, utm_crs


def build_input(bu, vol, pop, i: int, j: int) -> torch.Tensor:
    frames = []
    for yr in TRAIN_EPOCHS:
        frames.append(np.stack([
            bu[yr][i:i + TILE_PX, j:j + TILE_PX],
            vol[yr][i:i + TILE_PX, j:j + TILE_PX],
            pop[yr][i:i + TILE_PX, j:j + TILE_PX],
        ], axis=0))
    seq = np.stack(frames, axis=0)
    return torch.from_numpy(seq[None].astype(np.float32))


def main(geotiff_root: Path, out_path: Path, weights_root: Path,
         max_tiles: int, seed: int, device: torch.device) -> None:
    bu, vol, pop, transform, utm_crs = load_conus_rasters(geotiff_root)
    print(f"CONUS grid: {bu[TARGET_EPOCH].shape}, CRS={utm_crs}")

    gt = bu[TARGET_EPOCH]
    h, w = gt.shape

    refs = []
    for i in range(0, h - TILE_PX + 1, TILE_STRIDE):
        for j in range(0, w - TILE_PX + 1, TILE_STRIDE):
            block = gt[i:i + TILE_PX, j:j + TILE_PX]
            if block.mean() <= TILE_BUILTUP_THRESHOLD:
                continue
            x_min, y_max = transform * (j, i)
            x_max, y_min = transform * (j + TILE_PX, i + TILE_PX)
            refs.append(TileRef(
                tile_id=f"conus_{i:05d}_{j:05d}", city="conus", region="conus",
                i=i, j=j,
                bbox_utm=(float(x_min), float(y_min), float(x_max), float(y_max)),
                utm_crs=utm_crs,
            ))
    print(f"CONUS candidate tiles (builtup>{TILE_BUILTUP_THRESHOLD}): {len(refs)}")

    rng = np.random.default_rng(seed)
    if max_tiles and len(refs) > max_tiles:
        idx = rng.choice(len(refs), size=max_tiles, replace=False)
        idx.sort()
        refs = [refs[k] for k in idx]
    print(f"sampling {len(refs)} CONUS tiles")

    print(f"loading models from {weights_root} on {device}...")
    models = load_models(weights_root)
    print(f"loaded: {list(models)}")

    records = []
    prev_map = bu[TRAIN_EPOCHS[-1]]
    for n, ref in enumerate(refs):
        i, j = ref.i, ref.j
        gt_tile = gt[i:i + TILE_PX, j:j + TILE_PX]
        prev_tile = prev_map[i:i + TILE_PX, j:j + TILE_PX]
        mask = gt_tile > EVAL_MASK_THRESH

        x = build_input(bu, vol, pop, i, j).to(device)
        model_outputs = {}
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

        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(refs)} tiles processed")

    write_tile_records(records, out_path)
    print(f"wrote {out_path}")
    gc.collect()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geotiff", type=Path,
                   default=Path("data/processed/conus"),
                   help="Where the preprocessed CONUS rasters live.")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/conus/conus/eval_metrics.json"))
    p.add_argument("--weights", type=Path, default=Path("../models"))
    p.add_argument("--max-tiles", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260525)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    main(
        geotiff_root=(repo_root / args.geotiff).resolve(),
        out_path=repo_root / args.out,
        weights_root=(repo_root / args.weights).resolve(),
        max_tiles=args.max_tiles,
        seed=args.seed,
        device=torch.device(args.device),
    )
