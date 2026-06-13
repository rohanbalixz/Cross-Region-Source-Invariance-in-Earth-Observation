"""Per-tile settlement-morphology covariates Z_S from the GHSL built-up raster.

For each city, read the 2015 built-up raster prepared by
`scripts.preprocess.build_city_tiffs` and compute four scalar morphology
metrics for every benchmark tile:

  - builtup_frac:   mean built-up fraction within the tile
  - fragmentation:  edge density of built-up patches per km (scikit-image
                    region-boundary length / tile area)
  - compactness:    mean isoperimetric ratio across patches with >= 5 pixels
  - contiguity:     size of the largest built-up component / total built-up

Run:
    python -m scripts.covariates.settlement

Outputs:
    data/processed/{region}/{city}/settlement.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.acquire.regions import CITIES, City
from scripts.common import (
    TILE_PX,
    TILE_RES_M,
    enumerate_tiles_from_grid,
    tile_ref_to_dict,
    write_tile_records,
)

BUILTUP_THRESHOLD = 0.3   # binarize at >= 0.3 built-up fraction


def patch_stats(binary: np.ndarray) -> tuple[float, float, float]:
    """Return (fragmentation_km_per_km2, mean_compactness, contiguity).

    fragmentation: edge length (in pixels) / tile area (km^2), converted to
                   km via cell size.
    compactness:   mean of 4*pi*area / perimeter^2 over connected components
                   with >= 5 pixels; 1.0 = perfect circle.
    contiguity:    largest-component pixel count / total built-up pixels.
    """
    if binary.sum() == 0:
        return 0.0, 0.0, 0.0

    # 4-connected labels
    labels, n_labels = ndimage.label(binary, structure=np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    # edge pixels: built-up cells with any non-built-up 4-neighbor
    pad = np.pad(binary.astype(np.uint8), 1, mode="constant", constant_values=0)
    neighbor_any_zero = (
        (pad[:-2, 1:-1] == 0) | (pad[2:, 1:-1] == 0) |
        (pad[1:-1, :-2] == 0) | (pad[1:-1, 2:] == 0)
    )
    edge_pixels = int((binary.astype(bool) & neighbor_any_zero).sum())
    edge_len_m = edge_pixels * float(TILE_RES_M)
    tile_area_km2 = (TILE_PX * TILE_RES_M / 1000.0) ** 2
    fragmentation = (edge_len_m / 1000.0) / tile_area_km2  # km / km^2

    sizes = ndimage.sum(binary, labels, range(1, n_labels + 1))
    total = float(binary.sum())
    contiguity = float(sizes.max() / total) if total > 0 else 0.0

    compactness_vals = []
    for k in range(1, n_labels + 1):
        if sizes[k - 1] < 5:
            continue
        component = (labels == k)
        area_px = int(component.sum())
        comp_pad = np.pad(component.astype(np.uint8), 1, mode="constant")
        perim_px = int((
            (comp_pad[:-2, 1:-1] == 0) | (comp_pad[2:, 1:-1] == 0) |
            (comp_pad[1:-1, :-2] == 0) | (comp_pad[1:-1, 2:] == 0)
        )[component].sum())
        if perim_px > 0:
            iso = 4.0 * math.pi * area_px / (perim_px ** 2)
            compactness_vals.append(min(iso, 1.0))
    mean_comp = float(np.mean(compactness_vals)) if compactness_vals else 0.0

    return fragmentation, mean_comp, contiguity


def process_city(city: City, processed_root: Path, out_path: Path) -> None:
    bu_path = processed_root / city.region / city.name / "builtup_2015.tif"
    if not bu_path.exists():
        print(f"  SKIP {city.name}: missing {bu_path}")
        return

    with rasterio.open(str(bu_path)) as bu:
        builtup = bu.read(1).astype(np.float32)
        transform = bu.transform
        utm_crs = bu.crs.to_string()

    refs = enumerate_tiles_from_grid(
        builtup_2015=builtup, utm_transform=transform,
        city_name=city.name, region=city.region, utm_crs=utm_crs,
    )

    records = []
    for ref in refs:
        i, j = ref.i, ref.j
        blk = builtup[i:i + TILE_PX, j:j + TILE_PX]
        binary = (blk >= BUILTUP_THRESHOLD).astype(np.uint8)
        frag, comp, cont = patch_stats(binary)
        rec = tile_ref_to_dict(ref)
        rec.update({
            "builtup_frac": float(blk.mean()),
            "fragmentation": frag,
            "compactness": comp,
            "contiguity": cont,
        })
        records.append(rec)

    write_tile_records(records, out_path)
    print(f"  {city.name}: {len(records)} tiles -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--city", action="append", default=[])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    processed_root = repo_root / args.processed

    cities = CITIES if not args.city else [c for c in CITIES if c.name in args.city]
    for city in cities:
        out = processed_root / city.region / city.name / "settlement.json"
        process_city(city, processed_root, out)
    print("done.")
