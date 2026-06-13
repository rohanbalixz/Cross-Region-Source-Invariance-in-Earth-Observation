"""Per-tile terrain covariates Z_T = (slope_mean, slope_var, tri, elev_mean).

For each city we read the SRTM DEM acquired by `scripts.acquire.srtm`,
reproject to the city's UTM grid at 250 m, and compute the four scalar
terrain covariates for every benchmark tile.

Tile enumeration uses the 2015 GHSL built-up raster as the anchor — same
grid the model and the other covariate scripts see, so per-tile records
join cleanly downstream.

Run:
    python -m scripts.covariates.terrain

Outputs:
    data/processed/{region}/{city}/terrain.json

Each record:
    {
      "tile_id": "...", "city": "...", "region": "...",
      "i": int, "j": int,
      "bbox_utm": [x_min, y_min, x_max, y_max],
      "utm_crs": "EPSG:326XX",
      "slope_mean": float, "slope_var": float,
      "tri": float, "elev_mean": float
    }
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.acquire.regions import CITIES, City
from scripts.common import (
    TILE_PX,
    TILE_RES_M,
    enumerate_tiles_from_grid,
    tile_ref_to_dict,
    write_tile_records,
)


def reproject_srtm_to_city_grid(
    srtm_path: Path, builtup_2015_path: Path,
) -> tuple[np.ndarray, rasterio.Affine, str]:
    """Resample the SRTM DEM into the *exact same* UTM grid as the city's
    2015 built-up raster, so SRTM pixels align 1:1 with the model tiles."""
    with rasterio.open(str(builtup_2015_path)) as bu:
        dst_transform = bu.transform
        height, width = bu.height, bu.width
        utm_crs = bu.crs.to_string()

    dst = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(str(srtm_path)) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=CRS.from_string(utm_crs),
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return dst, dst_transform, utm_crs


def compute_slope_deg(elev: np.ndarray, cell_m: float) -> np.ndarray:
    """Slope magnitude in degrees via central differences (Horn 1981)."""
    elev = np.where(np.isfinite(elev), elev, 0.0).astype(np.float32)
    dz_dy, dz_dx = np.gradient(elev, cell_m, cell_m)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    return np.degrees(slope_rad).astype(np.float32)


def compute_tri(elev: np.ndarray) -> np.ndarray:
    """Riley 1999 Terrain Ruggedness Index: mean |center - neighbor| over a
    3x3 window. Returns a TRI raster the same shape as `elev`."""
    elev = np.where(np.isfinite(elev), elev, 0.0).astype(np.float32)
    diffs = np.zeros_like(elev, dtype=np.float32)
    counts = np.zeros_like(elev, dtype=np.float32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(np.roll(elev, di, axis=0), dj, axis=1)
            d = np.abs(elev - shifted)
            # mask out wrap-around border by zeroing rows/cols beyond grid
            if di == -1:
                d[-1, :] = 0
                counts_mask = np.ones_like(d); counts_mask[-1, :] = 0
            elif di == 1:
                d[0, :] = 0
                counts_mask = np.ones_like(d); counts_mask[0, :] = 0
            else:
                counts_mask = np.ones_like(d)
            if dj == -1:
                d[:, -1] = 0
                counts_mask[:, -1] = 0
            elif dj == 1:
                d[:, 0] = 0
                counts_mask[:, 0] = 0
            diffs += d
            counts += counts_mask
    return np.where(counts > 0, diffs / counts, 0.0).astype(np.float32)


def process_city(city: City, processed_root: Path, srtm_root: Path,
                 out_path: Path) -> None:
    bu_path = processed_root / city.region / city.name / "builtup_2015.tif"
    srtm_path = srtm_root / f"{city.name}.tif"
    if not bu_path.exists():
        print(f"  SKIP {city.name}: missing {bu_path}")
        return
    if not srtm_path.exists():
        print(f"  SKIP {city.name}: missing {srtm_path}")
        return

    elev, transform, utm_crs = reproject_srtm_to_city_grid(srtm_path, bu_path)
    with rasterio.open(str(bu_path)) as bu:
        builtup = bu.read(1).astype(np.float32)

    refs = enumerate_tiles_from_grid(
        builtup_2015=builtup, utm_transform=transform,
        city_name=city.name, region=city.region, utm_crs=utm_crs,
    )

    slope = compute_slope_deg(elev, cell_m=float(TILE_RES_M))
    tri = compute_tri(elev)

    records = []
    for ref in refs:
        i, j = ref.i, ref.j
        e_blk = elev[i:i + TILE_PX, j:j + TILE_PX]
        s_blk = slope[i:i + TILE_PX, j:j + TILE_PX]
        t_blk = tri[i:i + TILE_PX, j:j + TILE_PX]
        rec = tile_ref_to_dict(ref)
        rec.update({
            "slope_mean": float(np.nanmean(s_blk)),
            "slope_var":  float(np.nanvar(s_blk)),
            "tri":        float(np.nanmean(t_blk)),
            "elev_mean":  float(np.nanmean(e_blk)),
        })
        records.append(rec)

    write_tile_records(records, out_path)
    print(f"  {city.name}: {len(records)} tiles -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--srtm", type=Path, default=Path("data/raw/srtm"))
    p.add_argument("--city", action="append", default=[])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    processed_root = repo_root / args.processed
    srtm_root = repo_root / args.srtm

    cities = CITIES if not args.city else [c for c in CITIES if c.name in args.city]
    for city in cities:
        out = processed_root / city.region / city.name / "terrain.json"
        process_city(city, processed_root, srtm_root, out)
    print("done.")
