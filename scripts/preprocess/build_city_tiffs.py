"""Reproject raw GHSL Mollweide tiles into per-city UTM GeoTIFFs at 250 m.

For each city in `scripts.acquire.regions.CITIES`, this script reads the raw
GHSL built-up / built-volume / population rasters acquired by
`scripts.acquire.ghsl`, clips to the city bbox, reprojects to the city's UTM
zone, resamples to 250 m, and writes:

    data/processed/{region}/{city}/builtup_{year}.tif
    data/processed/{region}/{city}/builtvol_{year}.tif   (training years only)
    data/processed/{region}/{city}/pop_{year}.tif        (training years only)

These per-city UTM tiles are then consumed by `scripts.eval.cross_region_eval`
and by the three covariate scripts.

This script does *not* run inference and does *not* compute covariates. It is
the pure data-conditioning step.

Run:
    python -m scripts.preprocess.build_city_tiffs
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import (
    Resampling,
    reproject,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.acquire.regions import CITIES, City
from scripts.common import TILE_RES_M, utm_crs_for_bbox

warnings.filterwarnings("ignore")

TRAIN_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
TARGET_EPOCH = 2015
MOLLWEIDE = (
    "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)


def bbox_in_utm(city: City, utm_crs: str):
    t = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    x0, y0 = t.transform(city.bbox[0], city.bbox[1])
    x1, y1 = t.transform(city.bbox[2], city.bbox[3])
    x_min, x_max = sorted([x0, x1])
    y_min, y_max = sorted([y0, y1])
    return x_min, y_min, x_max, y_max


def reproject_mollweide_tif(
    src_tifs: list[Path],
    city: City,
    utm_crs: str,
    out_path: Path,
    is_density: bool,
    log_transform: bool,
) -> None:
    """Mosaic / reproject the supplied Mollweide GHSL TIFFs into a UTM raster
    clipped to the city bbox at TILE_RES_M m resolution.

    Multiple `src_tifs` are passed when the bbox spans more than one GHSL
    tile; they are mosaicked in destination space (sum, since they don't
    overlap).
    """
    x_min, y_min, x_max, y_max = bbox_in_utm(city, utm_crs)
    width  = int(np.ceil((x_max - x_min) / TILE_RES_M))
    height = int(np.ceil((y_max - y_min) / TILE_RES_M))
    dst_transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    dst_data = np.zeros((height, width), dtype=np.float32)

    for src_tif in src_tifs:
        with rasterio.open(str(src_tif)) as src:
            src_arr = src.read(1).astype(np.float32)
            # Mask out nodata explicitly: GHSL R2023A uses uint16 max (65535)
            # as nodata, and bilinear reprojection contaminates valid pixels
            # near the nodata mask unless we zero it out first.
            if src.nodata is not None:
                src_arr = np.where(src_arr == src.nodata, 0.0, src_arr)
            # Drop any pathological large values (a few ghsl tiles ship with
            # int overflow at extreme antarctic latitudes; ignore them).
            src_arr = np.where(src_arr > 1e7, 0.0, src_arr)

            chunk = np.zeros((height, width), dtype=np.float32)
            reproject(
                source=src_arr,
                destination=chunk,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=CRS.from_string(utm_crs),
                resampling=Resampling.bilinear,
                src_nodata=0.0,
                dst_nodata=0.0,
            )
            dst_data += np.nan_to_num(chunk, nan=0.0,
                                      posinf=0.0, neginf=0.0)

    if is_density:
        src_cell_m2 = 100.0 * 100.0   # GHSL 100 m Mollweide
        dst_data = np.clip(dst_data / src_cell_m2, 0.0, 1.0)
    if log_transform:
        dst_data[dst_data > 1e8] = 0.0
        dst_data = np.log1p(dst_data).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(out_path), "w",
        driver="GTiff",
        height=height, width=width,
        count=1, dtype="float32",
        crs=utm_crs, transform=dst_transform,
        nodata=0.0, compress="deflate",
    ) as dst:
        dst.write(dst_data, 1)


def find_raw_tifs_for_layer(
    raw_root: Path, layer: str, epoch: int, region: str,
) -> list[Path]:
    """Return all GHSL TIFFs the acquire step downloaded for this (layer,
    epoch, region). Acquire writes ZIPs; user is expected to unzip in place.
    """
    base = raw_root / "ghsl" / layer / str(epoch) / region
    if not base.exists():
        return []
    return sorted(base.glob("**/*.tif"))


def process_city(
    city: City, raw_root: Path, out_root: Path, layers: list[str],
    skip_existing: bool,
) -> None:
    utm = utm_crs_for_bbox(city.bbox)
    print(f"\n[{city.region}/{city.name}] UTM={utm}")

    out_dir = out_root / city.region / city.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in TRAIN_EPOCHS + [TARGET_EPOCH]:
        if "BUILT_S" in layers:
            out = out_dir / f"builtup_{epoch}.tif"
            if not (skip_existing and out.exists()):
                src = find_raw_tifs_for_layer(raw_root, "BUILT_S", epoch, city.region)
                if not src:
                    print(f"  WARN: no BUILT_S source for {epoch}; skipping")
                else:
                    reproject_mollweide_tif(
                        src, city, utm, out, is_density=True, log_transform=False,
                    )
                    print(f"  wrote {out.name}")
        if epoch in TRAIN_EPOCHS and "BUILT_V" in layers:
            out = out_dir / f"builtvol_{epoch}.tif"
            if not (skip_existing and out.exists()):
                src = find_raw_tifs_for_layer(raw_root, "BUILT_V", epoch, city.region)
                if src:
                    reproject_mollweide_tif(
                        src, city, utm, out, is_density=False, log_transform=True,
                    )
                    print(f"  wrote {out.name}")
        if epoch in TRAIN_EPOCHS and "POP" in layers:
            out = out_dir / f"pop_{epoch}.tif"
            if not (skip_existing and out.exists()):
                src = find_raw_tifs_for_layer(raw_root, "POP", epoch, city.region)
                if src:
                    reproject_mollweide_tif(
                        src, city, utm, out, is_density=False, log_transform=True,
                    )
                    print(f"  wrote {out.name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--layer", action="append", default=[],
                   help="GHSL layer(s) to process. Default: BUILT_S BUILT_V POP.")
    p.add_argument("--city", action="append", default=[],
                   help="Process only the named cities (default: all).")
    p.add_argument("--region", action="append", default=[],
                   help="Process only cities in the named region(s). Combine "
                        "with workers to shard preprocessing across processes.")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-process even if output TIFFs already exist.")
    args = p.parse_args()
    if not args.layer:
        args.layer = ["BUILT_S", "BUILT_V", "POP"]
    return args


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    raw_root = repo_root / args.raw
    out_root = repo_root / args.out

    cities = list(CITIES)
    if args.city:
        cities = [c for c in cities if c.name in set(args.city)]
    if args.region:
        cities = [c for c in cities if c.region in set(args.region)]
    for city in cities:
        process_city(city, raw_root, out_root, args.layer,
                     skip_existing=not args.no_skip_existing)
    print("\ndone.")
