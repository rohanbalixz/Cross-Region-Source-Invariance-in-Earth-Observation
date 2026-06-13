"""Shared utilities for tiling, CRS choice, and covariate I/O.

Kept deliberately small. Anything beyond ~30 lines of dependency belongs in
its own module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# Tile geometry: 128 px × 250 m = 32 km square tiles, tiled with half-stride
# (TILE_PX // 2) to maximise sample size.
TILE_PX = 128
TILE_STRIDE = 64
TILE_RES_M = 250
TILE_KM = TILE_PX * TILE_RES_M / 1000.0   # 32.0

# Eval mask: only consider tiles whose 2015 built-up fraction exceeds this.
TILE_BUILTUP_THRESHOLD = 0.01


def utm_crs_for_bbox(bbox: Tuple[float, float, float, float]) -> str:
    """EPSG code of the UTM zone closest to the bbox centroid."""
    lon = (bbox[0] + bbox[2]) / 2.0
    lat = (bbox[1] + bbox[3]) / 2.0
    zone = int((lon + 180.0) / 6.0) + 1
    if zone < 1:
        zone = 1
    if zone > 60:
        zone = 60
    return f"EPSG:326{zone:02d}" if lat >= 0 else f"EPSG:327{zone:02d}"


@dataclass(frozen=True)
class TileRef:
    tile_id: str
    city: str
    region: str
    i: int            # pixel row offset in the city-grid raster
    j: int            # pixel column offset
    bbox_utm: Tuple[float, float, float, float]   # x_min, y_min, x_max, y_max
    utm_crs: str


def enumerate_tiles_from_grid(
    builtup_2015: np.ndarray,
    utm_transform,
    city_name: str,
    region: str,
    utm_crs: str,
) -> List[TileRef]:
    """Yield TileRefs for every 128x128 block whose 2015 built-up mean exceeds
    `TILE_BUILTUP_THRESHOLD`. `utm_transform` is the rasterio Affine giving
    UTM x,y of pixel centers in `builtup_2015`.
    """
    h, w = builtup_2015.shape
    refs: List[TileRef] = []
    for i in range(0, h - TILE_PX + 1, TILE_STRIDE):
        for j in range(0, w - TILE_PX + 1, TILE_STRIDE):
            block = builtup_2015[i:i + TILE_PX, j:j + TILE_PX]
            if block.mean() <= TILE_BUILTUP_THRESHOLD:
                continue
            # corner UTM coords from the affine
            x_min, y_max = utm_transform * (j, i)
            x_max, y_min = utm_transform * (j + TILE_PX, i + TILE_PX)
            refs.append(TileRef(
                tile_id=f"{city_name}_{i:05d}_{j:05d}",
                city=city_name,
                region=region,
                i=i, j=j,
                bbox_utm=(float(x_min), float(y_min), float(x_max), float(y_max)),
                utm_crs=utm_crs,
            ))
    return refs


def write_tile_records(records: Iterable[dict], path: Path) -> None:
    """Atomically write a JSON list of per-tile dicts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(list(records), f, indent=2)
    tmp.replace(path)


def tile_ref_to_dict(t: TileRef) -> dict:
    return asdict(t)
