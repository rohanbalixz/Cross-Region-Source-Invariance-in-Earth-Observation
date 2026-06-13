"""Download Copernicus DEM (GLO-30, 30 m) tiles per city bounding box.

We use the AWS S3 mirror of the Copernicus DEM 30 m product, which is publicly
accessible without authentication. Naming originally followed the OpenTopography
SRTM pattern; we kept the script name `srtm.py` and the per-city output filename
`{city}.tif` so downstream code (covariates.terrain, eval.cross_region_eval)
doesn't change.

For each city bbox we enumerate the 1° × 1° tiles that intersect the bbox,
download each, merge them into a single GeoTIFF, and clip to the bbox.

Run:
    python -m scripts.acquire.srtm

Outputs:
    data/raw/srtm/<city>.tif      (one merged GeoTIFF per city, EPSG:4326)

Note: file/dir is named "srtm" for continuity, but the source is Copernicus
DEM 30 m. The covariate computations are agnostic to source — both products
are bare-earth elevation in meters at ~30 m resolution.
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import urllib.request
from pathlib import Path

# Never let a stalled DEM tile read block forever (raises after 120s of silence).
socket.setdefaulttimeout(120)

import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds

from scripts.acquire.regions import CITIES, City

COP_DEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{lat_tag}_00_{lon_tag}_00_DEM/"
    "Copernicus_DSM_COG_10_{lat_tag}_00_{lon_tag}_00_DEM.tif"
)


def tile_id_for(lat_floor: int, lon_floor: int) -> tuple[str, str]:
    lat_tag = f"N{lat_floor:02d}" if lat_floor >= 0 else f"S{-lat_floor:02d}"
    lon_tag = f"E{lon_floor:03d}" if lon_floor >= 0 else f"W{-lon_floor:03d}"
    return lat_tag, lon_tag


def tiles_covering_bbox(bbox: tuple[float, float, float, float]):
    lon_min, lat_min, lon_max, lat_max = bbox
    lats = range(int(math.floor(lat_min)), int(math.floor(lat_max)) + 1)
    lons = range(int(math.floor(lon_min)), int(math.floor(lon_max)) + 1)
    return [(lat, lon) for lat in lats for lon in lons]


def fetch_one(lat: int, lon: int, cache_dir: Path) -> Path | None:
    lat_tag, lon_tag = tile_id_for(lat, lon)
    url = COP_DEM_URL.format(lat_tag=lat_tag, lon_tag=lon_tag)
    dest = cache_dir / f"Copernicus_DSM_COG_10_{lat_tag}_00_{lon_tag}_00_DEM.tif"
    if dest.exists():
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"    FAIL {lat_tag}_{lon_tag}: {e}", file=sys.stderr)
        return None
    return dest


def fetch_city(city: City, out_root: Path, tile_cache: Path) -> None:
    out_path = out_root / f"{city.name}.tif"
    if out_path.exists():
        print(f"  exists, skipping: {out_path.name}")
        return

    needed = tiles_covering_bbox(city.bbox)
    print(f"  {city.name}: {len(needed)} Copernicus DEM tile(s) needed")
    tile_paths: list[Path] = []
    for (lat, lon) in needed:
        p = fetch_one(lat, lon, tile_cache)
        if p is not None:
            tile_paths.append(p)
    if not tile_paths:
        print(f"    no tiles downloaded for {city.name}; skipping merge")
        return

    # merge + clip to bbox
    src_files = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff", "height": mosaic.shape[1],
        "width": mosaic.shape[2], "transform": transform,
        "compress": "deflate",
    })

    out_root.mkdir(parents=True, exist_ok=True)
    # write merged temp, then crop with a windowed read.
    tmp = out_path.with_suffix(".merge.tif")
    with rasterio.open(str(tmp), "w", **out_meta) as dst:
        dst.write(mosaic)

    with rasterio.open(str(tmp)) as src:
        window = from_bounds(
            city.bbox[0], city.bbox[1], city.bbox[2], city.bbox[3],
            transform=src.transform,
        )
        data = src.read(1, window=window)
        new_transform = src.window_transform(window)
        clip_meta = src.meta.copy()
        clip_meta.update({
            "height": data.shape[0], "width": data.shape[1],
            "transform": new_transform,
        })
        with rasterio.open(str(out_path), "w", **clip_meta) as dst:
            dst.write(data, 1)

    for s in src_files:
        s.close()
    tmp.unlink(missing_ok=True)
    print(f"    wrote {out_path.name} ({data.shape[0]}x{data.shape[1]})")


def main(out_root: Path, tile_cache: Path, only: list[str] | None = None,
         workers: int = 4) -> None:
    from concurrent.futures import ThreadPoolExecutor
    cities = CITIES if not only else [c for c in CITIES if c.name in set(only)]
    todo = [c for c in cities if not (out_root / f"{c.name}.tif").exists()]
    print(f"DEM: {len(cities)} cities, {len(todo)} to fetch, {workers} workers",
          flush=True)
    # Cities are geographically dispersed, so shared 1-degree tile-cache
    # collisions are rare; a modest pool keeps the download-bound step fast.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda c: fetch_city(c, out_root, tile_cache), todo))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw/srtm"))
    p.add_argument("--tile-cache", type=Path,
                   default=Path("data/raw/srtm/_tiles"))
    p.add_argument("--city", action="append", default=[],
                   help="Fetch only the named cities (default: all).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    main(repo_root / args.out, repo_root / args.tile_cache,
         only=args.city or None)
    print("done.")
