"""A uniform, non-classifier input. GHSL is the output of a
global classifier, so its source-invariance could be inherited from upstream
model-normalisation rather than from being a uniform product. MODIS NDVI
(MOD13Q1) is a uniformly-processed global MEASUREMENT (a band-ratio index), not
a land classifier. We acquire it on the SAME per-tile 6.72 km grid as the
Landsat/Sentinel-1 controls so it can be run through the matched-input matrix
against the built-up-2015 target.

Three annual frames (2013, 2014, 2015), the item nearest 1 July each year, read
per tile-centroid and decimated to 64x64. NDVI int16 (scale 1e-4) -> [0,1].

    data/raw/ndvi_aligned/{city}.npz : patches (n,3,64,64) float32, tile_ids

Run: python -m scripts.acquire.modis_ndvi_aligned [--region ssa] [--city nairobi]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds as window_from_bounds

from scripts.acquire.landsat8 import _load_tile_centers, _utm_centroid_to_lonlat
from scripts.acquire.regions import CITIES, City

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
OUT = REPO / "data/raw/ndvi_aligned"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "modis-13Q1-061"; ASSET = "250m_16_days_NDVI"
YEARS = [2013, 2014, 2015]
EXTENT_M = 6720.0; PATCH = 64
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]


def _ndvi_norm(dn):
    v = dn.astype(np.float32) * 1e-4          # MOD13Q1 scale
    v[dn <= -3000] = np.nan                    # fill value
    return np.clip(np.nan_to_num((v + 0.2) / 1.2, nan=0.0), 0.0, 1.0)   # ~[-0.2,1.0] -> [0,1]


def _read_patch(href, lon, lat):
    with rasterio.open(href) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs.to_string(), [lon], [lat])
        cx, cy = xs[0], ys[0]; h = EXTENT_M / 2.0
        try:
            win = window_from_bounds(cx - h, cy - h, cx + h, cy + h, src.transform)
        except Exception:
            return None
        arr = src.read(1, window=win, out_shape=(PATCH, PATCH), boundless=True,
                       fill_value=-3000)
    return _ndvi_norm(arr)


def _item_near_july(client, bbox, year):
    s = client.search(collections=[COLLECTION], bbox=bbox,
                      datetime=f"{year}-01-01/{year}-12-31", max_items=40)
    items = list(s.items())
    if not items:
        return None
    mid = np.datetime64(f"{year}-07-01")

    def when(it):
        dt = it.datetime or it.properties.get("start_datetime") or it.properties.get("end_datetime")
        return np.datetime64(str(dt)[:10]) if dt else np.datetime64(f"{year}-01-01")
    items.sort(key=lambda it: abs(when(it) - mid))
    return items[0]


def acquire_city(client, city: City) -> bool:
    import planetary_computer
    out_path = OUT / f"{city.name}.npz"
    if out_path.exists():
        print(f"  {city.name}: already acquired", flush=True); return True
    centers = _load_tile_centers(city, PROC)
    if not centers:
        print(f"  {city.name}: no eval tiles; skip", flush=True); return False
    lonlat = [_utm_centroid_to_lonlat(x, y, crs) for (_, x, y, crs) in centers]
    patches = np.zeros((len(centers), len(YEARS), PATCH, PATCH), dtype=np.float32)
    ok_year = [False] * len(YEARS)
    for yi, year in enumerate(YEARS):
        it = _item_near_july(client, city.bbox, year)
        if it is None:
            continue
        href = planetary_computer.sign(it).assets[ASSET].href
        for ti, (lon, lat) in enumerate(lonlat):
            p = _read_patch(href, lon, lat)
            if p is not None:
                patches[ti, yi] = p
        ok_year[yi] = True
    if not all(ok_year):
        print(f"  {city.name}: missing year(s) {[YEARS[i] for i,o in enumerate(ok_year) if not o]}; skip",
              flush=True); return False
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, patches=patches,
                        tile_ids=np.array([c[0] for c in centers]))
    print(f"  {city.name}: {len(centers)} tiles x {len(YEARS)} yr -> {out_path.name}", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", action="append", default=[])
    p.add_argument("--city", action="append", default=[])
    a = p.parse_args()
    import planetary_computer
    from pystac_client import Client
    client = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    regions = a.region or REGIONS
    cities = [c for c in CITIES if c.name in a.city] if a.city else \
             [c for c in CITIES if c.region in set(regions)]
    print(f"acquiring aligned MODIS NDVI for {len(cities)} cities", flush=True)
    ok = 0
    for c in cities:
        try:
            ok += acquire_city(client, c)
        except Exception as e:
            print(f"  {c.name}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"done. {ok}/{len(cities)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
