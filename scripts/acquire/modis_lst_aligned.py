"""Provenance dissociation on a non-urban, non-land-cover,
climate biophysical target. MODIS Land Surface Temperature (MOD11A2) is a global
harmonised retrieval -- a value (Kelvin) that means the same physical thing
everywhere. We acquire annual (July-nearest) daytime LST 2011-2015 on the same
per-tile 6.72 km grid as the other controls, to test whether the
harmonised-product-transfers / raw-sensor-fails dissociation holds off built-up.

    data/raw/lst_aligned/{city}.npz : patches (n,5,64,64) float32 (LST K, normalised), tile_ids

Run: python -m scripts.acquire.modis_lst_aligned [--region ssa] [--city nairobi]
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
OUT = REPO / "data/raw/lst_aligned"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "modis-11A2-061"; ASSET = "LST_Day_1km"
YEARS = [2011, 2012, 2013, 2014, 2015]
EXTENT_M = 6720.0; PATCH = 64
LST_LO, LST_HI = 270.0, 330.0    # K clip -> [0,1]
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]


def _norm(dn):
    k = dn.astype(np.float32) * 0.02            # MOD11A2 scale -> Kelvin
    k[dn == 0] = np.nan                          # fill
    return np.clip(np.nan_to_num((k - LST_LO) / (LST_HI - LST_LO), nan=0.0), 0.0, 1.0)


def _read_patch(href, lon, lat):
    with rasterio.open(href) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs.to_string(), [lon], [lat])  # sinusoidal (m)
        cx, cy = xs[0], ys[0]; h = EXTENT_M / 2.0
        try:
            win = window_from_bounds(cx - h, cy - h, cx + h, cy + h, src.transform)
        except Exception:
            return None
        dn = src.read(1, window=win, out_shape=(PATCH, PATCH), boundless=True, fill_value=0)
    if (dn == 0).mean() > 0.6:
        return None
    return _norm(dn)


def _item_near_july(client, bbox, year):
    s = client.search(collections=[COLLECTION], bbox=bbox,
                      datetime=f"{year}-01-01/{year}-12-31", max_items=60)
    items = list(s.items())
    if not items:
        return None
    mid = np.datetime64(f"{year}-07-01")

    def when(it):
        dt = it.datetime or it.properties.get("start_datetime")
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
    ok = [False] * len(YEARS)
    for yi, year in enumerate(YEARS):
        it = _item_near_july(client, city.bbox, year)
        if it is None:
            continue
        href = planetary_computer.sign(it).assets[ASSET].href
        for ti, (lon, lat) in enumerate(lonlat):
            p = _read_patch(href, lon, lat)
            if p is not None:
                patches[ti, yi] = p
        ok[yi] = True
    if not all(ok):
        print(f"  {city.name}: missing yr {[YEARS[i] for i,o in enumerate(ok) if not o]}; skip", flush=True)
        return False
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, patches=patches, tile_ids=np.array([c[0] for c in centers]))
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
    print(f"acquiring aligned MODIS LST for {len(cities)} cities", flush=True)
    ok = 0
    for c in cities:
        try:
            ok += acquire_city(client, c)
        except Exception as e:
            print(f"  {c.name}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"done. {ok}/{len(cities)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
