"""A second, independent
harmonized-semantic product on the SUCCESS side of the matched-input swap. The
positive arm of this study rests entirely on GHSL (one upstream classifier, JRC).
ESA WorldCover is a global land-cover map from a DIFFERENT upstream classifier
(ESA, Sentinel-2 based). If WorldCover-as-input -> GHSL-built-up-2015 also
retains ~1.0, the claim converts from "everything-that-isn't-GHSL fails" to
"a harmonized product from two independent sources succeeds."

Acquired on the SAME per-tile 6.72 km grid as the Landsat/SAR/NDVI controls
(2021 v200 map), one-hot over the six dominant classes -> 6 channels.

    data/raw/worldcover_aligned/{city}.npz : patches (n,6,64,64) float32, tile_ids

Run: python -m scripts.acquire.worldcover_aligned [--region ssa] [--city nairobi]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds as window_from_bounds
from scripts.acquire.regions import CITIES, City
from scripts.acquire.landsat8 import _load_tile_centers, _utm_centroid_to_lonlat

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
OUT = REPO / "data/raw/worldcover_aligned"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "esa-worldcover"; ASSET = "map"
EXTENT_M = 6720.0; PATCH = 64
CLASSES = [10, 40, 50, 80, 30, 60]   # tree, crop, BUILT-UP, water, grass, bare -> 6 one-hot channels
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]


def _onehot(cls):
    out = np.zeros((len(CLASSES), *cls.shape), dtype=np.float32)
    for ci, c in enumerate(CLASSES):
        out[ci] = (cls == c)
    return out


def _read_patch(href, lon, lat):
    import math
    with rasterio.open(href) as src:
        if src.crs.is_geographic:                       # WorldCover is EPSG:4326 (degrees)
            dlat = (EXTENT_M / 2.0) / 111320.0
            dlon = (EXTENT_M / 2.0) / (111320.0 * max(0.1, math.cos(math.radians(lat))))
            minx, miny, maxx, maxy = lon - dlon, lat - dlat, lon + dlon, lat + dlat
        else:
            xs, ys = warp_transform("EPSG:4326", src.crs.to_string(), [lon], [lat])
            h = EXTENT_M / 2.0
            minx, miny, maxx, maxy = xs[0] - h, ys[0] - h, xs[0] + h, ys[0] + h
        try:
            win = window_from_bounds(minx, miny, maxx, maxy, src.transform)
        except Exception:
            return None
        cls = src.read(1, window=win, out_shape=(PATCH, PATCH), boundless=True,
                       fill_value=0, resampling=rasterio.enums.Resampling.nearest)
    if (cls == 0).mean() > 0.6:
        return None
    return _onehot(cls)


def _map_items(client, bbox):
    s = client.search(collections=[COLLECTION], bbox=bbox, max_items=10)
    # prefer the 2021 (v200) map
    items = sorted(s.items(), key=lambda it: -int(str(it.datetime)[:4] if it.datetime else "0"))
    return items


def acquire_city(client, city: City) -> bool:
    import planetary_computer
    out_path = OUT / f"{city.name}.npz"
    if out_path.exists():
        print(f"  {city.name}: already acquired", flush=True); return True
    centers = _load_tile_centers(city, PROC)
    if not centers:
        print(f"  {city.name}: no eval tiles; skip", flush=True); return False
    items = _map_items(client, city.bbox)
    if not items:
        print(f"  {city.name}: no WorldCover items; skip", flush=True); return False
    lonlat = [_utm_centroid_to_lonlat(x, y, crs) for (_, x, y, crs) in centers]
    patches = np.zeros((len(centers), len(CLASSES), PATCH, PATCH), dtype=np.float32)
    got = np.zeros(len(centers), dtype=bool)
    for it in items:
        if got.all():
            break
        href = planetary_computer.sign(it).assets[ASSET].href
        for ti, (lon, lat) in enumerate(lonlat):
            if got[ti]:
                continue
            p = _read_patch(href, lon, lat)
            if p is not None:
                patches[ti] = p; got[ti] = True
    n = int(got.sum())
    if n == 0 or n < 0.5 * len(centers):
        print(f"  {city.name}: only {n}/{len(centers)} tiles; skip", flush=True); return False
    keep = np.where(got)[0]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, patches=patches[keep],
                        tile_ids=np.array([centers[i][0] for i in keep]))
    print(f"  {city.name}: {n}/{len(centers)} tiles -> {out_path.name}", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", action="append", default=[])
    p.add_argument("--city", action="append", default=[])
    a = p.parse_args()
    from pystac_client import Client
    import planetary_computer
    client = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    regions = a.region or REGIONS
    cities = [c for c in CITIES if c.name in a.city] if a.city else \
             [c for c in CITIES if c.region in set(regions)]
    print(f"acquiring aligned WorldCover for {len(cities)} cities", flush=True)
    ok = 0
    for c in cities:
        try:
            ok += acquire_city(client, c)
        except Exception as e:
            print(f"  {c.name}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"done. {ok}/{len(cities)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
