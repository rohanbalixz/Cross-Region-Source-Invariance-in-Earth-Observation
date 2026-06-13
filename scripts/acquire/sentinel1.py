"""Acquire Sentinel-1 RTC (radiometrically terrain-corrected gamma-0) SAR for
each city's tile centroids, ~2015, dual-pol VV+VH. This is the non-optical
modality control: SAR is an exogenous raw sensor, like optical imagery, but
carries no spectral/optical information at all. If SAR->built-up retention is
also source-DEPENDENT (~0.8, like optical), the imagery-side source-dependence
is a property of raw-sensor inputs in general, not an optical quirk -- which is
the breadth result this study needs.

Mirrors scripts.acquire.landsat8: stream COGs from Microsoft Planetary Computer
(no creds), crop a 6.72 km window (matching the L8 control's CENTER extent)
around each tile centroid, decimate to 64x64, and save per city:

    data/raw/sentinel1/{city}.npz
        patches:  (n_tiles, 2, 64, 64) float32   # [VV, VH] gamma-0 dB, normalised [0,1]
        tile_ids: (n_tiles,)           str

Run:
    python -m scripts.acquire.sentinel1                       # all source-region cities
    python -m scripts.acquire.sentinel1 --region ssa
    python -m scripts.acquire.sentinel1 --city nairobi
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
OUT = REPO / "data/raw/sentinel1"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-1-rtc"
WINDOW = ("2014-10-01", "2016-03-31")     # centred on 2015 (S1A from late 2014)
EXTENT_M = 6720.0                          # 6.72 km, matches the L8 control's CENTER_PX
PATCH = 64
DB_LO, DB_HI = -30.0, 5.0                  # gamma-0 dB clip for [0,1] normalisation
# 8 source regions, matching the matched-input transfer matrix
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]


def _to_db_norm(power):
    """RTC linear gamma-0 power -> dB -> [0,1]; nodata/non-finite -> 0."""
    p = power.astype(np.float32)
    p[~np.isfinite(p)] = 0.0
    p[p <= 0] = np.nan
    db = 10.0 * np.log10(p)
    db = np.clip((db - DB_LO) / (DB_HI - DB_LO), 0.0, 1.0)
    return np.nan_to_num(db, nan=0.0)


def _read_patch(href, lon, lat):
    """Read a 6.72 km window centred on (lon,lat), decimated to PATCH px.
    Returns None if the centroid is outside the scene footprint (all nodata)."""
    with rasterio.open(href) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs.to_string(), [lon], [lat])
        cx, cy = xs[0], ys[0]
        h = EXTENT_M / 2.0
        try:
            win = window_from_bounds(cx - h, cy - h, cx + h, cy + h, src.transform)
        except Exception:
            return None
        arr = src.read(1, window=win, out_shape=(PATCH, PATCH), boundless=True,
                       fill_value=0).astype(np.float32)
    if not np.isfinite(arr).any() or (arr == 0).mean() > 0.6:
        return None
    return arr


def _dual_pol_items(client, bbox):
    """Items in the window that carry BOTH vv and vh, nearest mid-2015 first."""
    search = client.search(collections=[COLLECTION], bbox=bbox,
                           datetime=f"{WINDOW[0]}/{WINDOW[1]}", max_items=80)
    items = [it for it in search.items()
             if "vv" in it.assets and "vh" in it.assets]
    mid = np.datetime64("2015-07-01")
    items.sort(key=lambda it: abs(np.datetime64(it.datetime.replace(tzinfo=None)) - mid))
    return items


def acquire_city(client, city: City) -> bool:
    import planetary_computer
    out_path = OUT / f"{city.name}.npz"
    if out_path.exists():
        print(f"  {city.name}: already acquired", flush=True); return True
    centers = _load_tile_centers(city, PROC)
    if not centers:
        print(f"  {city.name}: no eval_metrics tiles; skip", flush=True); return False
    items = _dual_pol_items(client, city.bbox)
    if not items:
        print(f"  {city.name}: no dual-pol RTC scenes in window; skip", flush=True); return False

    lonlat = [_utm_centroid_to_lonlat(x, y, crs) for (_, x, y, crs) in centers]
    patches = np.zeros((len(centers), 2, PATCH, PATCH), dtype=np.float32)
    got = np.zeros(len(centers), dtype=bool)
    # Walk scenes nearest-2015 first; fill each tile from the first scene covering it.
    for it in items:
        if got.all():
            break
        signed = planetary_computer.sign(it)
        vv_href, vh_href = signed.assets["vv"].href, signed.assets["vh"].href
        for ti, (lon, lat) in enumerate(lonlat):
            if got[ti]:
                continue
            vv = _read_patch(vv_href, lon, lat)
            if vv is None:
                continue
            vh = _read_patch(vh_href, lon, lat)
            if vh is None:
                continue
            patches[ti, 0] = _to_db_norm(vv)
            patches[ti, 1] = _to_db_norm(vh)
            got[ti] = True
    n = int(got.sum())
    # Keep any city with real coverage. Small cities (e.g. high-Andes Arequipa,
    # Cusco) legitimately have only ~4 eval tiles; skipping on an absolute count
    # would silently drop them. Only a genuine coverage failure (no tile readable,
    # or under half the tiles) is skipped.
    if n == 0 or n < 0.5 * len(centers):
        print(f"  {city.name}: only {n}/{len(centers)} tiles covered; skip", flush=True); return False
    keep = np.where(got)[0]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, patches=patches[keep],
                        tile_ids=np.array([centers[i][0] for i in keep]))
    print(f"  {city.name}: {n}/{len(centers)} tiles from {min(len(items),8)}+ scenes "
          f"-> {out_path.name}", flush=True)
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
    cities = [c for c in CITIES if (c.name in a.city) if a.city] or \
             [c for c in CITIES if c.region in set(regions)]
    print(f"acquiring Sentinel-1 RTC for {len(cities)} cities in {regions}", flush=True)
    ok = 0
    for c in cities:
        try:
            ok += acquire_city(client, c)
        except Exception as e:
            print(f"  {c.name}: FAILED {type(e).__name__}: {e}", flush=True)
    print(f"done. {ok}/{len(cities)} cities acquired -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
