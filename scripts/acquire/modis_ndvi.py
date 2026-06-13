"""Non-urban temporal target: per-city MODIS MOD13Q1 annual max-NDVI series
(250 m, same grid as GHSL). This is the held-out endogenous task for the
input-representation law: vegetation dynamics are non-urban and non-monotonic
(NDVI rises and falls year to year), so source-invariance here cannot be a
built-up-area-grows-next-to-itself artefact. Pre-registered prediction:
retention ~0.99 (source-invariant), like the GHSL temporal tasks.

Smoke: python -m scripts.acquire.modis_ndvi --city sydney --pilot
Full:  python -m scripts.acquire.modis_ndvi --city <name>
Output: data/ndvi/<region>/<city>/ndvi_series.npz  (ndvi:(T,H,W) float32 [0,1], years)
"""
import argparse, time
from pathlib import Path
import numpy as np
import rioxarray  # noqa: F401 (registers .rio)
import odc.stac
from pystac_client import Client
import planetary_computer as pc
from scripts.acquire.regions import city_by_name

REPO = Path(__file__).resolve().parents[2]
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLL = "modis-13Q1-061"          # MOD13Q1 v061, 250 m, 16-day
NDVI = "250m_16_days_NDVI"       # int16, scale 1e-4, valid [-2000,10000]
YEARS = list(range(2013, 2023))  # 2013-2022; input 2013-2021 -> predict 2022


def _retry(fn, what, tries=8, base=10, cap=180):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1:
                raise
            wait = min(base * (2 ** k), cap)
            print(f"  [{what}] transient ({type(e).__name__}); retry {k+1}/{tries} in {wait}s", flush=True)
            time.sleep(wait)


def acquire_city(name, pilot=False, window_deg=0.60, res=250.0):
    city = city_by_name(name)
    out = REPO / f"data/ndvi/{city.region}/{name}"
    if (out / "ndvi_series.npz").exists():
        print(f"[{name}] already present, skipping", flush=True); return
    minx, miny, maxx, maxy = city.bbox
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    h = (0.15 if pilot else window_deg) / 2
    bbox = [cx - h, cy - h, cx + h, cy + h]
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    items = _retry(lambda: list(cat.search(collections=[COLL], bbox=bbox,
                   datetime=f"{YEARS[0]}-01-01/{YEARS[-1]}-12-31").items()), "modis-search")
    print(f"[{name}] {len(items)} MOD13Q1 scenes {YEARS[0]}-{YEARS[-1]}", flush=True)
    if not items:
        print(f"[{name}] NO ITEMS - skip", flush=True); return
    da = odc.stac.load(items, bands=[NDVI], bbox=bbox, resolution=res,
                       chunks={"x": 512, "y": 512}, groupby="solar_day")[NDVI]
    frames = []
    for y in YEARS:
        sub = da.sel(time=str(y))
        if sub.sizes.get("time", 0) == 0:
            print(f"[{name}] year {y} empty - abort city", flush=True); return
        comp = _retry(lambda s=sub: s.max(dim="time").compute(), f"max-{y}")
        v = comp.values.astype(np.float32) * 1e-4          # -> NDVI units
        frames.append(np.clip(v, 0.0, 1.0))                # veg in [0,1]
    arr = np.stack(frames, 0)                              # (T, H, W)
    arr = np.nan_to_num(arr, nan=0.0)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "ndvi_series.npz", ndvi=arr, years=np.array(YEARS))
    print(f"[{name}] saved {arr.shape} ndvi[min={arr.min():.2f} mean={arr.mean():.2f} "
          f"max={arr.max():.2f}] -> {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    acquire_city(a.city, a.pilot)
