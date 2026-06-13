"""Harder-task acquisition: per-city Sentinel-2 median composite (input) +
ESA WorldCover land-cover labels (target), from Microsoft Planetary Computer.

Pilot: python -m scripts.acquire.worldcover_s2 --city sydney --pilot
Full:  python -m scripts.acquire.worldcover_s2 --city <name>

Outputs: data/hardtask/<region>/<city>/{s2.tif, landcover.tif}
"""
import argparse, time
from pathlib import Path
import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import odc.stac
from pystac_client import Client
import planetary_computer as pc
from scripts.acquire.regions import city_by_name


def _retry(fn, what, tries=8, base=10, cap=180):
    """Retry a Planetary-Computer call through API timeouts / outages.
    Patient enough (up to ~15 min total) to ride out a transient MPC outage."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1:
                raise
            wait = min(base * (2 ** k), cap)
            print(f"  [{what}] transient error ({type(e).__name__}); retry {k+1}/{tries} in {wait}s", flush=True)
            time.sleep(wait)

REPO = Path(__file__).resolve().parents[2]
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = ["B04", "B03", "B02", "B08"]   # R,G,B,NIR at 10 m


def acquire_city(name: str, pilot: bool, res: float = 10.0,
                 n_patches: int = 0, window_deg: float = 0.30, seed: int = 20260525):
    city = city_by_name(name)
    if n_patches and (REPO / f"data/hardtask/{city.region}/{name}/patches.npz").exists():
        print(f"[{name}] patches already present, skipping", flush=True); return
    minx, miny, maxx, maxy = city.bbox
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    if pilot:  # small central window to validate the path cheaply
        minx, maxx, miny, maxy = cx - 0.075, cx + 0.075, cy - 0.075, cy + 0.075
    elif n_patches:  # central window we then tile into 128px patches
        h = window_deg / 2
        minx, maxx, miny, maxy = cx - h, cx + h, cy - h, cy + h
    bbox = [minx, miny, maxx, maxy]
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    # southern-hemisphere summer window for a low-cloud composite; adjust per-hemi
    start, end = ("2021-11-01", "2022-02-28") if cy < 0 else ("2021-06-01", "2021-09-30")
    s2_items = _retry(lambda: list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                               datetime=f"{start}/{end}",
                               query={"eo:cloud_cover": {"lt": 10}}).items()), "s2-search")
    print(f"[{name}] {len(s2_items)} low-cloud S2 scenes in window", flush=True)
    s2 = odc.stac.load(s2_items, bands=BANDS, bbox=bbox, resolution=res,
                       chunks={"x": 1024, "y": 1024}, groupby="solar_day")
    # median composite over time, materialise, and restore the CRS that
    # to_array() drops before writing.
    comp = _retry(lambda: s2.to_array().median(dim="time").transpose("variable", "y", "x").compute(), "s2-load")
    comp = comp.rio.write_crs(s2.odc.crs)
    wc_items = _retry(lambda: list(cat.search(collections=["esa-worldcover"], bbox=bbox).items()), "wc-search")
    wc = _retry(lambda: odc.stac.load(wc_items, bands=["map"], like=s2)["map"].isel(time=0).compute(), "wc-load")
    wc = wc.rio.write_crs(s2.odc.crs)
    out = REPO / f"data/hardtask/{city.region}/{name}"; out.mkdir(parents=True, exist_ok=True)
    S = comp.values.astype(np.float32)            # (4, H, W)
    L = wc.values.astype(np.uint8)                # (H, W)
    if n_patches:
        rng = np.random.default_rng(seed); P = 128
        H, W = L.shape; ys = range(0, H - P + 1, P); xs = range(0, W - P + 1, P)
        cands = [(i, j) for i in ys for j in xs]
        good = []
        for i, j in cands:
            lp = L[i:i+P, j:j+P]; sp = S[:, i:i+P, j:j+P]
            if (lp > 0).mean() > 0.95 and np.isfinite(sp).all() and sp.std() > 0:
                good.append((i, j))
        rng.shuffle(good); good = good[:n_patches]
        Sp = np.stack([S[:, i:i+P, j:j+P] for i, j in good])
        Lp = np.stack([L[i:i+P, j:j+P] for i, j in good])
        np.savez_compressed(out / "patches.npz", s2=Sp, label=Lp)
        cls, cnt = np.unique(Lp[Lp > 0], return_counts=True)
        dist = {int(c): round(float(n / cnt.sum()), 3) for c, n in zip(cls, cnt)}
        print(f"[{name}] {len(good)} patches saved; classes {dist}", flush=True)
    else:
        comp.rio.to_raster(out / "s2.tif"); wc.rio.to_raster(out / "landcover.tif")
        cls, cnt = np.unique(L[L > 0], return_counts=True)
        print(f"[{name}] s2 {S.shape} label {L.shape}; classes "
              f"{ {int(c): round(float(n/cnt.sum()),3) for c,n in zip(cls,cnt)} }", flush=True)
    print(f"[{name}] saved -> {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--patches", type=int, default=0, help="N 128px patches/city")
    a = p.parse_args()
    acquire_city(a.city, a.pilot, n_patches=a.patches)
