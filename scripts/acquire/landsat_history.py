"""A multi-decade raw Landsat surface-reflectance history
per tile -- the proper raw analogue of the GHSL growth history, replacing the
3-seasonal-frame L8 pull (scripts.acquire.landsat8). Four epochs (~1990, 2000,
2010, 2014) drawn from Landsat 5/7/8 Collection-2 Level-2 (common band asset
names blue/green/red/nir08/swir16/swir22), least-cloud scene per tile per epoch,
streamed from Microsoft Planetary Computer (no creds). This gives a ~25-year raw
reflectance history ending before the 2015 target, so a model is fed genuine
multi-decade raw dynamics rather than a single year's seasons.

Output: data/raw/landsat_history/{city}.npz
    patches      (n_tiles, 4, 6, 224, 224) uint16
    tile_ids     (n_tiles,)
    epoch_labels (4,)

Caveat (first hardened pass): single least-cloud scene per tile per epoch (not a
median composite), and mixed sensors per epoch (L7 post-2003 has SLC-off stripes);
both add noise but not regional bias. Median compositing is the next hardening.

Run: python -m scripts.acquire.landsat_history            # all cities
     python -m scripts.acquire.landsat_history --city lagos
"""
from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

import numpy as np

import scripts.acquire.landsat8 as L8
from scripts.acquire.regions import CITIES

REPO = Path(__file__).resolve().parents[2]

# Wide windows for scene availability; labelled by nominal epoch.
EPOCHS = [
    ("1988-01-01", "1992-12-31"),
    ("1998-01-01", "2002-12-31"),
    ("2007-01-01", "2011-12-31"),
    ("2013-06-01", "2014-12-31"),
]
EPOCH_LABELS = [1990, 2000, 2010, 2014]
PLATFORMS = ("landsat-4", "landsat-5", "landsat-7", "landsat-8", "landsat-9")


def gather_scenes(client, bbox, window):
    """All Landsat scenes (any platform) covering bbox in window, relaxing the
    cloud cap as a fallback for older/sparse coverage."""
    for cap in (30.0, 60.0, 90.0):
        items = L8._stac_search(client, bbox, window[0], window[1],
                                max_cloud=cap, platforms=PLATFORMS)
        if items:
            return items
    return []


def acquire_city(client, city, processed_root, out_dir) -> bool:
    out_path = out_dir / f"{city.name}.npz"
    if out_path.exists():
        print(f"  {city.name}: already acquired", flush=True)
        return True
    tile_centers = L8._load_tile_centers(city, processed_root)
    if not tile_centers:
        print(f"  {city.name}: no eval tiles; skip", flush=True)
        return False
    tile_lonlat = [L8._utm_centroid_to_lonlat(x, y, c)
                   for (_, x, y, c) in tile_centers]
    n = len(tile_centers)
    patches = np.zeros((n, len(EPOCHS), 6, L8.PATCH_SIZE, L8.PATCH_SIZE), dtype=np.uint16)

    for e, window in enumerate(EPOCHS):
        items = gather_scenes(client, list(city.bbox), window)
        if not items:
            print(f"    epoch {EPOCH_LABELS[e]}: NO scenes in bbox", flush=True)
            continue
        per_tile_scene = {ti: (L8._best_scene_for_point(items, lon, lat) or items[0])
                          for ti, (lon, lat) in enumerate(tile_lonlat)}
        scene_to_tiles: dict = {}
        for ti, sc in per_tile_scene.items():
            scene_to_tiles.setdefault(sc.id, []).append(ti)
        uniq = {sc.id: sc for sc in per_tile_scene.values()}
        work = []
        for sid, sc in uniq.items():
            try:
                signed = L8._build_signed_assets(sc)
            except Exception as ex:
                print(f"      {sid}: sign/assets failed ({type(ex).__name__}); skip", flush=True)
                continue
            tis = scene_to_tiles[sid]
            for b_idx, band in enumerate(L8.L8_BANDS):
                work.append((signed[band], tis, b_idx, f"{sid}:{band}"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, max(1, len(work)))) as exx:
            futs = [exx.submit(L8._read_band_for_tiles, href, tis, tile_centers, e, b_idx, patches, lp)
                    for (href, tis, b_idx, lp) in work]
            for f in concurrent.futures.as_completed(futs):
                f.result()
        print(f"    epoch {EPOCH_LABELS[e]}: {len(uniq)} scene(s)", flush=True)

    tile_ids = np.array([tc[0] for tc in tile_centers], dtype=object)
    np.savez_compressed(out_path, patches=patches, tile_ids=tile_ids,
                        epoch_labels=np.array(EPOCH_LABELS))
    nz = float((patches != 0).mean())
    print(f"    {city.name}: saved ({out_path.stat().st_size/1e6:.1f} MB, non-zero {nz:.0%})", flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", type=str, default=None)
    a = p.parse_args()
    out_dir = REPO / "data/raw/landsat_history"; out_dir.mkdir(parents=True, exist_ok=True)
    processed_root = REPO / "data/processed"
    from pystac_client import Client
    import planetary_computer
    client = Client.open(L8.STAC_URL, modifier=planetary_computer.sign_inplace)
    cities = [c for c in CITIES if c.region != "conus"]
    if a.city:
        cities = [c for c in cities if c.name == a.city]
    for c in cities:
        print(f"\n=== {c.region}/{c.name} ===", flush=True)
        try:
            acquire_city(client, c, processed_root, out_dir)
        except Exception as ex:
            print(f"  {c.name}: FAILED {type(ex).__name__}: {ex}", flush=True)
    print("\nlandsat-history acquisition: done.", flush=True)


if __name__ == "__main__":
    main()
