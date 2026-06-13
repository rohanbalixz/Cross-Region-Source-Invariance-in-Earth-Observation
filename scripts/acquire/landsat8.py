"""Acquire Landsat-8 Collection-2 Level-2 surface-reflectance imagery for
each city's tile centroids, at 3 timesteps spanning 2013-2014, in the
6 HLS-equivalent bands Prithvi was pretrained on:

    Landsat-8 band -> HLS equivalent (Prithvi expects in this order)
        B02 (blue)   -> HLS B02
        B03 (green)  -> HLS B03
        B04 (red)    -> HLS B04
        B05 (NIR)    -> HLS B8A   (close, slightly different bandpass)
        B06 (SWIR1)  -> HLS B11
        B07 (SWIR2)  -> HLS B12

Per-city pipeline:
  1. Query Microsoft Planetary Computer STAC for `landsat-c2-l2` items
     intersecting the city bbox, at three temporal windows
     (2013-06 to 2013-09, 2014-01 to 2014-04, 2014-07 to 2014-10).
     Pick the least-cloudy scene per window.
  2. For each existing per-tile centroid (from eval_metrics.json):
       - Project the centroid from UTM into the L8 scene's CRS
       - Crop a 224x224 patch at L8's native 30m resolution,
         centered on the tile centroid (covers 6.72 km x 6.72 km)
       - Read the 6 bands at this window
  3. Save patches per city as a single npz:
       data/raw/landsat8/{city}.npz
         contains arrays:
           patches:  (n_tiles, 3, 6, 224, 224)  uint16
           tile_ids: (n_tiles,)                  str
           timestep_dates: (3,)                  str
           scene_ids:      (3, n_tiles)           str
  4. Raw scene COGs are streamed via rio (`with rasterio.open(url)`) and
     never materialised on disk, so the persistent disk cost is just the
     output npzs (~few MB per city).

Run:
    python -m scripts.acquire.landsat8                  # all 44 cities
    python -m scripts.acquire.landsat8 --city lagos     # one city
    python -m scripts.acquire.landsat8 --conus          # CONUS reference sampling
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.acquire.regions import CITIES, City

# --------------------------------------------------------------------------- #
#                              configuration                                    #
# --------------------------------------------------------------------------- #


# Three temporal windows in 2013-2014 to give us seasonal diversity.
# Prithvi expects 3-frame input.
TEMPORAL_WINDOWS = [
    ("2013-06-01", "2013-10-01"),  # boreal summer
    ("2014-01-01", "2014-04-30"),  # winter / dry season
    ("2014-07-01", "2014-10-31"),  # second summer
]

# HLS-equivalent Landsat-8 bands, in the order Prithvi expects.
L8_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
# Planetary Computer Landsat-C2-L2 asset keys for these bands:
#   blue, green, red, nir08, swir16, swir22  (matches the listing above)

PATCH_SIZE = 224          # Prithvi's expected spatial input
L8_RES_M = 30             # Landsat-8 native resolution

MAX_CLOUD_PCT = 30.0      # reject scenes with > this many % cloud cover
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "landsat-c2-l2"


# --------------------------------------------------------------------------- #
#                               STAC search                                     #
# --------------------------------------------------------------------------- #


def _stac_search(client, bbox, start, end, max_cloud=MAX_CLOUD_PCT,
                 limit=50, platforms=("landsat-8",)):
    """Returns items in window, Landsat-8 only, sorted by cloud cover ascending.

    Landsat Collection-2 on Planetary Computer pools Landsat 4/5/7/8/9; the
    `platform` filter restricts to L8 so the per-band asset keys match
    (LE07's SR product has different asset names) and so the imagery is
    consistent with what Prithvi was pretrained on.
    """
    search = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={
            "eo:cloud_cover": {"lt": max_cloud},
            "platform": {"in": list(platforms)},
        },
        limit=limit,
    )
    items = list(search.items())
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    return items


def _gather_scenes(client, bbox, window):
    """All Landsat-8 scenes covering bbox in window, with relaxed cloud
    cap as a fallback. Returns list of items sorted by cloud cover."""
    for cap in (MAX_CLOUD_PCT, 60.0, 90.0):
        items = _stac_search(client, bbox, window[0], window[1], max_cloud=cap)
        if items:
            return items
    return []


def _best_scene_for_point(items, lon, lat):
    """Pick the lowest-cloud scene from `items` whose footprint contains
    (lon, lat). Returns None if no scene covers the point."""
    from shapely.geometry import Point, shape
    pt = Point(lon, lat)
    for it in items:                                  # already sorted by cloud cover
        if shape(it.geometry).contains(pt):
            return it
    return None


# --------------------------------------------------------------------------- #
#                           per-tile patch extraction                            #
# --------------------------------------------------------------------------- #


def _utm_to_l8_crs(x_utm, y_utm, utm_crs, l8_crs):
    """Project a single (x, y) from a city's UTM CRS into the L8 scene's CRS."""
    xs, ys = warp_transform(utm_crs, l8_crs, [x_utm], [y_utm])
    return xs[0], ys[0]


def _extract_patch_from_band(href, signed_href, target_x, target_y,
                              patch_size=PATCH_SIZE):
    """Open a single L8 band COG via streaming HTTP, locate the pixel
    corresponding to (target_x, target_y) in the COG's CRS, and read a
    patch_size x patch_size window centered on it.

    Returns: (uint16 array of shape (patch_size, patch_size), the
              COG's CRS string).
    """
    with rasterio.open(signed_href) as src:
        # Convert world coords to pixel space
        px_x, px_y = ~src.transform * (target_x, target_y)
        # Center the window
        col_off = int(px_x - patch_size / 2)
        row_off = int(px_y - patch_size / 2)
        # Clamp to scene bounds
        col_off = max(0, min(src.width - patch_size, col_off))
        row_off = max(0, min(src.height - patch_size, row_off))
        window = Window(col_off, row_off, patch_size, patch_size)
        arr = src.read(1, window=window)
        # Ensure uint16 output
        if arr.dtype != np.uint16:
            arr = arr.astype(np.uint16)
        return arr, src.crs.to_string()


def _build_signed_assets(item):
    """Sign the item's asset hrefs for streamed read (Planetary Computer
    requires SAS tokens for blob storage). Returns dict band_key -> href."""
    import planetary_computer
    signed = planetary_computer.sign(item)
    return {b: signed.assets[b].href for b in L8_BANDS}


# --------------------------------------------------------------------------- #
#                           per-city acquisition                                 #
# --------------------------------------------------------------------------- #


def _load_tile_centers(city: City, processed_root: Path):
    """Read the city's eval_metrics.json and return one (tile_id, x_utm,
    y_utm, utm_crs) per tile, using the bbox_utm center as the tile
    centroid."""
    ev_path = processed_root / city.region / city.name / "eval_metrics.json"
    if not ev_path.exists():
        return []
    out = []
    for rec in json.loads(ev_path.read_text()):
        bbox = rec["bbox_utm"]
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        out.append((rec["tile_id"], cx, cy, rec["utm_crs"]))
    return out


def _utm_centroid_to_lonlat(x_utm, y_utm, utm_crs):
    """Convert one UTM centroid to lon/lat for STAC footprint testing."""
    xs, ys = warp_transform(utm_crs, "EPSG:4326", [x_utm], [y_utm])
    return xs[0], ys[0]


def _read_band_for_tiles(href, tile_indices, tile_centers, t_idx, b_idx,
                          patches, log_prefix=""):
    """Open one COG (one band of one scene) and read patches for each of
    the assigned tile centers in turn. Writes into `patches[ti, t, b]` for
    each ti in tile_indices. Designed to be called from a thread pool."""
    try:
        with rasterio.open(href) as src:
            l8_crs = src.crs.to_string()
            for ti in tile_indices:
                _, x_utm, y_utm, utm_crs = tile_centers[ti]
                tx, ty = _utm_to_l8_crs(x_utm, y_utm, utm_crs, l8_crs)
                px_x, px_y = ~src.transform * (tx, ty)
                col_off = max(0, min(src.width - PATCH_SIZE,
                                      int(px_x - PATCH_SIZE / 2)))
                row_off = max(0, min(src.height - PATCH_SIZE,
                                      int(px_y - PATCH_SIZE / 2)))
                win = Window(col_off, row_off, PATCH_SIZE, PATCH_SIZE)
                arr = src.read(1, window=win)
                if arr.shape != (PATCH_SIZE, PATCH_SIZE):
                    padded = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=arr.dtype)
                    padded[:arr.shape[0], :arr.shape[1]] = arr
                    arr = padded
                if arr.dtype != np.uint16:
                    arr = arr.astype(np.uint16)
                patches[ti, t_idx, b_idx] = arr
        return True
    except Exception as e:
        print(f"      {log_prefix} read FAILED: {type(e).__name__}: {e}", flush=True)
        return False


def acquire_city(client, city: City, processed_root: Path,
                  out_dir: Path) -> bool:
    out_path = out_dir / f"{city.name}.npz"
    if out_path.exists():
        print(f"  {city.name}: already acquired ({out_path.name})", flush=True)
        return True

    tile_centers = _load_tile_centers(city, processed_root)
    if not tile_centers:
        print(f"  {city.name}: no eval_metrics tiles; skip", flush=True)
        return False

    print(f"  {city.name}: {len(tile_centers)} tiles, fetching 3 timesteps ...",
          flush=True)

    # Precompute tile lon/lat so we can pick the right scene per tile.
    tile_lonlat = [
        _utm_centroid_to_lonlat(x_utm, y_utm, utm_crs)
        for (_, x_utm, y_utm, utm_crs) in tile_centers
    ]

    patches = np.zeros((len(tile_centers), 3, 6, PATCH_SIZE, PATCH_SIZE),
                       dtype=np.uint16)
    scene_ids_per_timestep = []        # list of {tile_id: scene_id}
    dates_per_timestep = []            # earliest representative date per timestep

    for t, window in enumerate(TEMPORAL_WINDOWS):
        items = _gather_scenes(client, list(city.bbox), window)
        if not items:
            print(f"    timestep {t} ({window[0]} to {window[1]}): "
                  f"NO L8 scenes found in city bbox", flush=True)
            return False

        # Map each tile to the lowest-cloud L8 scene that covers it.
        per_tile_scene = {}
        for ti, (lon, lat) in enumerate(tile_lonlat):
            sc = _best_scene_for_point(items, lon, lat)
            if sc is None:
                # Fall back to the most globally low-cloud scene; patches
                # may end up empty if the tile isn't in this scene.
                sc = items[0]
            per_tile_scene[ti] = sc

        unique_scenes = {sc.id: sc for sc in per_tile_scene.values()}
        scene_id_to_tiles: dict[str, list[int]] = {}
        for ti, sc in per_tile_scene.items():
            scene_id_to_tiles.setdefault(sc.id, []).append(ti)

        # Representative date = earliest scene picked at this timestep
        all_dates = [sc.properties.get("datetime", "")[:10]
                     for sc in unique_scenes.values()]
        dates_per_timestep.append(min(all_dates) if all_dates else "")

        tile_to_scene_id = {tile_centers[ti][0]: sc.id
                            for ti, sc in per_tile_scene.items()}
        scene_ids_per_timestep.append(tile_to_scene_id)

        print(f"    timestep {t}: {len(unique_scenes)} unique L8 scene(s) "
              f"covering {len(per_tile_scene)} tiles", flush=True)
        for sid, sc in unique_scenes.items():
            cc = sc.properties.get("eo:cloud_cover", -1)
            n_tiles = len(scene_id_to_tiles[sid])
            print(f"      {sid}  cloud={cc:.1f}%  -> {n_tiles} tile(s)",
                  flush=True)

        # Build the work list: one (scene, band) work-item per band per
        # unique scene, each carrying the list of tiles it should read.
        # Process in a thread pool — disjoint write slots in `patches`
        # mean no locking is needed for the array writes.
        work_items = []          # list of (sc, band_idx, band_key, tile_indices)
        for sid, sc in unique_scenes.items():
            signed = _build_signed_assets(sc)
            tile_indices = scene_id_to_tiles[sid]
            for b_idx, band_key in enumerate(L8_BANDS):
                work_items.append((sc, b_idx, band_key,
                                    signed[band_key], tile_indices, sid))

        n_workers = min(16, len(work_items))
        n_done = 0
        n_failed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = []
            for sc, b_idx, band_key, href, tile_indices, sid in work_items:
                futures.append(ex.submit(
                    _read_band_for_tiles,
                    href, tile_indices, tile_centers, t, b_idx,
                    patches, f"{sid}:{band_key}",
                ))
            for f in concurrent.futures.as_completed(futures):
                ok = f.result()
                if ok:
                    n_done += 1
                else:
                    n_failed += 1
                if (n_done + n_failed) % 50 == 0:
                    print(f"      t{t} band-reads: {n_done} ok / "
                          f"{n_failed} failed / {len(work_items)} total",
                          flush=True)
        if n_failed:
            print(f"    timestep {t}: {n_failed}/{len(work_items)} band-reads failed",
                  flush=True)

    tile_ids = np.array([tc[0] for tc in tile_centers], dtype=object)
    np.savez_compressed(
        out_path,
        patches=patches,
        tile_ids=tile_ids,
        timestep_dates=np.array(dates_per_timestep, dtype=object),
        # scene_ids is a list of dicts; numpy stores it as object array
        scene_ids=np.array(scene_ids_per_timestep, dtype=object),
    )

    # Quick coverage check: fraction of non-zero pixels overall.
    nz_frac = float((patches != 0).mean())
    print(f"    {city.name}: saved -> {out_path.name} "
          f"({out_path.stat().st_size/1e6:.1f} MB, non-zero {nz_frac:.1%})",
          flush=True)
    return True


# --------------------------------------------------------------------------- #
#                                    main                                       #
# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw/landsat8"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--city", type=str, default=None,
                   help="Only fetch this city (test mode).")
    p.add_argument("--conus", action="store_true",
                   help="Fetch CONUS reference tiles instead of cities.")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / args.out
    processed_root = repo_root / args.processed
    out_dir.mkdir(parents=True, exist_ok=True)

    import planetary_computer
    from pystac_client import Client
    client = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    if args.conus:
        # CONUS reference: derive bbox from tile centroids in eval_metrics.json
        # by reprojecting to lon/lat. We pass per-tile per-timestep STAC
        # queries, so the city bbox is only used as the initial STAC search
        # filter; the per-tile `_best_scene_for_point` call picks the right
        # scene per tile from candidates returned for the broader region.
        from pyproj import Transformer
        ev = json.loads((processed_root / "conus" / "conus" / "eval_metrics.json").read_text())
        t_to_ll = Transformer.from_crs(ev[0]["utm_crs"], "EPSG:4326", always_xy=True)
        lons, lats = [], []
        for rec in ev:
            bb = rec["bbox_utm"]
            cx = 0.5 * (bb[0] + bb[2])
            cy = 0.5 * (bb[1] + bb[3])
            lon, lat = t_to_ll.transform(cx, cy)
            lons.append(lon); lats.append(lat)
        synthetic_conus = City(
            name="conus", region="conus",
            bbox=(min(lons) - 0.5, min(lats) - 0.5,
                  max(lons) + 0.5, max(lats) + 0.5),
        )
        print(f"CONUS synthetic bbox: {synthetic_conus.bbox}", flush=True)
        acquire_city(client, synthetic_conus, processed_root, out_dir)
        return

    cities = CITIES
    if args.city:
        cities = [c for c in cities if c.name == args.city]
    cities = [c for c in cities if c.region != "conus"]

    for c in cities:
        print(f"\n=== {c.region}/{c.name} bbox={c.bbox} ===", flush=True)
        ok = acquire_city(client, c, processed_root, out_dir)
        if not ok:
            print(f"  {c.name}: FAILED", flush=True)

    print("\nlandsat-8 acquisition: done.")


if __name__ == "__main__":
    main()
