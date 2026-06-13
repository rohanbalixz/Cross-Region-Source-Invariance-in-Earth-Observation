"""Download GHSL built-up tiles for each test-region city bounding box.

GHSL (Global Human Settlement Layer) is distributed by the European Commission
JRC. We use the GHS-BUILT-S epoch series (1975, 1990, 2000, 2015, 2020) and the
GHS-POP grids at the same epochs, all at 100 m resolution in Mollweide
projection (ESRI:54009).

Public download root:
    https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/

Run:
    python -m scripts.acquire.ghsl --epoch 2015 --epoch 2020 --layer BUILT_S

Outputs: data/raw/ghsl/{layer}/{epoch}/<tile>.tif
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable, List

# A single unresponsive JRC tile must never hang the whole run. Cap every
# socket read so a stalled connection raises instead of blocking forever.
socket.setdefaulttimeout(120)

# These tile names cover the four test regions at 100 m / 1 km grids.
# Use the GHS-BUILT-S 2023A release (R2023A), 100 m Mollweide.
# Tile IDs follow the JRC R6_Cxx scheme, derived by intersecting city bboxes
# with the JRC tile manifest.

GHSL_TILES_BY_REGION = {
    # Tile IDs computed from city bboxes via Mollweide projection; each URL
    # returns real (non-ocean) data of 4-28 MB.
    # Expanded for 44 cities (11/region); tile IDs computed from each city's
    # bbox via Mollweide ESRI:54009 grid and verified to contain the original
    # known tiles. Ocean/no-data tiles 404 gracefully during fetch.
    "south_asia": ["R6_C25", "R6_C26", "R7_C25", "R7_C26", "R7_C27", "R8_C26"],
    "ssa":        ["R8_C17", "R8_C22", "R9_C18", "R9_C19", "R9_C22",
                   "R10_C20", "R10_C22", "R11_C20", "R13_C21"],
    "east_asia":  ["R5_C28", "R5_C29", "R5_C30", "R5_C31", "R6_C28",
                   "R6_C29", "R6_C30", "R7_C29", "R7_C30"],
    "andes":      ["R8_C12", "R9_C11", "R10_C11", "R11_C11", "R11_C12",
                   "R12_C11", "R12_C12", "R13_C12", "R14_C12"],
}

# GHSL R2023A 100m Mollweide (ESRI:54009) tiling: 1,000,000 m tiles, origin
# top-left of the canonical World-Mollweide bounding box. The Mollweide tile
# computer below reproduces every explicit tile above as a strict superset
# (<=1 harmless extra adjacent tile per region, which 404s gracefully if
# ocean/no-data). The remaining regions are tiled automatically.
_X_MIN = -18041000.0
_Y_MAX = 9000000.0
_TILE = 1000000.0


def compute_region_tiles(region: str) -> List[str]:
    """Compute the GHSL tile IDs covering every city bbox in `region`."""
    from pyproj import Transformer  # local import; only needed for new regions

    from scripts.acquire.regions import REGIONS

    tf = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    tiles: set[str] = set()
    for city in REGIONS.get(region, []):
        lon0, lat0, lon1, lat1 = city.bbox
        xs, ys = [], []
        n = 11
        for i in range(n):
            for j in range(n):
                lon = lon0 + (lon1 - lon0) * i / (n - 1)
                lat = lat0 + (lat1 - lat0) * j / (n - 1)
                x, y = tf.transform(lon, lat)
                xs.append(x)
                ys.append(y)
        cmin = int((min(xs) - _X_MIN) // _TILE) + 1
        cmax = int((max(xs) - _X_MIN) // _TILE) + 1
        rmin = int((_Y_MAX - max(ys)) // _TILE) + 1
        rmax = int((_Y_MAX - min(ys)) // _TILE) + 1
        for r in range(rmin, rmax + 1):
            for c in range(cmin, cmax + 1):
                tiles.add(f"R{r}_C{c}")
    return sorted(tiles)


def tiles_for_region(region: str) -> List[str]:
    """Explicit tiles for the original regions; auto-computed for the rest."""
    if region in GHSL_TILES_BY_REGION:
        return GHSL_TILES_BY_REGION[region]
    return compute_region_tiles(region)


GHSL_BASE = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_{layer}_GLOBE_R2023A/GHS_{layer}_E{epoch}_GLOBE_R2023A_54009_100/"
    "V1-0/tiles/GHS_{layer}_E{epoch}_GLOBE_R2023A_54009_100_V1_0_{tile}.zip"
)


def url_for(layer: str, epoch: int, tile: str) -> str:
    return GHSL_BASE.format(layer=layer, epoch=epoch, tile=tile)


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    if dest.exists():
        print(f"  exists, skipping: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"  fetching {url} (attempt {attempt})", flush=True)
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)          # atomic: only a complete file lands
            return
        except Exception as e:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                print(f"  FAILED {url}: {e}", file=sys.stderr, flush=True)
            else:
                time.sleep(2 * attempt)


def main(epochs: Iterable[int], layers: Iterable[str], out_root: Path,
         regions: Iterable[str] | None = None, workers: int = 8) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from scripts.acquire.regions import REGIONS
    region_list = list(regions) if regions else list(REGIONS.keys())

    # Build the full job list, then fetch concurrently. The JRC server throttles
    # per-connection, so parallel reads of these small (~5 MB) tiles are far
    # faster than sequential; fetch() is independent per file and atomic.
    jobs: list[tuple[str, Path]] = []
    for layer in layers:
        for epoch in epochs:
            for region in region_list:
                tiles = tiles_for_region(region)
                print(f"[{layer} {epoch}] {region}: {len(tiles)} tiles")
                for tile in tiles:
                    url = url_for(layer, epoch, tile)
                    dest = out_root / layer / str(epoch) / region / f"{tile}.zip"
                    jobs.append((url, dest))

    todo = [(u, d) for (u, d) in jobs if not d.exists()]
    print(f"GHSL: {len(jobs)} tiles total, {len(todo)} to fetch, "
          f"{workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda j: fetch(j[0], j[1]), todo))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epoch", type=int, action="append", default=[],
                   help="Epoch to download (repeatable). Default: 2015 2020.")
    p.add_argument("--layer", action="append", default=[],
                   help="GHSL layer (BUILT_S, POP, ...). Default: BUILT_S.")
    p.add_argument("--out", type=Path,
                   default=Path("data/raw/ghsl"),
                   help="Output root relative to NeurIPS26/.")
    p.add_argument("--region", action="append", default=[],
                   help="Region key to fetch (repeatable). Default: all.")
    args = p.parse_args()
    if not args.epoch:
        args.epoch = [2015, 2020]
    if not args.layer:
        args.layer = ["BUILT_S"]
    return args


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_root = repo_root / args.out
    main(args.epoch, args.layer, out_root, regions=args.region or None)
    print(f"done. files under {out_root}")
