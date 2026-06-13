"""Per-tile transport covariates Z_R from OSM road networks.

For each city, load the OSM GraphML produced by `scripts.acquire.osm`,
project the edges into the city UTM grid, and for every benchmark tile
compute:

  - road_density: total edge length / tile area, in km/km^2
  - intersection_density: number of degree>=3 nodes / tile area, in km^-2
  - grid_entropy: Shannon entropy of edge orientations binned to 18 bins
                  on [0, pi). Low = grid-aligned; high = isotropic/organic.
  - osm_completeness: heuristic 0..1 score; here a placeholder driven by
                  global road_density rank within the city. Replace with a
                  cross-source benchmark if a road-mask source is available.

Run:
    python -m scripts.covariates.transport

Outputs:
    data/processed/{region}/{city}/transport.json
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.acquire.regions import CITIES, City
from scripts.common import (
    TILE_PX, TILE_RES_M,
    enumerate_tiles_from_grid, tile_ref_to_dict, write_tile_records,
)

try:
    import networkx as nx
    import osmnx as ox
except ImportError:  # pragma: no cover
    ox = None
    nx = None


GRID_BINS = 18  # 10 degrees per bin over [0, pi)


def load_edges_utm(graphml_path: Path, utm_crs: str):
    """Return a list of (x0, y0, x1, y1, length_m) tuples in the city UTM CRS,
    along with a list of node coordinates (x, y, degree) for intersection
    counts.
    """
    if ox is None:
        raise ImportError("osmnx is required: pip install osmnx")

    # networkx.read_graphml allocates millions of small objects for these
    # GDAL-derived graphs (per-vertex nodes). Python's cyclic GC then rescans
    # the growing object set on every pass -> quadratic thrash that stalls the
    # load. Disable GC for the parse, free the graph, and collect once.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        G = ox.load_graphml(graphml_path)
        G = ox.project_graph(G, to_crs=utm_crs)

        edges = []
        for u, v, data in G.edges(data=True):
            x0, y0 = G.nodes[u]["x"], G.nodes[u]["y"]
            x1, y1 = G.nodes[v]["x"], G.nodes[v]["y"]
            length = float(data.get("length",
                                    math.hypot(x1 - x0, y1 - y0)))
            edges.append((x0, y0, x1, y1, length))

        nodes = []
        for n, data in G.nodes(data=True):
            deg = G.degree(n)
            nodes.append((data["x"], data["y"], int(deg)))
    finally:
        G = None
        if gc_was_enabled:
            gc.enable()
        gc.collect()

    return edges, nodes


def edge_length_in_bbox(x0, y0, x1, y1, length_m, bbox):
    """Approximate length of an edge falling inside `bbox = (xmin,ymin,xmax,ymax)`.
    We use a midpoint-in-bbox heuristic: cheaper than full clipping, accurate
    enough at 32 km tiles vs. typical OSM segments under 500 m.
    """
    mx = 0.5 * (x0 + x1)
    my = 0.5 * (y0 + y1)
    if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
        return length_m, math.atan2(y1 - y0, x1 - x0)
    return 0.0, None


def orientation_entropy(angles: list[float]) -> float:
    """Shannon entropy (nats) of edge orientations folded onto [0, pi)."""
    if not angles:
        return 0.0
    a = np.asarray(angles, dtype=np.float64)
    a = np.where(a < 0, a + math.pi, a)
    a = np.where(a >= math.pi, a - math.pi, a)
    hist, _ = np.histogram(a, bins=GRID_BINS, range=(0.0, math.pi))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def process_city(city: City, processed_root: Path, osm_root: Path,
                 out_path: Path) -> None:
    bu_path = processed_root / city.region / city.name / "builtup_2015.tif"
    graphml_path = osm_root / f"{city.name}.graphml"
    if not bu_path.exists():
        print(f"  SKIP {city.name}: missing {bu_path}")
        return
    if not graphml_path.exists():
        print(f"  SKIP {city.name}: missing {graphml_path}")
        return

    with rasterio.open(str(bu_path)) as bu:
        builtup = bu.read(1).astype(np.float32)
        transform = bu.transform
        utm_crs = bu.crs.to_string()

    refs = enumerate_tiles_from_grid(
        builtup_2015=builtup, utm_transform=transform,
        city_name=city.name, region=city.region, utm_crs=utm_crs,
    )

    edges, nodes = load_edges_utm(graphml_path, utm_crs)

    tile_area_km2 = (TILE_PX * TILE_RES_M / 1000.0) ** 2

    records = []
    # Precompute global road-density rank for completeness heuristic.
    densities = []
    cached_lengths = []
    cached_orients = []
    cached_intersections = []

    for ref in refs:
        bbox = ref.bbox_utm
        total_len = 0.0
        orient_list: list[float] = []
        for (x0, y0, x1, y1, length) in edges:
            l, ang = edge_length_in_bbox(x0, y0, x1, y1, length, bbox)
            if l > 0:
                total_len += l
                if ang is not None:
                    orient_list.append(ang)
        n_intersections = sum(
            1 for (x, y, deg) in nodes
            if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3] and deg >= 3
        )
        cached_lengths.append(total_len)
        cached_orients.append(orient_list)
        cached_intersections.append(n_intersections)
        densities.append((total_len / 1000.0) / tile_area_km2)  # km/km^2

    density_arr = np.asarray(densities)
    if len(density_arr) > 0 and density_arr.max() > 0:
        completeness = np.clip(density_arr / np.percentile(density_arr, 90), 0, 1)
    else:
        completeness = np.zeros_like(density_arr)

    for k, ref in enumerate(refs):
        rec = tile_ref_to_dict(ref)
        rec.update({
            "road_density": float(density_arr[k]),
            "intersection_density": float(cached_intersections[k] / tile_area_km2),
            "grid_entropy": orientation_entropy(cached_orients[k]),
            "osm_completeness": float(completeness[k]),
        })
        records.append(rec)

    write_tile_records(records, out_path)
    print(f"  {city.name}: {len(records)} tiles -> {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--osm", type=Path, default=Path("data/raw/osm"))
    p.add_argument("--city", action="append", default=[])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    processed_root = repo_root / args.processed
    osm_root = repo_root / args.osm

    cities = CITIES if not args.city else [c for c in CITIES if c.name in args.city]
    for city in cities:
        out = processed_root / city.region / city.name / "transport.json"
        process_city(city, processed_root, osm_root, out)
    print("done.")
