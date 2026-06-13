"""CONUS reference covariates on the exact tiles scored by conus_baseline.

The attribution step joins per-tile JSONs by `tile_id`, so the CONUS reference
must carry terrain/transport/settlement covariates for the *same* tiles that
`conus_baseline` produced FoM for. We therefore read the tile list straight
from `data/processed/conus/conus/eval_metrics.json` (each record has the pixel
offset `i,j` into the CONUS grid and its EPSG:5070 `bbox_utm`) rather than
re-sampling.

Axes (run one at a time):
  settlement  -- from the already-local CONUS_builtup_2015 raster (no download)
  terrain     -- from Copernicus DEM 30m 1-deg tiles (downloaded to --cache)
  transport   -- from a us-latest OSM .pbf (downloaded to --cache)

Heavy DEM/OSM data goes to --cache (default /tmp, NON-iCloud) so the iCloud
Desktop sync never touches it. Only the small output JSON lands in the repo.

Run:
  python -m scripts.eval.conus_covariates settlement
  python -m scripts.eval.conus_covariates terrain   --cache ./cache/conus_dem
  python -m scripts.eval.conus_covariates transport --cache ./cache/conus_osm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod, Transformer
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.acquire.osm_pbf import _SNAP, DRIVE_HIGHWAY, _iter_linestrings
from scripts.acquire.srtm import fetch_one, tiles_covering_bbox
from scripts.common import TILE_PX, TILE_RES_M
from scripts.covariates.settlement import BUILTUP_THRESHOLD, patch_stats
from scripts.covariates.terrain import compute_slope_deg, compute_tri
from scripts.covariates.transport import edge_length_in_bbox, orientation_entropy

REPO = Path(__file__).resolve().parents[2]
EVAL_METRICS = REPO / "data/processed/conus/conus/eval_metrics.json"
# CONUS rasters live in the processed-data directory.
GEOTIFF = (REPO / "data/processed/conus").resolve()
TARGET_EPOCH = 2015


def load_tiles() -> list[dict]:
    recs = json.load(open(EVAL_METRICS))
    recs = recs if isinstance(recs, list) else recs.get("tiles", recs)
    return recs


def base_record(t: dict) -> dict:
    """Common identity fields every covariate JSON must carry for the join."""
    return {
        "tile_id": t["tile_id"], "city": "conus", "region": "conus",
        "i": t["i"], "j": t["j"], "bbox_utm": t["bbox_utm"],
        "utm_crs": t["utm_crs"],
    }


def run_settlement(out_path: Path) -> None:
    tiles = load_tiles()
    bu_path = GEOTIFF / f"CONUS_builtup_{TARGET_EPOCH}.tif"
    print(f"reading {bu_path.name} ...", flush=True)
    with rasterio.open(str(bu_path)) as src:
        builtup = src.read(1).astype(np.float32)
    print(f"  grid {builtup.shape}; computing settlement for {len(tiles)} tiles",
          flush=True)

    records = []
    for n, t in enumerate(tiles, 1):
        i, j = t["i"], t["j"]
        blk = builtup[i:i + TILE_PX, j:j + TILE_PX]
        binary = (blk >= BUILTUP_THRESHOLD).astype(np.uint8)
        frag, comp, cont = patch_stats(binary)
        rec = base_record(t)
        rec.update({
            "builtup_frac": float(blk.mean()),
            "fragmentation": frag,
            "compactness": comp,
            "contiguity": cont,
        })
        records.append(rec)
        if n % 100 == 0:
            print(f"  {n}/{len(tiles)}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {out_path}  ({len(records)} tiles)", flush=True)


_TO_WGS84 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)


def tile_wgs84_bbox(bbox_utm) -> tuple[float, float, float, float]:
    """EPSG:5070 [x_min,y_min,x_max,y_max] -> WGS84 lon/lat bbox covering it."""
    x_min, y_min, x_max, y_max = bbox_utm
    lons, lats = [], []
    for x in (x_min, x_max):
        for y in (y_min, y_max):
            lo, la = _TO_WGS84.transform(x, y)
            lons.append(lo); lats.append(la)
    return (min(lons), min(lats), max(lons), max(lats))


def dem_elev_on_tile(cell_paths: list[Path], bbox_utm) -> np.ndarray | None:
    """Reproject covering Copernicus DEM cell(s) onto the tile's 128x128
    EPSG:5070 grid at 250 m. Returns the elevation array or None."""
    srcs = []
    for p in cell_paths:
        try:
            srcs.append(rasterio.open(str(p)))
        except Exception:
            pass
    if not srcs:
        return None
    try:
        mosaic, src_transform = rio_merge(srcs)
        src = mosaic[0].astype(np.float32)
        src_crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()

    x_min, y_min, x_max, y_max = bbox_utm
    dst_transform = from_origin(x_min, y_max, TILE_RES_M, TILE_RES_M)
    dst = np.full((TILE_PX, TILE_PX), np.nan, dtype=np.float32)
    reproject(
        source=src, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs="EPSG:5070",
        resampling=Resampling.bilinear,
        src_nodata=src.min() if not np.isfinite(src).all() else None,
    )
    return dst


def run_terrain(out_path: Path, cache: Path, limit: int | None) -> None:
    tiles = load_tiles()
    if limit:
        tiles = tiles[:limit]
    # collect unique 1-degree DEM cells across all tiles
    per_tile_cells = []
    unique = {}
    for t in tiles:
        wb = tile_wgs84_bbox(t["bbox_utm"])
        cells = tiles_covering_bbox(wb)
        per_tile_cells.append(cells)
        for c in cells:
            unique[c] = None
    print(f"{len(tiles)} tiles -> {len(unique)} unique Copernicus DEM cells; "
          f"downloading to {cache}", flush=True)

    cache.mkdir(parents=True, exist_ok=True)
    done = 0
    for (lat, lon) in unique:
        p = fetch_one(lat, lon, cache)
        unique[(lat, lon)] = p
        done += 1
        if done % 25 == 0:
            print(f"  DEM {done}/{len(unique)} cells", flush=True)
    n_ok = sum(1 for v in unique.values() if v is not None)
    print(f"  downloaded {n_ok}/{len(unique)} cells (missing = ocean/no-data)",
          flush=True)

    records = []
    for n, (t, cells) in enumerate(zip(tiles, per_tile_cells), 1):
        cell_paths = [unique[c] for c in cells if unique.get(c) is not None]
        rec = base_record(t)
        elev = dem_elev_on_tile(cell_paths, t["bbox_utm"]) if cell_paths else None
        if elev is None or not np.isfinite(elev).any():
            rec.update({"slope_mean": None, "slope_var": None,
                        "tri": None, "elev_mean": None})
        else:
            slope = compute_slope_deg(elev, cell_m=float(TILE_RES_M))
            tri = compute_tri(elev)
            rec.update({
                "slope_mean": float(np.nanmean(slope)),
                "slope_var": float(np.nanvar(slope)),
                "tri": float(np.nanmean(tri)),
                "elev_mean": float(np.nanmean(elev)),
            })
        records.append(rec)
        if n % 100 == 0:
            print(f"  terrain {n}/{len(tiles)}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    n_valid = sum(1 for r in records if r["slope_mean"] is not None)
    print(f"wrote {out_path}  ({n_valid}/{len(records)} tiles with terrain)",
          flush=True)


_GEOD = Geod(ellps="WGS84")
_TO_5070 = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)


def build_us_roads_gpkg(pbf: Path, gpkg: Path) -> None:
    """One-time ogr2ogr: extract driving-road LineStrings from the US pbf into a
    spatially-indexed WGS84 GeoPackage so per-tile bbox reads are fast."""
    if gpkg.exists():
        print(f"  roads gpkg cached: {gpkg.name} "
              f"({gpkg.stat().st_size/1e9:.1f} GB)", flush=True)
        return
    classes = ",".join(f"'{c}'" for c in sorted(DRIVE_HIGHWAY))
    cmd = ["ogr2ogr", "-f", "GPKG", str(gpkg), str(pbf), "lines",
           "-where", f"highway IN ({classes})",
           "-nln", "roads", "-nlt", "PROMOTE_TO_MULTI",
           "-skipfailures", "-progress"]
    print(f"  ogr2ogr -> {gpkg.name} (driving roads only) ...", flush=True)
    subprocess.run(cmd, check=True)
    print(f"  built {gpkg.name} ({gpkg.stat().st_size/1e9:.1f} GB)", flush=True)


def tile_edges_nodes_5070(gpkg: Path, wgs84_bbox):
    """Read driving roads in the tile's WGS84 bbox, build edges (x0,y0,x1,y1,
    length_m) and nodes (x,y,degree) in EPSG:5070 -- same construction as
    osm_pbf.extract_city_graph, in-memory for one tile."""
    import pyogrio
    minx, miny, maxx, maxy = wgs84_bbox
    gdf = pyogrio.read_dataframe(str(gpkg), layer="roads",
                                 bbox=(minx, miny, maxx, maxy),
                                 columns=["highway"])
    if len(gdf) == 0:
        return [], []
    node_xy: dict[tuple, int] = {}
    deg: dict[int, int] = {}
    edges = []

    def node_for(x, y):
        key = (round(x, _SNAP), round(y, _SNAP))
        nid = node_xy.get(key)
        if nid is None:
            nid = len(node_xy)
            node_xy[key] = nid
            deg[nid] = 0
        return nid

    for geom in gdf.geometry:
        for line in _iter_linestrings(geom):
            coords = list(line.coords)
            for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
                a = node_for(x0, y0)
                b = node_for(x1, y1)
                if a == b:
                    continue
                length = float(_GEOD.line_length([x0, x1], [y0, y1]))
                X0, Y0 = _TO_5070.transform(x0, y0)
                X1, Y1 = _TO_5070.transform(x1, y1)
                edges.append((X0, Y0, X1, Y1, length))
                deg[a] += 1
                deg[b] += 1
    # node coords (5070) + degree
    nodes = []
    for (lon, lat), nid in node_xy.items():
        X, Y = _TO_5070.transform(lon, lat)
        nodes.append((X, Y, deg[nid]))
    return edges, nodes


def run_transport(out_path: Path, cache: Path) -> None:
    tiles = load_tiles()
    pbf = cache / "us-latest.osm.pbf"
    gpkg = cache / "us_roads.gpkg"
    if not gpkg.exists():
        if not pbf.exists():
            raise SystemExit(f"missing {pbf}; download us-latest.osm.pbf first")
        build_us_roads_gpkg(pbf, gpkg)

    tile_area_km2 = (TILE_PX * TILE_RES_M / 1000.0) ** 2
    densities, intersections, orients = [], [], []
    print(f"computing transport for {len(tiles)} CONUS tiles ...", flush=True)
    for n, t in enumerate(tiles, 1):
        bbox5070 = t["bbox_utm"]
        edges, nodes = tile_edges_nodes_5070(gpkg, tile_wgs84_bbox(bbox5070))
        total_len = 0.0
        orient_list = []
        for (x0, y0, x1, y1, length) in edges:
            l, ang = edge_length_in_bbox(x0, y0, x1, y1, length, bbox5070)
            if l > 0:
                total_len += l
                if ang is not None:
                    orient_list.append(ang)
        n_int = sum(1 for (x, y, d) in nodes
                    if bbox5070[0] <= x <= bbox5070[2]
                    and bbox5070[1] <= y <= bbox5070[3] and d >= 3)
        densities.append((total_len / 1000.0) / tile_area_km2)
        intersections.append(n_int)
        orients.append(orient_list)
        if n % 50 == 0:
            print(f"  transport {n}/{len(tiles)}", flush=True)

    density_arr = np.asarray(densities)
    if density_arr.max() > 0:
        completeness = np.clip(density_arr / np.percentile(density_arr, 90), 0, 1)
    else:
        completeness = np.zeros_like(density_arr)

    records = []
    for k, t in enumerate(tiles):
        rec = base_record(t)
        rec.update({
            "road_density": float(density_arr[k]),
            "intersection_density": float(intersections[k] / tile_area_km2),
            "grid_entropy": orientation_entropy(orients[k]),
            "osm_completeness": float(completeness[k]),
        })
        records.append(rec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {out_path}  ({len(records)} tiles)", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("axis", choices=["settlement", "terrain", "transport"])
    p.add_argument("--cache", type=Path, default=Path("./cache/conus_cov"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N tiles (for validation).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = args.out or (REPO / f"data/processed/conus/conus/{args.axis}.json")
    if args.axis == "settlement":
        run_settlement(out)
    elif args.axis == "terrain":
        run_terrain(out, args.cache, args.limit)
    elif args.axis == "transport":
        run_transport(out, args.cache)
