"""Download OSM road networks for each city bounding box.

Uses osmnx (which wraps the Overpass API) to pull the drive-network for each
city bbox. Saves results as GraphML for downstream covariate computation.

Run:
    python -m scripts.acquire.osm

Outputs:
    data/raw/osm/<city>.graphml

We also record OSM-completeness diagnostics per city as JSON. SSA cities
(Lagos, Nairobi, Kinshasa) are expected to show lower completeness; this is
honest to report, not a bug.

Dependencies: osmnx >= 1.8, networkx, requests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scripts.acquire.regions import CITIES, City

try:
    import osmnx as ox
except ImportError:  # pragma: no cover
    ox = None


# Public Overpass API base URLs. osmnx appends `/interpreter` itself, so we
# must NOT include that path here — doing so doubles the suffix and 404s.
OVERPASS_ENDPOINTS = [
    # osm.ch consistently responds when overpass-api.de + lz4 mirror 406
    # (Swiss community mirror, lower load).
    "https://overpass.osm.ch/api",
    "https://overpass-api.de/api",
    "https://lz4.overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
]


def fetch_city(city: City, out_root: Path) -> dict:
    if ox is None:
        raise ImportError("osmnx is required: pip install osmnx")

    dest = out_root / f"{city.name}.graphml"
    if dest.exists():
        print(f"  exists, skipping: {dest.name}")
        return {"city": city.name, "status": "skipped"}

    lon_min, lat_min, lon_max, lat_max = city.bbox
    print(f"  fetching OSM drive network for {city.name}")

    last_err = None
    for attempt, endpoint in enumerate(OVERPASS_ENDPOINTS):
        try:
            ox.settings.overpass_url = endpoint
        except Exception:
            pass  # older osmnx versions name the attribute differently
        try:
            # osmnx 2.x: bbox is (left, bottom, right, top) = (W, S, E, N).
            G = ox.graph_from_bbox(
                bbox=(lon_min, lat_min, lon_max, lat_max),
                network_type="drive", simplify=True,
            )
            break
        except Exception as e:
            last_err = e
            print(f"    attempt {attempt+1} via {endpoint} failed: {e}",
                  file=sys.stderr)
            time.sleep(5 * (attempt + 1))   # back off
    else:
        print(f"  FAILED {city.name}: all endpoints exhausted",
              file=sys.stderr)
        return {"city": city.name, "status": "failed",
                "error": str(last_err)}

    dest.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, dest)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"  saved {dest.name}: nodes={n_nodes} edges={n_edges}")

    return {
        "city": city.name,
        "region": city.region,
        "status": "ok",
        "n_nodes": n_nodes,
        "n_edges": n_edges,
    }


def main(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for i, city in enumerate(CITIES):
        diagnostics.append(fetch_city(city, out_root))
        if i < len(CITIES) - 1:
            time.sleep(3)   # be polite to public Overpass servers

    diag_path = out_root / "_diagnostics.json"
    with diag_path.open("w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"diagnostics written to {diag_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw/osm"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    main(repo_root / args.out)
    print("done.")
