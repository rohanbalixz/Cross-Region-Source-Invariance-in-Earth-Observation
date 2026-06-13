"""OSM acquisition via Geofabrik / BBBike .osm.pbf extracts.

Downloads country-level (Geofabrik) or city-level (BBBike) `.osm.pbf` files
and extracts each test city's driving road graph from its bbox, saving as
`data/raw/osm/{city}.graphml` in an osmnx-compatible format so
`scripts.covariates.transport` (which uses `ox.load_graphml` +
`ox.project_graph`) does not need to change.

Extraction uses GDAL's OSM driver through pyogrio rather than pyrosm: pyrosm
has no Python 3.13 wheels and its Cython sources cannot build against the
3.13 C API. pyogrio/GDAL is already a transitive dependency via geopandas.
The "lines" layer yields highway LineStrings; we snap shared OSM node
coordinates to recover intersections (degree>=3) and compute geodesic edge
lengths. All 11 cities are extracted with this single path for internal
consistency.

Run:
    python -m scripts.acquire.osm_pbf
"""

from __future__ import annotations

import argparse
import socket
import sys
import urllib.request
from pathlib import Path

# Never let a stalled OSM .pbf read block forever (raises after 300s of silence;
# higher than DEM/GHSL because country files are large and stream in bursts).
socket.setdefaulttimeout(300)

import networkx as nx
import osmnx as ox
import pyogrio
from pyproj import Geod

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.acquire.regions import CITIES, City

# Map each city → (source, path/url). Geofabrik gives country-level files,
# BBBike gives smaller city extracts when available.
GEOFABRIK = "https://download.geofabrik.de/{}"
BBBIKE    = "https://download.bbbike.org/osm/bbbike/{}/{}.osm.pbf"

CITY_SOURCE: dict[str, tuple[str, str]] = {
    # Original 11 cities
    "mumbai":            ("geofabrik", "asia/india-latest.osm.pbf"),
    "delhi":             ("geofabrik", "asia/india-latest.osm.pbf"),
    "dhaka":             ("geofabrik", "asia/bangladesh-latest.osm.pbf"),
    "lagos":             ("geofabrik", "africa/nigeria-latest.osm.pbf"),
    "nairobi":           ("geofabrik", "africa/kenya-latest.osm.pbf"),
    "kinshasa":          ("geofabrik", "africa/congo-democratic-republic-latest.osm.pbf"),
    "pearl_river_delta": ("geofabrik", "asia/china-latest.osm.pbf"),
    "shanghai":          ("geofabrik", "asia/china-latest.osm.pbf"),
    "quito":             ("geofabrik", "south-america/ecuador-latest.osm.pbf"),
    "bogota":            ("bbbike",    "Bogota"),
    "lima":              ("bbbike",    "Lima"),

    # Additional cities (3 per region beyond the original 3)
    "kolkata":           ("geofabrik", "asia/india-latest.osm.pbf"),
    "bengaluru":         ("geofabrik", "asia/india-latest.osm.pbf"),
    "karachi":           ("geofabrik", "asia/pakistan-latest.osm.pbf"),
    "accra":             ("geofabrik", "africa/ghana-latest.osm.pbf"),
    "addis_ababa":       ("geofabrik", "africa/ethiopia-latest.osm.pbf"),
    "johannesburg":      ("geofabrik", "africa/south-africa-latest.osm.pbf"),
    "beijing":           ("geofabrik", "asia/china-latest.osm.pbf"),
    "tokyo":             ("geofabrik", "asia/japan-latest.osm.pbf"),
    "seoul":             ("geofabrik", "asia/south-korea-latest.osm.pbf"),
    "cusco":             ("geofabrik", "south-america/peru-latest.osm.pbf"),
    "la_paz":            ("geofabrik", "south-america/bolivia-latest.osm.pbf"),
    "medellin":          ("geofabrik", "south-america/colombia-latest.osm.pbf"),

    # Further cities completing the 44-city set.
    # South Asia (5)
    "chennai":           ("geofabrik", "asia/india-latest.osm.pbf"),
    "hyderabad":         ("geofabrik", "asia/india-latest.osm.pbf"),
    "ahmedabad":         ("geofabrik", "asia/india-latest.osm.pbf"),
    "pune":              ("geofabrik", "asia/india-latest.osm.pbf"),
    "lahore":            ("geofabrik", "asia/pakistan-latest.osm.pbf"),
    # SSA (5)
    "abidjan":           ("geofabrik", "africa/ivory-coast-latest.osm.pbf"),
    "dar_es_salaam":     ("geofabrik", "africa/tanzania-latest.osm.pbf"),
    "kampala":           ("geofabrik", "africa/uganda-latest.osm.pbf"),
    "luanda":            ("geofabrik", "africa/angola-latest.osm.pbf"),
    "dakar":             ("geofabrik", "africa/senegal-and-gambia-latest.osm.pbf"),
    # East Asia (6)
    "osaka":             ("geofabrik", "asia/japan-latest.osm.pbf"),
    "chengdu":           ("geofabrik", "asia/china-latest.osm.pbf"),
    "wuhan":             ("geofabrik", "asia/china-latest.osm.pbf"),
    "chongqing":         ("geofabrik", "asia/china-latest.osm.pbf"),
    "taipei":            ("geofabrik", "asia/taiwan-latest.osm.pbf"),
    "busan":             ("geofabrik", "asia/south-korea-latest.osm.pbf"),
    # Andes (5)
    "arequipa":          ("geofabrik", "south-america/peru-latest.osm.pbf"),
    "cochabamba":        ("geofabrik", "south-america/bolivia-latest.osm.pbf"),
    "cali":              ("geofabrik", "south-america/colombia-latest.osm.pbf"),
    "santiago":          ("geofabrik", "south-america/chile-latest.osm.pbf"),
    "caracas":           ("geofabrik", "south-america/venezuela-latest.osm.pbf"),

    # ===== Region scale-up (4 new regions, 11 cities each) =====
    # MENA
    "cairo":             ("geofabrik", "africa/egypt-latest.osm.pbf"),
    "istanbul":          ("geofabrik", "europe/turkey-latest.osm.pbf"),
    "tehran":            ("geofabrik", "asia/iran-latest.osm.pbf"),
    "casablanca":        ("geofabrik", "africa/morocco-latest.osm.pbf"),
    "riyadh":            ("geofabrik", "asia/gcc-states-latest.osm.pbf"),
    "baghdad":           ("geofabrik", "asia/iraq-latest.osm.pbf"),
    "amman":             ("geofabrik", "asia/jordan-latest.osm.pbf"),
    "tunis":             ("geofabrik", "africa/tunisia-latest.osm.pbf"),
    "algiers":           ("geofabrik", "africa/algeria-latest.osm.pbf"),
    "dubai":             ("geofabrik", "asia/gcc-states-latest.osm.pbf"),
    "beirut":            ("geofabrik", "asia/lebanon-latest.osm.pbf"),
    # Southeast Asia
    "jakarta":           ("geofabrik", "asia/indonesia-latest.osm.pbf"),
    "manila":            ("geofabrik", "asia/philippines-latest.osm.pbf"),
    "bangkok":           ("geofabrik", "asia/thailand-latest.osm.pbf"),
    "ho_chi_minh":       ("geofabrik", "asia/vietnam-latest.osm.pbf"),
    "kuala_lumpur":      ("geofabrik", "asia/malaysia-singapore-brunei-latest.osm.pbf"),
    "surabaya":          ("geofabrik", "asia/indonesia-latest.osm.pbf"),
    "hanoi":             ("geofabrik", "asia/vietnam-latest.osm.pbf"),
    "yangon":            ("geofabrik", "asia/myanmar-latest.osm.pbf"),
    "phnom_penh":        ("geofabrik", "asia/cambodia-latest.osm.pbf"),
    "singapore":         ("geofabrik", "asia/malaysia-singapore-brunei-latest.osm.pbf"),
    "bandung":           ("geofabrik", "asia/indonesia-latest.osm.pbf"),
    # Eastern Europe / Central Asia
    "moscow":            ("geofabrik", "russia/central-fed-district-latest.osm.pbf"),
    "kyiv":              ("geofabrik", "europe/ukraine-latest.osm.pbf"),
    "warsaw":            ("geofabrik", "europe/poland-latest.osm.pbf"),
    "bucharest":         ("geofabrik", "europe/romania-latest.osm.pbf"),
    "tashkent":          ("geofabrik", "asia/uzbekistan-latest.osm.pbf"),
    "almaty":            ("geofabrik", "asia/kazakhstan-latest.osm.pbf"),
    "baku":              ("geofabrik", "asia/azerbaijan-latest.osm.pbf"),
    "minsk":             ("geofabrik", "europe/belarus-latest.osm.pbf"),
    "kharkiv":           ("geofabrik", "europe/ukraine-latest.osm.pbf"),
    "novosibirsk":       ("geofabrik", "russia/siberian-fed-district-latest.osm.pbf"),
    "yekaterinburg":     ("geofabrik", "russia/ural-fed-district-latest.osm.pbf"),
    # Oceania (control)
    "sydney":            ("geofabrik", "australia-oceania/australia-latest.osm.pbf"),
    "melbourne":         ("geofabrik", "australia-oceania/australia-latest.osm.pbf"),
    "brisbane":          ("geofabrik", "australia-oceania/australia-latest.osm.pbf"),
    "perth":             ("geofabrik", "australia-oceania/australia-latest.osm.pbf"),
    "auckland":          ("geofabrik", "australia-oceania/new-zealand-latest.osm.pbf"),
    "adelaide":          ("geofabrik", "australia-oceania/australia-latest.osm.pbf"),
    "gold_coast":        ("geofabrik", "australia-oceania/australia-latest.osm.pbf"),
    "canberra":          ("geofabrik", "australia-oceania/australia-latest.osm.pbf"),
    "wellington":        ("geofabrik", "australia-oceania/new-zealand-latest.osm.pbf"),
    "newcastle_au":      ("geofabrik", "australia-oceania/australia-latest.osm.pbf"),
    "christchurch":      ("geofabrik", "australia-oceania/new-zealand-latest.osm.pbf"),

    # ===== Region scale-up #2 (4 new regions, 11 cities each) =====
    # Western Europe
    "london":      ("geofabrik", "europe/great-britain-latest.osm.pbf"),
    "paris":       ("geofabrik", "europe/france-latest.osm.pbf"),
    "berlin":      ("geofabrik", "europe/germany-latest.osm.pbf"),
    "madrid":      ("geofabrik", "europe/spain-latest.osm.pbf"),
    "rome":        ("geofabrik", "europe/italy-latest.osm.pbf"),
    "amsterdam":   ("geofabrik", "europe/netherlands-latest.osm.pbf"),
    "vienna":      ("geofabrik", "europe/austria-latest.osm.pbf"),
    "barcelona":   ("geofabrik", "europe/spain-latest.osm.pbf"),
    "milan":       ("geofabrik", "europe/italy-latest.osm.pbf"),
    "munich":      ("geofabrik", "europe/germany-latest.osm.pbf"),
    "lisbon":      ("geofabrik", "europe/portugal-latest.osm.pbf"),
    # Lowland Latin America
    "sao_paulo":      ("geofabrik", "south-america/brazil-latest.osm.pbf"),
    "rio_de_janeiro": ("geofabrik", "south-america/brazil-latest.osm.pbf"),
    "buenos_aires":   ("geofabrik", "south-america/argentina-latest.osm.pbf"),
    "brasilia":       ("geofabrik", "south-america/brazil-latest.osm.pbf"),
    "montevideo":     ("geofabrik", "south-america/uruguay-latest.osm.pbf"),
    "asuncion":       ("geofabrik", "south-america/paraguay-latest.osm.pbf"),
    "belo_horizonte": ("geofabrik", "south-america/brazil-latest.osm.pbf"),
    "curitiba":       ("geofabrik", "south-america/brazil-latest.osm.pbf"),
    "porto_alegre":   ("geofabrik", "south-america/brazil-latest.osm.pbf"),
    "salvador_br":    ("geofabrik", "south-america/brazil-latest.osm.pbf"),
    "recife":         ("geofabrik", "south-america/brazil-latest.osm.pbf"),
    # Central America & Caribbean
    "mexico_city":    ("geofabrik", "north-america/mexico-latest.osm.pbf"),
    "guatemala_city": ("geofabrik", "central-america/guatemala-latest.osm.pbf"),
    "havana":         ("geofabrik", "central-america/cuba-latest.osm.pbf"),
    "san_salvador":   ("geofabrik", "central-america/el-salvador-latest.osm.pbf"),
    "panama_city":    ("geofabrik", "central-america/panama-latest.osm.pbf"),
    "tegucigalpa":    ("geofabrik", "central-america/honduras-latest.osm.pbf"),
    "managua":        ("geofabrik", "central-america/nicaragua-latest.osm.pbf"),
    "santo_domingo":  ("geofabrik", "central-america/haiti-and-domrep-latest.osm.pbf"),
    "san_jose_cr":    ("geofabrik", "central-america/costa-rica-latest.osm.pbf"),
    "kingston":       ("geofabrik", "central-america/jamaica-latest.osm.pbf"),
    "port_au_prince": ("geofabrik", "central-america/haiti-and-domrep-latest.osm.pbf"),
    # Canada
    "toronto":     ("geofabrik", "north-america/canada-latest.osm.pbf"),
    "montreal":    ("geofabrik", "north-america/canada-latest.osm.pbf"),
    "vancouver":   ("geofabrik", "north-america/canada-latest.osm.pbf"),
    "calgary":     ("geofabrik", "north-america/canada-latest.osm.pbf"),
    "ottawa":      ("geofabrik", "north-america/canada-latest.osm.pbf"),
    "edmonton":    ("geofabrik", "north-america/canada-latest.osm.pbf"),
    "winnipeg":    ("geofabrik", "north-america/canada-latest.osm.pbf"),
    "quebec_city": ("geofabrik", "north-america/canada-latest.osm.pbf"),
    "hamilton_ca": ("geofabrik", "north-america/canada-latest.osm.pbf"),
    "halifax":     ("geofabrik", "north-america/canada-latest.osm.pbf"),
    "victoria":    ("geofabrik", "north-america/canada-latest.osm.pbf"),
}


def url_for(source: str, path: str) -> str:
    if source == "geofabrik":
        return GEOFABRIK.format(path)
    if source == "bbbike":
        return BBBIKE.format(path, path)
    raise ValueError(f"unknown source: {source}")


def _is_valid_pbf(path: Path) -> bool:
    """A real Geofabrik .pbf is large and is not an HTML error/index page.
    A wrong path can return an HTTP-200 HTML page that urlretrieve saves as
    .pbf; reading it later crashes pyogrio. Reject such files here."""
    try:
        if path.stat().st_size < 100_000:   # no country extract is this small
            return False
        with open(path, "rb") as fh:
            head = fh.read(512).lstrip().lower()
        return not (head.startswith(b"<") or b"<html" in head or
                    b"<!doctype" in head or b"<?xml" in head)
    except OSError:
        return False


def download_pbf(url: str, dest: Path) -> Path | None:
    if dest.exists() and dest.stat().st_size > 0:
        if _is_valid_pbf(dest):
            print(f"  cached: {dest.name}  ({dest.stat().st_size/1024/1024:.0f} MB)",
                  flush=True)
            return dest
        print(f"  cached file invalid, re-fetching: {dest.name}", flush=True)
        dest.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} ...", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  FAILED {url}: {e}", file=sys.stderr, flush=True)
        return None
    if not _is_valid_pbf(dest):
        print(f"  INVALID (not a .pbf, likely 404/HTML): {url}",
              file=sys.stderr, flush=True)
        dest.unlink(missing_ok=True)
        return None
    print(f"  saved {dest.name}  ({dest.stat().st_size/1024/1024:.0f} MB)",
          flush=True)
    return dest


# Driving road classes (osmnx "drive" convention: excludes service/track/
# path/footway). Matched against the GDAL OSM "lines" layer `highway` field.
DRIVE_HIGHWAY = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street",
}

_GEOD = Geod(ellps="WGS84")
_SNAP = 7  # decimals to round lon/lat to for node identity (~1 cm)


def _iter_linestrings(geom):
    """Yield each LineString in a (Multi)LineString geometry."""
    if geom is None:
        return
    if geom.geom_type == "LineString":
        yield geom
    elif geom.geom_type == "MultiLineString":
        yield from geom.geoms


def extract_city_graph(pbf_path: Path, city: City, out_path: Path) -> bool:
    if out_path.exists():
        print(f"    graphml exists: {out_path.name}")
        return True
    minx, miny, maxx, maxy = city.bbox   # lon/lat
    print(f"    reading {pbf_path.name} lines for {city.name} "
          f"bbox={city.bbox} ...", flush=True)
    gdf = pyogrio.read_dataframe(
        str(pbf_path), layer="lines",
        bbox=(minx, miny, maxx, maxy),
        columns=["highway", "osm_id"],
    )
    if len(gdf) == 0:
        print(f"    no lines in bbox for {city.name}", file=sys.stderr)
        return False
    gdf = gdf[gdf["highway"].isin(DRIVE_HIGHWAY)]
    if len(gdf) == 0:
        print(f"    no driving roads for {city.name}", file=sys.stderr)
        return False

    # Build an osmnx-compatible MultiDiGraph. Nodes are unique snapped
    # coordinates (integer ids); edges are consecutive-vertex segments.
    # Shared OSM node coordinates collapse to one node, so degree>=3 marks
    # a real intersection for the transport covariate.
    G = nx.MultiDiGraph()
    G.graph["crs"] = "epsg:4326"
    node_index: dict[tuple[float, float], int] = {}

    def node_for(x: float, y: float) -> int:
        key = (round(x, _SNAP), round(y, _SNAP))
        nid = node_index.get(key)
        if nid is None:
            nid = len(node_index)
            node_index[key] = nid
            G.add_node(nid, x=float(x), y=float(y))
        return nid

    for osmid, hw, geom in zip(gdf["osm_id"], gdf["highway"], gdf.geometry):
        try:
            oid = int(osmid)
        except (TypeError, ValueError):
            oid = 0
        for line in _iter_linestrings(geom):
            coords = list(line.coords)
            for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
                a = node_for(x0, y0)
                b = node_for(x1, y1)
                if a == b:
                    continue
                length = float(_GEOD.line_length([x0, x1], [y0, y1]))
                G.add_edge(a, b, osmid=oid, highway=hw, length=length)

    if G.number_of_edges() == 0:
        print(f"    empty graph for {city.name}", file=sys.stderr)
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, out_path)
    print(f"    saved {out_path.name}: nodes={G.number_of_nodes()} "
          f"edges={G.number_of_edges()}", flush=True)
    return True


def main(out_root: Path, pbf_cache: Path, only_city: str | None,
         only_regions: set[str] | None = None) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    pbf_cache.mkdir(parents=True, exist_ok=True)

    # Group cities by .pbf source so each .pbf is downloaded only once.
    by_source: dict[tuple[str, str], list[City]] = {}
    for city in CITIES:
        if only_city and city.name != only_city:
            continue
        if only_regions and city.region not in only_regions:
            continue
        if city.name not in CITY_SOURCE:
            continue
        key = CITY_SOURCE[city.name]
        by_source.setdefault(key, []).append(city)

    if not by_source:
        print("nothing to do (all cities have graphml or none selected)")
        return

    for (source, path), cities in by_source.items():
        print(f"\n=== {source}:{path} -> {[c.name for c in cities]} ===", flush=True)
        url = url_for(source, path)
        pbf_name = Path(path).name if source == "geofabrik" else f"{path}.osm.pbf"
        pbf_path = download_pbf(url, pbf_cache / pbf_name)
        if pbf_path is None:
            continue
        for city in cities:
            out = out_root / f"{city.name}.graphml"
            if out.exists():
                print(f"  skip {city.name}: graphml already present", flush=True)
                continue
            try:
                extract_city_graph(pbf_path, city, out)
            except Exception as e:   # one bad extraction must not abort the run
                print(f"  ERROR extracting {city.name}: {e}",
                      file=sys.stderr, flush=True)
        # After all cities sharing this pbf are extracted, evict the pbf to
        # bound disk usage (cumulative pbf cache would otherwise hit ~3 GB).
        all_done = all((out_root / f"{c.name}.graphml").exists() for c in cities)
        if all_done and pbf_path.exists():
            print(f"  evicting cached pbf: {pbf_path.name}", flush=True)
            pbf_path.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/raw/osm"))
    p.add_argument("--cache", type=Path, default=Path("data/raw/osm/_pbf"))
    p.add_argument("--city", type=str, default=None,
                   help="Only fetch this city.")
    p.add_argument("--region", action="append", default=[],
                   help="Only fetch cities in these regions (repeatable).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    main(repo_root / args.out, repo_root / args.cache, args.city,
         only_regions=set(args.region) or None)
    print("done.")
