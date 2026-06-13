"""Study-area map. Real city centroids (recovered from each GHSL tile's CRS and
bounds) for the benchmark: eight target regions of eleven metropolitan areas each
(88 cities, coloured), the CONUS training source (star), and the twelve further
regions that extend the change-rate analysis to twenty (grey). Robinson
projection, Natural Earth land. Deliberately reaches beyond the Europe/North-
America concentration that geographic audits of EO benchmarks have flagged."""
import warnings; warnings.filterwarnings("ignore")
import glob
import os

import matplotlib

matplotlib.use("Agg")
import os as _os
import sys as _sys

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import transform as rwt
from shapely.geometry import Point

_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # for figstyle
import figstyle as fs; fs.use_style()

DATA = _os.path.join(_REPO, "data")
NE = _os.environ.get("NATURALEARTH_SHP",
     _os.path.join(_REPO, "data", "naturalearth_lowres", "naturalearth_lowres.shp"))
# Natural Earth 1:110m countries; set $NATURALEARTH_SHP or place the shapefile at the path above.

TARGETS = {  # region folder -> (display name, colour)
    "east_asia":  ("East Asia",            "#0072B2"),
    "south_asia": ("South Asia",           "#56B4E9"),
    "sea":        ("Southeast Asia",       "#009E73"),
    "oceania":    ("Oceania",              "#CC79A7"),
    "andes":      ("Andes",                "#E69F00"),
    "ssa":        ("Sub-Saharan Africa",   "#D55E00"),
    "mena":       ("MENA",                 "#F0E442"),
    "eeca":       ("E. Europe / C. Asia",  "#6A3D9A"),
}
EXPANSION = ["nordic","camcar","south_asia_2","southern_africa","latam",
             "central_europe","brazil_north","china_interior","weur",
             "mediterranean","japan_korea_2","canada"]


def centroid(reg, city):
    fs_ = (glob.glob(f"{DATA}/processed/{reg}/{city}/builtup_2015.tif")
           or glob.glob(f"{DATA}/processed/{reg}/{city}/builtup_*.tif"))
    for f in fs_:
        try:
            with rasterio.open(f) as s:
                if s.crs is None: continue
                cx = (s.bounds.left + s.bounds.right) / 2
                cy = (s.bounds.top + s.bounds.bottom) / 2
                lon, lat = rwt(s.crs, "EPSG:4326", [cx], [cy])
            return lon[0], lat[0]
        except Exception:
            continue
    return None


def cities_of(reg):
    p = f"{DATA}/processed/{reg}"
    return [c for c in os.listdir(p) if os.path.isdir(f"{p}/{c}")] if os.path.isdir(p) else []


# ---- collect points ---------------------------------------------------------
recs = []
for reg, (name, col) in TARGETS.items():
    for c in cities_of(reg):
        xy = centroid(reg, c)
        if xy: recs.append((xy[0], xy[1], "target", reg, col))
for reg in EXPANSION:
    for c in cities_of(reg):
        xy = centroid(reg, c)
        if xy: recs.append((xy[0], xy[1], "expansion", reg, "#9a9a9a"))
# CONUS is the training *source region* (the whole contiguous US), not a single
# city; mark it at the contiguous-US geographic centre (39.83 N, 98.58 W).
recs.append((-98.58, 39.83, "conus", "conus", fs.C["ink"]))

ROB = "+proj=robin"
pts = gpd.GeoDataFrame(
    {"kind": [r[2] for r in recs], "col": [r[4] for r in recs]},
    geometry=[Point(r[0], r[1]) for r in recs], crs="EPSG:4326").to_crs(ROB)
world = gpd.read_file(NE).to_crs(ROB)

# ---- draw -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 5.0))
world.plot(ax=ax, color="#eef0f2", edgecolor="#c8ccd0", linewidth=0.4, zorder=1)

exp = pts[pts.kind == "expansion"]
exp.plot(ax=ax, color="#9a9a9a", markersize=9, alpha=0.5, zorder=2,
         edgecolor="white", linewidth=0.25)
tgt = pts[pts.kind == "target"]
ax.scatter(tgt.geometry.x, tgt.geometry.y, c=list(tgt["col"]), s=42, zorder=4,
           edgecolor="black", linewidth=0.5)
conus = pts[pts.kind == "conus"]
ax.scatter(conus.geometry.x, conus.geometry.y, marker="*", s=340, zorder=5,
           c=fs.C["ink"], edgecolor="white", linewidth=0.8)

ax.set_axis_off()
ax.set_title("88 cities, eight target regions, six continents",
             fontsize=11, color=fs.C["ink"], pad=4)

# legend
from matplotlib.lines import Line2D

handles = [Line2D([0],[0], marker="o", ls="", mfc=c, mec="black", mew=0.5, ms=7.5,
                  label=n) for n, c in (v for v in TARGETS.values())]
handles += [
    Line2D([0],[0], marker="*", ls="", mfc=fs.C["ink"], mec="white", mew=0.6,
           ms=14, label="CONUS (training source)"),
    Line2D([0],[0], marker="o", ls="", mfc="#9a9a9a", mec="white", mew=0.3,
           ms=7, alpha=0.7, label="+12 regions (change-rate analysis, $n{=}20$)"),
]
ax.legend(handles=handles, loc="lower left", ncol=2, fontsize=7.4,
          frameon=False, handletextpad=0.4, columnspacing=1.1, labelspacing=0.45,
          bbox_to_anchor=(0.0, 0.0))

fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02)
fig.savefig(_os.path.join(_REPO, "figures", "fig_studyarea.pdf"))
print(f"wrote figures/fig_studyarea.pdf  ({len(tgt)} target cities, "
      f"{len(exp)} expansion, {len(conus)} CONUS)")
