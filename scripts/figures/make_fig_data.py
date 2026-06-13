"""Figure: what the model actually sees. Real example tiles from four cities in
four biomes, across five input representations. The two globally-harmonised
products (GHSL built-up, WorldCover land cover) share the same visual character
down every row; the three raw-sensor inputs (Sentinel-2, Landsat, Sentinel-1)
look completely different city to city. That contrast is this study's thesis,
made visible: harmonised meaning travels, raw appearance does not.

All tiles are real data from the released benchmark (no mock-ups). Sentinel-2 and
its WorldCover label are the co-registered hard-task pair; Landsat and Sentinel-1
share the historical tile grid; GHSL is the processed built-up raster. Tiles are
representative of each city, not pixel-co-registered across sources."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # for figstyle
import figstyle as fs; fs.use_style()

# city, region, pretty label
CITIES = [
    ("cairo",   "mena",      "Cairo · desert"),
    ("jakarta", "sea",       "Jakarta · tropical"),
    ("beijing", "east_asia", "Beijing · temperate"),
    ("sydney",  "oceania",   "Sydney · coastal"),
]
import os
DATA = _os.path.join(_REPO, "data")

WC = {10:'#006400',20:'#ffbb22',30:'#ffff4c',40:'#f096ff',50:'#fa0000',
      60:'#b4b4b4',70:'#f0f0f0',80:'#0064c8',90:'#0096a0',95:'#00cf75',100:'#fae6a0'}
_codes = sorted(WC)
WC_CMAP = ListedColormap([WC[c] for c in _codes])
WC_NORM = BoundaryNorm([c-0.5 for c in _codes]+[_codes[-1]+0.5], len(_codes))


def truecolor(bands_rgb, scale=0.22, gamma=0.66):
    """bands_rgb: list of 3 reflectance planes; fixed scaling preserves real colour."""
    return np.clip(np.stack(bands_rgb, -1) / scale, 0, 1) ** gamma


def s2_rgb(city, reg):
    d = np.load(f"{DATA}/hardtask/{reg}/{city}/patches.npz")
    s2, lab = d["s2"].astype(np.float32), d["label"]
    bu = (lab == 50).mean((1, 2))
    bright = s2[:, :3].mean((1, 2, 3))               # cloud proxy
    valid = (bu > 0.06) & (bu < 0.6) & (bright < 2400)
    if not valid.any(): valid = (bu > 0.03)
    idx = np.where(valid)[0]
    pick = idx[np.argmin(np.abs(bu[idx] - 0.33))]
    refl = s2[pick, :3] / 10000.0                    # B,G,R -> reflectance
    return truecolor([refl[2], refl[1], refl[0]]), lab[pick]


def landsat_sar(city):
    l8 = np.load(f"{DATA}/raw/landsat8/{city}.npz", allow_pickle=True)
    lp = l8["patches"].astype(np.float32)            # (20,3,6,224,224) clean L8
    refl = np.clip(lp * 2.75e-5 - 0.2, 0, 1)         # C2 L2 SR scaling
    cloud = (refl[:, :, :3].min(2) > 0.30).mean((2, 3))  # white in all visible = cloud
    ts = int(np.argmin(cloud.mean(0)))               # least-cloudy timestep overall
    red = refl[:, ts, 2]
    tex = red.std((1, 2)); bright = red.mean((1, 2))
    valid = (bright > 0.04) & (cloud[:, ts] < 0.05)
    cand = np.where(valid)[0] if valid.any() else np.arange(len(red))
    li = cand[np.argmax(tex[cand])]
    e = refl[li, ts]
    lrgb = truecolor([e[2], e[1], e[0]])
    sar = np.load(f"{DATA}/raw/sentinel1/{city}.npz")["patches"]  # (20,2,64,64)
    vv = sar[min(li, sar.shape[0]-1), 0]
    lo, hi = np.percentile(vv, [2, 98]); vvn = np.clip((vv-lo)/(hi-lo+1e-9), 0, 1)
    return lrgb, vvn


def ghsl_core(city, reg, n=160):
    with rasterio.open(f"{DATA}/processed/{reg}/{city}/builtup_2015.tif") as s:
        g = s.read(1)
    H, W = g.shape
    # centre on the built-up centroid so we crop the urban core
    ys, xs = np.where(g > 0.2)
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (H//2, W//2)
    y0 = np.clip(cy-n//2, 0, max(0, H-n)); x0 = np.clip(cx-n//2, 0, max(0, W-n))
    return g[y0:y0+n, x0:x0+n]


# ---- layout -------------------------------------------------------------
GREEN, RED = fs.HARMONISED, fs.RAWSENSOR
COLS = [("GHSL\nbuilt-up", GREEN), ("WorldCover\nland cover", GREEN),
        ("Sentinel-2\noptical", RED), ("Landsat\noptical", RED),
        ("Sentinel-1\nSAR", RED)]
nC, nR = len(COLS), len(CITIES)

fig, axes = plt.subplots(nR, nC, figsize=(9.6, 8.0))
fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.012,
                    wspace=0.05, hspace=0.06)

for r, (city, reg, label) in enumerate(CITIES):
    g = ghsl_core(city, reg)
    s2, wclab = s2_rgb(city, reg)
    lrgb, vv = landsat_sar(city)
    panels = [
        (g,     dict(cmap="inferno", vmin=0, vmax=max(0.4, float(g.max())))),
        (wclab, dict(cmap=WC_CMAP, norm=WC_NORM)),
        (s2, {}), (lrgb, {}), (vv, dict(cmap="gray")),
    ]
    for c, (img, kw) in enumerate(panels):
        ax = axes[r, c]
        ax.imshow(img, interpolation="nearest", aspect="equal", **kw)
        ax.set_xticks([]); ax.set_yticks([])
        col = COLS[c][1]
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.7); sp.set_edgecolor(col)
    axes[r, 0].set_ylabel(label, fontsize=9.5, labelpad=6, color=fs.C["ink"])

for c, (name, col) in enumerate(COLS):           # column headers
    axes[0, c].set_title(name, fontsize=9, color=col, pad=4, linespacing=1.0)

# group banners, placed from the axes' real positions so they never drift
fig.canvas.draw()
def span_of(c0, c1):
    p0 = axes[0, c0].get_position(); p1 = axes[0, c1].get_position()
    return p0.x0, p1.x1
def banner(c0, c1, text, color):
    x0, x1 = span_of(c0, c1); y = 0.945
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=color, lw=3.6,
                   solid_capstyle="butt", transform=fig.transFigure))
    fig.text((x0+x1)/2, y+0.010, text, ha="center", va="bottom",
             fontsize=11.5, fontweight="bold", color=color)
banner(0, 1, "HARMONISED  ·  transfers", GREEN)
banner(2, 4, "RAW SENSOR  ·  region-specific appearance", RED)

# faint divider between the harmonised group and the raw-sensor group
xdiv = (axes[0, 1].get_position().x1 + axes[0, 2].get_position().x0) / 2
ytop = axes[0, 0].get_position().y1; ybot = axes[nR-1, 0].get_position().y0
fig.add_artist(plt.Line2D([xdiv, xdiv], [ybot, ytop], color="0.55", lw=1.0,
               ls=(0, (2, 2)), transform=fig.transFigure))

fig.savefig(_os.path.join(_REPO, "figures", "fig_data.pdf"))
print("wrote figures/fig_data.pdf")
