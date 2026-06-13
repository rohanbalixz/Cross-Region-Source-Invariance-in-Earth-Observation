"""Figure: cross-region retention is a SPECTRUM, not a clean harmonised/raw
binary. Same target (built-up 2015), vary only the input; retention is ordered by
how globally stable the input-to-label mapping is. Colour marks provenance class,
and the key honest point is that the classes OVERLAP: a raw sensor with a
globally-invariant physical signal (Sentinel-1 SAR, 0.89) out-transfers a
harmonised land-cover classification (WorldCover, 0.85). Bars are mean +/- 1 SD
over five seeds (three for none here); the dagger marks a degenerate control."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # for figstyle
import figstyle as fs; fs.use_style()

HARM, RAW, MEAS = "harmonised representation", "raw sensor", "uniform measurement"
COL = {HARM: fs.HARMONISED, RAW: fs.RAWSENSOR, MEAS: fs.MEASUREMENT}
# label, retention mean, retention SD, class, degenerate?  (all n=20, 5 seeds,
# EXCEPT single-date Sentinel-2 which stays n=8 -- marked with * in its label)
rows = [
    ("GHSL history",            1.00, 0.01, HARM, False),
    ("Sentinel-1 radar",        0.899, 0.028, RAW,  False),
    ("WorldCover land cover\n(built-up class withheld)", 0.817, 0.015, HARM, False),
    ("single-date Sentinel-2$^{*}$",  0.814, 0.045, RAW,  False),
    ("MODIS NDVI",              0.679, 0.036, MEAS, True),
    ("NDBI index",              0.5951, 0.027, MEAS, False),
    ("raw Landsat history",     0.550, 0.008, RAW,  False),
]
rows.sort(key=lambda r: r[1])                       # ascending -> highest at top
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
sds = [r[2] for r in rows]; cols = [COL[r[3]] for r in rows]; deg = [r[4] for r in rows]
y = np.arange(len(rows))

fig, ax = plt.subplots(figsize=(6.4, 3.3))
bars = ax.barh(y, vals, xerr=sds, color=cols, edgecolor="black", linewidth=0.7,
               height=0.64, zorder=3,
               error_kw=dict(ecolor="0.25", elinewidth=0.9, capsize=2.2, capthick=0.9))
for b, d in zip(bars, deg):
    if d:
        b.set_hatch("////"); b.set_alpha(0.55)
ax.axvline(1.0, color="0.55", ls=":", lw=1.0, zorder=1)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlim(0, 1.30); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("cross-region retention  (out-of-region $\\div$ in-region score)")
for yi, v, s, d in zip(y, vals, sds, deg):
    ax.text(v + s + 0.015, yi, f"{v:.2f}" + ("$^\\dagger$" if d else ""),
            va="center", ha="left", fontsize=7.8)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=3)

# mark the overlap: a raw input (SAR) sits above a harmonised one (WorldCover)
i_sar = labels.index("Sentinel-1 radar")
i_wc = [i for i, l in enumerate(labels) if l.startswith("WorldCover")][0]
xb = 1.10
ax.plot([xb, xb], [i_wc, i_sar], color="0.35", lw=1.0, zorder=5, clip_on=False)
for yy in (i_wc, i_sar):
    ax.plot([xb, xb - 0.02], [yy, yy], color="0.35", lw=1.0, zorder=5, clip_on=False)
ax.text(xb + 0.03, (i_wc + i_sar) / 2, "raw $>$\nharmonised\nhere", fontsize=6.8,
        color="0.20", ha="left", va="center", linespacing=1.05)

leg = [Patch(facecolor=COL[c], edgecolor="black", lw=0.7, label=c) for c in (HARM, MEAS, RAW)]
ax.legend(handles=leg, frameon=False, fontsize=7.3, loc="lower right", handlelength=1.1,
          borderpad=0.3, labelspacing=0.3)
ax.set_title("Cross-region retention is a spectrum, not a clean harmonised/raw split",
             fontsize=9.2, pad=6)
ax.text(0.0, -0.235, r"$n=20$ regions, 5 seeds  ($^{*}$single-date Sentinel-2 at $n=8$)",
        transform=ax.transAxes, fontsize=6.6, color="0.4", ha="left")
fig.tight_layout()
fig.savefig(_os.path.join(_REPO, "figures", "fig_dissociation.pdf"), bbox_inches="tight")
print("wrote figures/fig_dissociation.pdf")
