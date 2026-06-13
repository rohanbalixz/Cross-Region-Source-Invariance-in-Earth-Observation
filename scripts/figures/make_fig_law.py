"""Input-family retention figure, house style. Out-of-region vs in-region score:
temporal-history inputs sit on the r=1 line; spectral-imagery inputs on ~r=0.8,
including a frozen foundation-model encoder. Crop is greyed (degenerate)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # for figstyle
import figstyle as fs; fs.use_style()

temporal = [("built-up", 0.341, 0.338), ("built-volume", 0.303, 0.302),
            ("population", 0.428, 0.426)]                      # FoM
ndvi     = [("NDVI nowcast (held-out)", 0.968, 0.961)]         # field corr
imagery  = [("water", 0.908, 0.717), ("built-up", 0.684, 0.559),
            ("tree", 0.479, 0.422), ("11-class", 0.319, 0.246)]  # IoU
fm       = [("DINOv2 (frozen)", 0.650, 0.508)]
crop     = [("crop (degenerate)", 0.164, 0.063)]

fig, ax = plt.subplots(figsize=(5.0, 4.6))
lim = 0.98
ax.fill_between([0, lim], [0, lim], [0, 0.8*lim], color=fs.C["grey"], alpha=0.05, lw=0)
ax.plot([0, lim], [0, lim], color="#888888", lw=1.0, zorder=1)
ax.plot([0, lim], [0, 0.8*lim], color=fs.IMAGERY, lw=1.0, ls=(0, (4, 3)), zorder=1)
ax.text(0.95, 0.95, "retention 1.0", rotation=45, color="#888888", fontsize=7, va="bottom", ha="right")
ax.text(0.95, 0.74, "0.8", rotation=39, color=fs.IMAGERY, fontsize=7, va="top", ha="right")


def scat(pts, color, marker, label, lab=True, dx=8, dy=-2):
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    ax.scatter(xs, ys, c=color, marker=marker, s=62, zorder=3, edgecolor="white",
               linewidth=0.8, label=label if lab else None)
    for n, x, y in pts:
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=7, color=color)


scat(temporal, fs.HARMONISED, "o", "temporal-history (own past)")
scat(ndvi,     fs.HARMONISED, "D", "held-out non-urban (NDVI)", dx=-8, dy=6)
scat(imagery,  fs.IMAGERY,  "s", "spectral imagery (exogenous)")
scat(fm,       fs.IMAGERY,  "^", "frozen foundation model", dx=6, dy=-12)
scat(crop,     fs.C["grey"], "x", None, lab=False)

ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_yticks([0, 0.25, 0.5, 0.75])
ax.set_xlabel("in-region score  (train & test same region)")
ax.set_ylabel("out-of-region score  (trained elsewhere)")
ax.set_title("Retention is set by the input representation", fontsize=9, pad=4)
ax.legend(loc="lower right", fontsize=6.8, handletextpad=0.4)
fig.tight_layout(); fig.savefig(_os.path.join(_REPO, "figures", "fig_law.pdf"))
