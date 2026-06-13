"""The retention spectrum is a MEASURED property, not a post-hoc story. x-axis:
mapping stability, computed independently of the CNN (does one global probe
predict built-up as well as per-region probes? -- no transfer evaluation).
y-axis: the CNN cross-region retention. Across the non-degenerate inputs the two
are nearly collinear (Pearson 0.97), so retention reflects how region-invariant
each input's relationship to the target intrinsically is."""
import json
import matplotlib; matplotlib.use("Agg")
import os as _os
import sys as _sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # for figstyle
import figstyle as fs; fs.use_style()

d = json.load(open(_os.path.join(_REPO, "results/metrics/mapping_stability.json")))["per_input"]
CLASS = {"ghsl_history": ("GHSL history", fs.HARMONISED, "o", False),
         "worldcover": ("WorldCover", fs.HARMONISED, "o", False),
         "sar": ("SAR", fs.RAWSENSOR, "s", False),
         "raw_landsat": ("raw Landsat", fs.RAWSENSOR, "s", False),
         "ndbi": ("NDBI", fs.MEASUREMENT, "D", False),
         "ndvi": ("MODIS NDVI", fs.MEASUREMENT, "D", True)}   # degenerate

fig, ax = plt.subplots(figsize=(4.3, 3.4))
nd_s, nd_r = [], []
for k, (lbl, col, mk, degen) in CLASS.items():
    s, r = d[k]["stability"], d[k]["cnn_retention"]
    ax.scatter(s, r, c="white" if degen else col, edgecolor=col, marker=mk, s=70,
               linewidth=1.4, zorder=3, hatch="////" if degen else None)
    dx, dy = (-8, 6) if k in ("worldcover", "sar") else (8, -2)
    ax.annotate(lbl + (r"$^\dagger$" if degen else ""), (s, r), textcoords="offset points",
                xytext=(dx, dy), fontsize=7, color=col)
    if not degen:
        nd_s.append(s); nd_r.append(r)
# fit through the non-degenerate inputs
A = np.polyfit(nd_s, nd_r, 1); xs = np.linspace(min(nd_s) - 0.02, 1.005, 50)
ax.plot(xs, np.polyval(A, xs), color="#999999", lw=1.2, zorder=1)
pr, _ = pearsonr(nd_s, nd_r)
ax.text(0.04, 0.93, f"Pearson $r={pr:.2f}$\n(non-degenerate inputs)", transform=ax.transAxes,
        fontsize=7.5, va="top", color=fs.C["ink"])
ax.set_xlabel("mapping stability  (one global map $\\div$ per-region maps)")
ax.set_ylabel("CNN cross-region retention")
ax.set_title("Retention tracks an independently measured\nmapping stability",
             fontsize=9, pad=4)
ax.set_xlim(0.70, 1.03); ax.set_ylim(0.45, 1.05)
fig.tight_layout()
fig.savefig(_os.path.join(_REPO, "figures", "fig_mapstab.pdf"))
print("wrote figures/fig_mapstab.pdf")
