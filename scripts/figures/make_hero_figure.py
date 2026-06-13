"""Hero figure (the two headline results), recomputed from live receipts,
in the shared house style.

(a) deflation: a parameter-free baseline beats the trained CNN on the figure of
    merit in every region (all points below y=x).
(b) the collapse: same target (built-up 2015), retention falls as the input goes
    from a globally-harmonised product (GHSL history) to single-date Sentinel-2
    to a raw multi-decade Landsat history.

Run: python -m scripts.paper.make_hero_figure  ->  paper/figures/fig_hero.pdf
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for figstyle
import figstyle as fs
fs.use_style()


def _ret(path, *keys):
    d = json.load(open(REPO / path))
    for k in keys:
        d = d[k]
    return d["mean"], d["sd"]


mb = json.load(open(REPO / "results/metrics/morph_baseline.json"))["per_region"]
regions = list(mb)
linext = np.array([mb[r]["linext"] for r in regions])
cnn = np.array([mb[r]["cnn_in"] for r in regions])

g_m, g_s = _ret("results/metrics/multiseed_matrix_cnn.json", "aggregate", "retention")
s_m, s_s = _ret("results/metrics/imagery_multiseed.json", "builtup_binary", "aggregate", "retention")
l_m, l_s = _ret("results/metrics/landsat_temporal_matrix_landsat_history_g64_n20_multiseed.json", "aggregate", "retention")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.95))

# (a) deflation scatter
lim = [0.18, 0.70]
a1.fill_between(lim, lim, [lim[0], lim[0]], color=fs.C["red"], alpha=0.05, zorder=0, lw=0)
a1.plot(lim, lim, ls=(0, (4, 3)), c="#999999", lw=1.0, zorder=1)
a1.scatter(linext, cnn, s=44, c=fs.C["red"], edgecolor="white", linewidth=0.7, zorder=3)
a1.text(0.66, 0.665, "$y=x$", color="#888888", fontsize=7.5, rotation=40, va="center")
a1.annotate("every region below the line:\nthe model never beats the baseline",
            xy=(float(linext.mean()), float(cnn.mean())), xytext=(0.52, 0.27),
            color=fs.C["red"], fontsize=7.5, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=fs.C["red"], lw=0.7, alpha=0.6))
a1.set_xlim(lim); a1.set_ylim(lim); a1.set_aspect("equal")
a1.set_xticks([0.2, 0.4, 0.6]); a1.set_yticks([0.2, 0.4, 0.6])
a1.set_xlabel("parameter-free baseline  (FoM)")
a1.set_ylabel("trained CNN  (FoM)")
a1.set_title("Source-invariance is trivial", fontsize=9, color=fs.C["ink"], pad=14)
a1.text(0.5, 1.045, "a one-line baseline wins in every region", transform=a1.transAxes,
        ha="center", va="bottom", fontsize=7.5, color="#666666")
fs.panel(a1, "a", x=-0.22)

# (b) provenance collapse
labels = ["GHSL\nhistory", "single-date\nSentinel-2", "raw Landsat\nhistory"]
means = [g_m, s_m, l_m]; sds = [g_s, s_s, l_s]
cols = [fs.HARMONISED, fs.RAWSENSOR, fs.RAWSENSOR]
xb = np.arange(3)
a2.axhline(1.0, ls=":", c="#aaaaaa", lw=0.9, zorder=1)
a2.bar(xb, means, yerr=sds, color=cols, capsize=3, edgecolor="white", linewidth=0.6,
       width=0.6, error_kw=dict(lw=0.9, ecolor="#444444"), zorder=2)
for i, (m, s) in enumerate(zip(means, sds)):
    a2.text(i, m + s + 0.04, f"{m:.2f}", ha="center", fontsize=8, color=fs.C["ink"])
a2.set_xticks(xb); a2.set_xticklabels(labels, fontsize=7.3)
for tick, c in zip(a2.get_xticklabels(), cols):    # tick colour = provenance class
    tick.set_color(c)
a2.set_ylabel("cross-region retention  (out $\\div$ in)")
a2.set_ylim(0, 1.22); a2.set_yticks([0, 0.5, 1.0])
a2.text(0.02, 0.93, "harmonised", transform=a2.transAxes, fontsize=6.8,
        color=fs.HARMONISED, ha="left")
a2.text(0.97, 0.40, "raw sensor", transform=a2.transAxes, fontsize=6.8,
        color=fs.RAWSENSOR, ha="right")
a2.set_title("Provenance decides transfer", fontsize=9, color=fs.C["ink"], pad=14)
a2.text(0.5, 1.045, "same target, only the input changes", transform=a2.transAxes,
        ha="center", va="bottom", fontsize=7.5, color="#666666")
fs.panel(a2, "b", x=-0.2)

fig.subplots_adjust(wspace=0.42)
fig.savefig(OUT / "fig_hero.pdf"); plt.close(fig)
print(f"wrote fig_hero.pdf | deflation CNN<linext {(cnn<linext).sum()}/{len(regions)} | "
      f"retention {g_m:.2f}->{s_m:.2f}->{l_m:.2f}")
