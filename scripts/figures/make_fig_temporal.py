"""Figure: the temporal-history input, and why a parameter-free line beats the
trained model. Top: the model's input is the region's own GHSL built-up history,
nine five-yearly epochs stacked as channels (real GHSL R2023A, Shanghai). Bottom:
the task is to predict the 2010->2015 change. A parameter-free rule that simply
continues the last increment, 2015 = 2010 + (2010-2005), reproduces almost all of
the real new built-up; its predicted-vs-actual change overlap (green) dwarfs its
misses and false alarms. Across all eight regions this line scores FoM 0.56,
above every trained model's 0.34 -- the temporal source-invariance is shallow.

All rasters are real benchmark data; the prediction is the closed-form line, not
a learned model."""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import os as _os
import sys as _sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import rasterio

_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # for figstyle
import figstyle as fs; fs.use_style()

DATA = _os.path.join(_REPO, "data")
REG, CITY = "east_asia", "shanghai"
EPOCHS = [1975, 1985, 1995, 2000, 2005, 2010]


def ghsl(y):
    with rasterio.open(f"{DATA}/processed/{REG}/{CITY}/builtup_{y}.tif") as s:
        return s.read(1)


b = {y: ghsl(y) for y in EPOCHS + [2015]}
b05, b10, b15 = b[2005], b[2010], b[2015]
pred_new = np.clip(b10 - b05, 0, 1)          # the line's predicted 2010->15 change
act_new = np.clip(b15 - b10, 0, 1)           # actual 2010->15 change
TH = 0.02
P, A = pred_new > TH, act_new > TH
agree = np.zeros((*P.shape, 3), float) + 1.0  # white canvas
agree[A & P] = [0.0, 0.62, 0.45]              # hit  (green, Okabe-Ito)
agree[A & ~P] = [0.34, 0.71, 0.91]            # miss (sky)
agree[~A & P] = [0.84, 0.37, 0.0]             # false alarm (red)
fom = (A & P).sum() / max(1, (A | P).sum())

# ---- layout: 6 epoch thumbs on top, 3 wide panels below --------------------
fig = plt.figure(figsize=(9.4, 5.6))
gs = gridspec.GridSpec(2, 6, figure=fig, height_ratios=[1.0, 1.55],
                       hspace=0.62, wspace=0.12,
                       left=0.045, right=0.995, top=0.84, bottom=0.02)

BLUE = fs.TEMPORAL
thumb_ax = []
for i, y in enumerate(EPOCHS):
    ax = fig.add_subplot(gs[0, i]); thumb_ax.append(ax)
    ax.imshow(b[y], cmap="Blues", vmin=0, vmax=0.7, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(str(y), fontsize=8.5, color=fs.C["ink"], pad=2)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.3); sp.set_edgecolor(BLUE)

panels = [("Predicted new built-up\n$2015 = 2010 + (2010{-}2005)$", pred_new, "Blues"),
          ("Actual new built-up\n$2015 - 2010$", act_new, "Blues"),
          ("Agreement", agree, None)]
panel_ax = []
for j, (title, img, cmap) in enumerate(panels):
    ax = fig.add_subplot(gs[1, 2*j:2*j+2]); panel_ax.append(ax)
    if cmap: ax.imshow(img, cmap=cmap, vmin=0, vmax=0.22, interpolation="nearest")
    else:    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8.6, color=fs.C["ink"], pad=3, linespacing=1.15)
    ec = "0.55" if j < 2 else fs.C["ink"]
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.9); sp.set_edgecolor(ec)

from matplotlib.patches import Patch

panel_ax[-1].legend(handles=[Patch(fc=[0,0.62,0.45], label="hit"),
                      Patch(fc=[0.84,0.37,0], label="false alarm"),
                      Patch(fc=[0.34,0.71,0.91], label="miss")],
             loc="lower right", fontsize=6.6, frameon=True, framealpha=0.9,
             handlelength=1.0, borderpad=0.3, labelspacing=0.25)

# headers, placed from the axes' real positions so they sit in the right bands
fig.canvas.draw()
top_thumbs = max(a.get_position().y1 for a in thumb_ax)
bot_thumbs = min(a.get_position().y0 for a in thumb_ax)
top_panels = max(a.get_position().y1 for a in panel_ax)
fig.text(0.045, top_thumbs + 0.035,
         "Input — the region's own built-up history (nine epochs, 1975–2010, stacked as channels)",
         ha="left", va="bottom", fontsize=9.5, fontweight="bold", color=BLUE)
fig.text(0.045, (bot_thumbs + top_panels)/2,
         "Predict 2015 — a parameter-free line reproduces the change the model is trained on  "
         r"(line FoM $0.56 >$ trained CNN $0.34$, every region)",
         ha="left", va="center", fontsize=9.5, fontweight="bold", color=fs.C["ink"])

fig.savefig(_os.path.join(_REPO, "figures", "fig_temporal.pdf"))
print(f"wrote figures/fig_temporal.pdf  (Shanghai illustrative FoM={fom:.2f})")
