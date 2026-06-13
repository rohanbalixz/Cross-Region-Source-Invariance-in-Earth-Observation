"""Per-region difficulty vs change rate at n=20, house style. Difficulty = full
20x20 transfer-matrix column mean; change = all-pixel change fraction (the same
definition as the original n=8 panel). Data: results/metrics/confound_n20_allpix.json.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json
from pathlib import Path
import os as _os, sys as _sys
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # for figstyle
import figstyle as fs; fs.use_style()

d = json.load(open(Path(_REPO) / "results/metrics/confound_n20_allpix.json"))
regs = d["regions"]
chg = np.array([d["change_allpix"][r] for r in regs])
fom = np.array([d["difficulty"][r] for r in regs])
rho = d["spearman"]; ci = d["ci95"]; R2 = d["R2"]

nice = {"china_interior": "China interior", "east_asia": "East Asia", "weur": "W. Europe",
        "oceania": "Oceania", "andes": "Andes", "canada": "Canada"}
order = np.argsort(fom)
label_idx = set(order[:3].tolist()).union(order[-3:].tolist())

A = np.polyfit(chg, fom, 1); xs = np.linspace(chg.min(), chg.max(), 50)
fig, ax = plt.subplots(figsize=(5.0, 3.3))
ax.plot(xs, np.polyval(A, xs), '-', color="#999999", lw=1.3, zorder=1,
        label=f"linear fit  ($R^2$ = {R2:.2f})")
ax.scatter(chg, fom, s=40, color=fs.C["blue"], zorder=3, edgecolor="white", linewidth=0.7)
for k in range(len(regs)):
    if k in label_idx and regs[k] in nice:
        ax.annotate(nice[regs[k]], (chg[k], fom[k]), fontsize=6.8,
                    xytext=(5, -1), textcoords="offset points", color="#555555")
ax.set_xlabel("fraction of pixels that changed, 2010–15")
ax.set_ylabel("per-region difficulty\n(mean FoM as a target)")
ax.set_title(f"Difficulty tracks change rate  (Spearman {rho:.2f}, 95% CI "
             f"[{ci[0]:.2f}, {ci[1]:.2f}], $n=20$)", fontsize=8.5, pad=5)
ax.legend(loc="lower right", fontsize=7.2)
fig.tight_layout(); fig.savefig(Path(_REPO) / "figures/fig_confound.pdf")
print("wrote figures/fig_confound.pdf (n=20, house style)")
