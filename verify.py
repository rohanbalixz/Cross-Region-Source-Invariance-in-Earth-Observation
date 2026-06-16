#!/usr/bin/env python3
"""Smoke test: re-derive the headline results from the committed receipts.

This needs no acquired data and no deep-learning stack -- only numpy/scipy and
the JSON receipts in results/metrics/. It recomputes, from scratch, the numbers
the study reports and checks them against the saved values, so anyone can
confirm the key results in a few seconds:

  * the PANGAEA variance decomposition, recomputed from the published score table;
  * the twenty-region source-inertness (variance decomposition of the matrix);
  * the parameter-free deflation (linear extrapolation beats the trained CNN);
  * the mapping-stability / retention correlation.

Run:  python verify.py        (exit code 0 = all checks pass)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
METRICS = REPO / "results" / "metrics"
sys.path.insert(0, str(REPO))

_checks = []


def check(name, ok, detail=""):
    _checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def load(name):
    return json.load(open(METRICS / f"{name}.json"))


# --- 0. every receipt is valid JSON --------------------------------------
print("Receipts:")
bad = []
for f in sorted(METRICS.glob("*.json")):
    try:
        json.load(open(f))
    except Exception as e:  # noqa: BLE001
        bad.append((f.name, str(e)))
check(f"all {len(list(METRICS.glob('*.json')))} receipts parse as valid JSON",
      not bad, "; ".join(f"{n}: {e}" for n, e in bad))

# --- 1. PANGAEA: recompute from the published table -----------------------
print("\nPANGAEA re-analysis (recomputed from the published score matrix):")
from scripts.eval.pangaea_reanalysis import BINARY_SEG, DATASETS, M, model_effect_pct  # noqa: E402

nm, nd = M.shape
mu, colm, rowm = M.mean(), M.mean(0), M.mean(1)
ss_tot = ((M - mu) ** 2).sum()
ds_pct = 100 * nm * ((colm - mu) ** 2).sum() / ss_tot
md_pct = 100 * nd * ((rowm - mu) ** 2).sum() / ss_tot
me_pct = model_effect_pct(M)
bin_pct = model_effect_pct(M[:, [DATASETS.index(d) for d in BINARY_SEG]])
rec = load("pangaea_reanalysis")
check("dataset variance share 90.9%", close(ds_pct, 90.9, 0.2), f"got {ds_pct:.1f}")
check("model variance share 0.9%", close(md_pct, 0.9, 0.2), f"got {md_pct:.1f}")
check("standardized model effect 14.4%", close(me_pct, 14.4, 0.3), f"got {me_pct:.1f}")
check("binary-seg model effect 41.7%", close(bin_pct, 41.7, 0.3), f"got {bin_pct:.1f}")
check("recompute matches committed receipt",
      close(ds_pct, rec["naive_variance_decomposition"]["dataset_pct"], 0.1)
      and close(me_pct, rec["standardized_model_effect_pct"], 0.1))

# --- 2. twenty-region matrix: source is inert -----------------------------
print("\nTwenty-region transfer matrix (the source barely moves the score):")
d20 = load("full_matrix_n20_cnn")
regs = d20["regions"]
Mx = np.array([[d20["matrix"][s].get(t, np.nan) for t in regs] for s in regs], float)
gm, cm, rm = np.nanmean(Mx), np.nanmean(Mx, 0), np.nanmean(Mx, 1)
tot = np.nanvar(Mx)
tgt_pct = 100 * np.nanmean((cm - gm) ** 2) / tot
src_pct = 100 * np.nanmean((rm - gm) ** 2) / tot
check("target explains ~99.7% of matrix variance", close(tgt_pct, 99.7, 0.6), f"got {tgt_pct:.1f}")
check("source explains <=0.5%", src_pct <= 0.5, f"got {src_pct:.2f}")
check("home-field gap ~ 0", close(d20["home_field_gap"], 0.0, 0.01), f"{d20['home_field_gap']:.4f}")
check("source-invariance ~ 0.97", close(d20["source_invariance"], 0.97, 0.02))
ds_native = [r["home_field_gap"] for r in load("data_scaling_curve")["levels"]
             if r["stride"] == 64]
check("data-scaling: home-field gap stays ~0 over the native 10x data range",
      max(abs(g) for g in ds_native) <= 0.001,
      f"max|gap|={max(abs(g) for g in ds_native):.4f} over {len(ds_native)} levels")

# --- 3. deflation: a parameter-free line beats the trained model ----------
print("\nDeflation (parameter-free extrapolation vs trained CNN):")
mb = load("morph_baseline")["summary"]
check("linear extrapolation beats the CNN on FoM",
      mb["best_trivial_mean"] > mb["cnn_in_mean"],
      f"linext {mb['best_trivial_mean']:.3f} > cnn {mb['cnn_in_mean']:.3f}")
check("linext FoM ~ 0.56", close(mb["best_trivial_mean"], 0.56, 0.03))

# --- 4. retention spectrum is a measured property -------------------------
print("\nMapping stability vs cross-region retention:")
ms = load("mapping_stability")["nondegenerate"]
check("stability and retention correlate (Pearson ~ 0.96)",
      close(ms["pearson"], 0.96, 0.04), f"got {ms['pearson']}")

# --- summary --------------------------------------------------------------
n_pass, n_tot = sum(_checks), len(_checks)
print(f"\n{'='*60}\n{n_pass}/{n_tot} checks passed\n{'='*60}")
sys.exit(0 if n_pass == n_tot else 1)
