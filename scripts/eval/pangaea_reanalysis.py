"""Re-analysis of the published PANGAEA benchmark (Marsocci et al., arXiv
2412.04204v2, Table 5, 100% labelled data), with a reproducible receipt.

The raw 14-model x 10-mIoU-dataset matrix is hardcoded below verbatim from the
paper (BioMassters is excluded: it is RMSE regression, not mIoU). Every number
this study cites is recomputed here and written to
results/metrics/pangaea_reanalysis.json. The raw scores are transcribed from
the published paper.

Run:  python -m scripts.eval.pangaea_reanalysis
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
SOURCE = "Marsocci et al. 2024, PANGAEA (arXiv:2412.04204v2), Table 5 (100% labels)"

DATASETS = ["HLSBurnScars", "MADOS", "PASTIS", "Sen1Floods11", "xView2",
            "FiveBillionPixels", "DynamicEarthNet", "CropTypeMapping",
            "SpaceNet7", "AI4SmallFarms"]                      # 10 mIoU datasets
BINARY_SEG = ["HLSBurnScars", "Sen1Floods11", "SpaceNet7"]     # 2-class segmentation
MODELS = ["CROMA", "DOFA", "GFM-Swin", "Prithvi", "RemoteCLIP", "SatlasNet",
          "Scale-MAE", "SpectralGPT", "S12-MoCo", "S12-DINO", "S12-MAE",
          "S12-Data2Vec", "UNet", "ViT"]                       # 12 GFM + 2 supervised
SUPERVISED = {"UNet", "ViT"}
M = np.array([
    [82.42, 67.55, 32.32, 90.89, 53.27, 51.83, 38.29, 49.38, 59.28, 25.65],
    [80.63, 59.58, 30.02, 89.37, 59.64, 43.18, 39.29, 51.33, 61.84, 27.07],
    [76.90, 64.71, 21.24, 72.60, 59.15, 67.18, 34.09, 46.98, 60.89, 27.19],
    [83.62, 49.98, 33.93, 90.37, 49.35, 46.81, 27.86, 43.07, 56.54, 26.86],
    [76.59, 60.00, 18.23, 74.26, 57.41, 69.19, 31.78, 52.05, 57.76, 25.12],
    [79.96, 55.86, 17.51, 90.30, 52.23, 50.97, 36.31, 46.97, 61.88, 25.13],
    [76.68, 57.32, 24.55, 74.13, 60.72, 67.19, 35.11, 25.42, 62.96, 21.47],
    [80.47, 57.99, 35.44, 89.07, 48.40, 33.42, 37.85, 46.95, 58.86, 26.75],
    [81.58, 51.76, 34.49, 89.26, 51.59, 53.02, 35.44, 48.58, 57.64, 25.38],
    [81.72, 49.37, 36.18, 88.61, 50.56, 51.15, 34.81, 48.66, 56.47, 25.62],
    [81.91, 49.90, 32.03, 87.79, 50.44, 51.92, 34.08, 45.80, 57.13, 24.69],
    [81.91, 44.36, 34.32, 88.15, 51.36, 48.82, 35.90, 54.03, 58.23, 24.23],
    [84.51, 54.79, 31.60, 91.42, 58.68, 60.47, 39.46, 47.57, 62.09, 46.34],
    [81.58, 48.19, 38.53, 87.66, 57.43, 59.32, 36.83, 44.08, 52.57, 38.37],
])


def model_effect_pct(sub):
    """% of variance explained by the model (row) effect after per-dataset
    z-standardisation of the sub-matrix."""
    Z = (sub - sub.mean(0)) / sub.std(0, ddof=0)
    return float(100 * sub.shape[1] * ((Z.mean(1) - Z.mean()) ** 2).sum()
                 / ((Z - Z.mean()) ** 2).sum())


def main():
    nm, nd = M.shape
    mu, colm, rowm = M.mean(), M.mean(0), M.mean(1)
    SS_tot = ((M - mu) ** 2).sum()
    SS_ds = nm * ((colm - mu) ** 2).sum()
    SS_md = nd * ((rowm - mu) ** 2).sum()
    rhos = [spearmanr(M[:, a], M[:, b]).correlation
            for a in range(nd) for b in range(a + 1, nd)]
    naive, corrected = M.mean(1), ((M - colm) / M.std(0, ddof=0)).mean(1)
    wins = {}
    for j in range(nd):
        wins[MODELS[M[:, j].argmax()]] = wins.get(MODELS[M[:, j].argmax()], 0) + 1
    rep = {
        "source": SOURCE, "n_models": nm, "n_gfm": nm - len(SUPERVISED),
        "n_datasets_miou": nd, "datasets": DATASETS, "models": MODELS,
        "naive_variance_decomposition": {
            "dataset_pct": round(100 * SS_ds / SS_tot, 1),
            "model_pct": round(100 * SS_md / SS_tot, 1),
            "residual_pct": round(100 * (SS_tot - SS_ds - SS_md) / SS_tot, 1)},
        "between_dataset_sd": round(colm.std(ddof=0), 1),
        "within_dataset_sd": round((M - colm).std(ddof=0), 1),
        "standardized_model_effect_pct": round(model_effect_pct(M), 1),
        "binary_seg_model_effect_pct": round(
            model_effect_pct(M[:, [DATASETS.index(d) for d in BINARY_SEG]]), 1),
        "cross_dataset_ranking_spearman": round(float(np.mean(rhos)), 2),
        "naive_vs_corrected_ranking_spearman": round(
            float(spearmanr(naive, corrected).correlation), 2),
        "top_model_naive": MODELS[naive.argmax()],
        "top_model_corrected": MODELS[corrected.argmax()],
        "n_models_winning_ge1": len(wins),
        "max_datasets_won_by_one_model": max(wins.values()),
        "wins_per_model": dict(sorted(wins.items(), key=lambda x: -x[1])),
    }
    out = REPO / "results/metrics/pangaea_reanalysis.json"
    json.dump(rep, open(out, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
