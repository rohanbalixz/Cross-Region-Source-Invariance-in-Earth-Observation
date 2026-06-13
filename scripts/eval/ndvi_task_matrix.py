"""Held-out, non-urban temporal task: MODIS NDVI vegetation-dynamics nowcasting.

Predict 2022 NDVI from the 2013-2021 annual-max NDVI history with the same small
CNN and the same source-by-target protocol as the GHSL temporal tasks, then read
the retention matrix. Vegetation is non-urban and NDVI rises and falls year to
year, so source-invariance here cannot be a built-up-grows-next-to-itself
artefact. PRE-REGISTERED prediction: retention ~0.99 (source-invariant); a clear
departure would bound the temporal regime to monotone urban growth.

We score each cell four ways: explained variance (R2) of the NDVI field;
skill over persistence (1 - MSE_model/MSE_persist, the non-trivial test that the
model beats "next = last"); the field correlation; and the change correlation
dcorr (predicted vs true year-on-year NDVI change, the direct analogue of the
change-based figure of merit used for built-up). retention = off-diagonal mean /
diagonal mean for each score.

Usage: python -m scripts.eval.ndvi_task_matrix
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from scripts.eval.models import SimpleCNN

REPO = Path(__file__).resolve().parents[2]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
REGIONS = ["south_asia","ssa","east_asia","andes","mena","sea","eeca","oceania"]
SEED = 20260525; P = 128; STRIDE = 64; IN_T = 9   # years 0..8 -> predict year 9


def gather(region, stride):
    X, Y, Prev = [], [], []
    for f in glob.glob(str(REPO/f"data/ndvi/{region}/*/ndvi_series.npz")):
        a = np.load(f)["ndvi"].astype(np.float32)          # (10,H,W)
        if a.shape[0] < IN_T + 1: continue
        H, W = a.shape[1:]
        for i in range(0, H-P+1, stride):
            for j in range(0, W-P+1, stride):
                tile = a[:, i:i+P, j:j+P]
                if not np.isfinite(tile).all(): continue
                if tile[:IN_T].std() < 1e-4: continue       # skip dead/constant tiles
                X.append(tile[:IN_T]); Y.append(tile[IN_T]); Prev.append(tile[IN_T-1])
    if not X: return None, None, None
    X = torch.from_numpy(np.stack(X))[:, :, None]           # (N,9,1,P,P)
    Y = torch.from_numpy(np.stack(Y))[:, None]              # (N,1,P,P)
    Prev = torch.from_numpy(np.stack(Prev))[:, None]
    return X, Y, Prev


def train(region, epochs=15):
    X, Y, _ = gather(region, STRIDE)
    if X is None or len(X) < 20: return None
    n = len(X); tr = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:int(0.85*n)].to(DEV)
    X, Y = X.to(DEV), Y.to(DEV)
    m = SimpleCNN(input_channels=IN_T).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator(device=DEV).manual_seed(SEED)
    for _ in range(epochs):
        m.train(); perm = tr[torch.randperm(len(tr), generator=g, device=DEV)]
        for k in range(0, len(perm), 16):
            b = perm[k:k+16]; opt.zero_grad()
            loss = nn.functional.mse_loss(m(X[b]), Y[b]); loss.backward(); opt.step()
    m.eval(); return m


def scores(model, region):
    X, Y, Prev = gather(region, P)                          # non-overlap eval
    if X is None: return None
    with torch.no_grad():
        pr = torch.cat([model(X[k:k+16].to(DEV)).cpu() for k in range(0, len(X), 16)])
    p = pr.numpy().reshape(-1); y = Y.numpy().reshape(-1); pv = Prev.numpy().reshape(-1)
    r2 = 1 - ((y-p)**2).sum() / ((y-y.mean())**2).sum()
    skill = 1 - ((y-p)**2).mean() / max(((y-pv)**2).mean(), 1e-9)
    corr = pearsonr(p, y)[0]
    dt, dp = y-pv, p-pv
    dcorr = pearsonr(dp, dt)[0] if dp.std() > 1e-9 and dt.std() > 1e-9 else float("nan")
    return {"r2": float(r2), "persist_skill": float(skill), "corr": float(corr), "dcorr": float(dcorr)}


def main():
    models = {r: train(r) for r in REGIONS}; models = {r: m for r, m in models.items() if m is not None}
    rr = list(models)
    print(f"trained sources: {rr}", flush=True)
    mat = {s: {t: scores(models[s], t) for t in rr} for s in models}
    out = {"task": "ndvi_nowcast(temporal,non-urban)", "n_sources": len(rr), "regions": rr,
           "preregistered": "retention ~0.99 (source-invariant)"}
    for key in ["r2", "persist_skill", "corr", "dcorr"]:
        diag = np.nanmean([mat[r][r][key] for r in rr])
        off = np.nanmean([mat[s][t][key] for s in rr for t in rr if s != t])
        M = np.array([[mat[s][t][key] for t in rr] for s in rr])
        inv = np.nanmean([spearmanr(M[a], M[b]).correlation for a in range(len(rr)) for b in range(a+1, len(rr))])
        out[key] = {"in": round(float(diag), 3), "out": round(float(off), 3),
                    "retention": round(float(off/diag), 3), "source_inv": round(float(inv), 3)}
        print(f"{key:14} in={diag:.3f} out={off:.3f} retention={off/diag:.3f} src_inv={inv:.3f}", flush=True)
    json.dump({**out, "matrix": mat}, open(REPO/"results/metrics/ndvi_task_matrix.json", "w"), indent=1)
    print("saved results/metrics/ndvi_task_matrix.json", flush=True)


if __name__ == "__main__":
    main()
