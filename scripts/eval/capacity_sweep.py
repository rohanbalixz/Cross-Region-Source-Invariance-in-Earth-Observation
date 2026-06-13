"""Capacity x data-size sweep for the no-home-field result .

For each (width multiplier, training-fraction) we retrain the CNN on every
source region and evaluate the full matrix, then report mean in-region vs
out-of-region FoM. If the home-field gap stays ~0 as capacity and data grow,
the no-home-field finding is not an under-fitting artefact.

Usage: python -m scripts.eval.capacity_sweep
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.acquire.regions import CITIES
from scripts.eval.cross_region_eval import process_city
from scripts.eval.cross_region_train import gather_tiles, soft_jaccard

REPO = Path(__file__).resolve().parents[2]; PROC = REPO/"data/processed"; T="t=0.01"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
REGIONS = ["south_asia","ssa","east_asia","andes","mena","sea","eeca","oceania"]
WIDTHS = [1.0, 2.0, 4.0]      # 1x, 2x, 4x channel width
FRACS  = [0.5, 1.0]          # half vs full per-source training set


class WidthCNN(nn.Module):
    def __init__(self, w=1.0, in_ch=24):
        super().__init__()
        c = [max(4, round(x*w)) for x in (64, 64, 32, 16)]
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c[0], 3, padding=1), nn.BatchNorm2d(c[0]), nn.ReLU(),
            nn.Conv2d(c[0], c[1], 3, padding=1), nn.BatchNorm2d(c[1]), nn.ReLU(),
            nn.Conv2d(c[1], c[2], 3, padding=1), nn.BatchNorm2d(c[2]), nn.ReLU(),
            nn.Conv2d(c[2], c[3], 3, padding=1), nn.BatchNorm2d(c[3]), nn.ReLU(),
            nn.Conv2d(c[3], 1, 1))
    def forward(self, x):
        B,Tt,C,H,W = x.shape
        return torch.sigmoid(self.net(x.reshape(B, Tt*C, H, W)))


def train_one(source, w, frac, epochs=20, seed=20260525):
    torch.manual_seed(seed); X, Y = gather_tiles(source)
    n = len(X); k = max(16, int(frac*n))
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:k]
    X, Y = X[idx].to(DEV), Y[idx].to(DEV)
    cut = int(0.85*len(X)); tr = torch.arange(len(X))[:cut].to(DEV)
    m = WidthCNN(w).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(epochs):
        m.train(); perm = tr[torch.randperm(len(tr), generator=g, device=DEV)]
        for b0 in range(0, len(perm), 16):
            b = perm[b0:b0+16]; opt.zero_grad()
            p = m(X[b]); loss = nn.functional.mse_loss(p, Y[b]) + 0.5*soft_jaccard(p, Y[b])
            loss.backward(); opt.step()
    m.eval(); return m


def region_fom(model, region):
    out = Path(tempfile.mkdtemp()); fs=[]
    for c in [c for c in CITIES if c.region==region]:
        if not (PROC/region/c.name/"eval_metrics.json").exists(): continue
        try: process_city(c, PROC, {"cnn":model}, out/f"{c.name}.json", DEV)
        except Exception: continue
        per=[r["models"]["cnn"]["fom"][T]["fom"] for r in json.load(open(out/f"{c.name}.json"))
             if r.get("models",{}).get("cnn",{}).get("fom",{}).get(T,{}) and r["models"]["cnn"]["fom"][T]["fom"] is not None]
        if per: fs.append(np.mean(per))
    return float(np.mean(fs)) if fs else None


def main():
    rows=[]
    for w in WIDTHS:
        for frac in FRACS:
            models={s: train_one(s, w, frac) for s in REGIONS}
            mat={s:{t:region_fom(models[s],t) for t in REGIONS} for s in REGIONS}
            diag=np.mean([mat[r][r] for r in REGIONS if mat[r][r] is not None])
            off=np.mean([mat[s][t] for s in REGIONS for t in REGIONS if s!=t and mat[s][t] is not None])
            params=sum(p.numel() for p in WidthCNN(w).parameters())
            rows.append({"width":w,"frac":frac,"params":params,"in_region":round(float(diag),4),
                         "out_region":round(float(off),4),"home_field_gap":round(float(diag-off),4)})
            print(f"[sweep] w={w} frac={frac} params={params} in={diag:.4f} out={off:.4f} gap={diag-off:+.4f}", flush=True)
    json.dump(rows, open(REPO/"results/metrics/capacity_sweep.json","w"), indent=1)
    print("\nsaved results/metrics/capacity_sweep.json")
    print("=> if home_field_gap stays ~0 across all rows (esp. high width+full data), the no-home-field result is NOT under-fitting.")
if __name__=="__main__": main()
