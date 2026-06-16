"""Single-source data-scaling curve (reviewer item: rule out the data-starvation
confound behind the no-home-field result).

For a fixed CNN, sweep the per-source training-set size from 10% of the default
sliding-window tiles up to ~3x the default (denser stride). At each level we
retrain every region, build the full 8x8 matrix, and report in-region (diagonal)
vs out-of-region (off-diagonal) FoM and the home-field gap. If the gap stays ~0
as data grows -- and shows no upward trend -- the absent home-field advantage is
NOT an under-fitting artefact. If the gap climbs with data, the headline claim is
wrong and must be revised.

Usage: python -m scripts.eval.data_scaling_curve
"""
import json, tempfile
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scripts.acquire.regions import CITIES
from scripts.eval.cross_region_eval import process_city
from scripts.eval.cross_region_train import gather_tiles, soft_jaccard
from scripts.eval.capacity_sweep import WidthCNN, region_fom

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"; T = "t=0.01"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]
# (label, fraction of tiles, sliding-window stride). Smaller stride = denser = more tiles.
LEVELS = [("10%", 0.10, 64), ("25%", 0.25, 64), ("50%", 0.50, 64),
          ("100%", 1.00, 64), ("~3x", 1.00, 32)]
SEED = 20260525


def train(source, frac, stride, epochs=20, seed=SEED):
    torch.manual_seed(seed)
    X, Y = gather_tiles(source, stride=stride)
    n = len(X); k = max(16, int(frac * n))
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:k]
    X, Y = X[idx].to(DEV), Y[idx].to(DEV)
    cut = int(0.85 * len(X)); tr = torch.arange(len(X))[:cut].to(DEV)
    m = WidthCNN(1.0).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(epochs):
        m.train(); perm = tr[torch.randperm(len(tr), generator=g, device=DEV)]
        for b0 in range(0, len(perm), 16):
            b = perm[b0:b0+16]; opt.zero_grad()
            p = m(X[b]); loss = nn.functional.mse_loss(p, Y[b]) + 0.5 * soft_jaccard(p, Y[b])
            loss.backward(); opt.step()
    m.eval(); return m, k


def main():
    rows = []
    for label, frac, stride in LEVELS:
        models = {}; ntiles = []
        for s in REGIONS:
            mdl, k = train(s, frac, stride); models[s] = mdl; ntiles.append(k)
        mat = {s: {t: region_fom(models[s], t) for t in REGIONS} for s in REGIONS}
        diag = np.mean([mat[r][r] for r in REGIONS if mat[r][r] is not None])
        off = np.mean([mat[s][t] for s in REGIONS for t in REGIONS
                       if s != t and mat[s][t] is not None])
        rows.append({"level": label, "frac": frac, "stride": stride,
                     "mean_tiles_per_source": round(float(np.mean(ntiles)), 1),
                     "in_region": round(float(diag), 4), "out_region": round(float(off), 4),
                     "home_field_gap": round(float(diag - off), 4)})
        print(f"[scaling] {label:5s} tiles~{np.mean(ntiles):4.0f}  in={diag:.4f}  "
              f"out={off:.4f}  gap={diag-off:+.4f}", flush=True)
    out = REPO / "results/metrics/data_scaling_curve.json"
    json.dump({"seed": SEED, "width": 1.0, "arch": "cnn", "levels": rows}, open(out, "w"), indent=1)
    gaps = [r["home_field_gap"] for r in rows]
    print(f"\nsaved {out}")
    print(f"gap range over a {rows[-1]['mean_tiles_per_source']/rows[0]['mean_tiles_per_source']:.0f}x "
          f"data range: [{min(gaps):+.4f}, {max(gaps):+.4f}]")
    print("=> flat & ~0 -> not under-fitting; rising with data -> headline wrong.")


if __name__ == "__main__":
    main()
