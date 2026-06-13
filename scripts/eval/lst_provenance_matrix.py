"""Does the provenance dissociation (harmonised product
transfers, raw sensor does not) hold on a NON-urban, NON-land-cover target?
Target = MODIS LST 2015 (a climate biophysical variable, Kelvin). Same target,
two inputs, source-by-target matrix, field-correlation retention:

  * harmonised input : LST's own 2011-2014 history -> LST 2015   (expect ~1.0)
  * raw input        : raw multi-temporal Landsat reflectance    (expect <1.0)

Regression (MSE), retention = mean off-diagonal field-correlation / mean
diagonal field-correlation. If harmonised >> raw here too, the provenance
mechanism generalises off built-up; if not, it is narrower and we say so.

Usage: python -m scripts.eval.lst_provenance_matrix --seeds 20260525 1 2
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from scipy.stats import pearsonr, spearmanr

from scripts.acquire.regions import city_by_name
from scripts.eval.multitask_difficulty import SegUNet

REPO = Path(__file__).resolve().parents[2]
LSTDIR = REPO / "data/raw/lst_aligned"; L8DIR = REPO / "data/raw/landsat8"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 20260525; GRID = 64
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania",
           "weur", "latam", "camcar", "canada", "nordic", "central_europe",
           "china_interior", "south_asia_2", "southern_africa", "mediterranean",
           "brazil_north", "japan_korea_2"]  # 8 original + 12 expansion (n=20)


def l8_to_refl(dn):
    return np.clip(dn.astype(np.float32) * 2.75e-5 - 0.2, 0.0, 1.0)


def load_region(region, source):
    """source in {hist, raw}. Returns X (N,C,64,64), Y (N,1,64,64) LST-2015."""
    X, Y = [], []
    for npz in glob.glob(str(LSTDIR / "*.npz")):
        city = city_by_name(Path(npz).stem)
        if city is None or city.region != region:
            continue
        d = np.load(npz, allow_pickle=True)
        lst = d["patches"]; lst_ids = {t: k for k, t in enumerate(d["tile_ids"])}   # (n,5,64,64)
        if source == "raw":
            l8p = L8DIR / f"{city.name}.npz"
            if not l8p.exists():
                continue
            e = np.load(l8p, allow_pickle=True)
            l8 = e["patches"]; l8_ids = {t: k for k, t in enumerate(e["tile_ids"])}
            common = [t for t in lst_ids if t in l8_ids]
        else:
            common = list(lst_ids)
        for t in common:
            k = lst_ids[t]
            tgt = lst[k, 4][None]                                    # LST 2015 (1,64,64)
            if source == "hist":
                xi = lst[k, :4]                                      # 2011-2014 (4,64,64)
            else:
                raw = l8_to_refl(l8[l8_ids[t]].reshape(18, 224, 224))
                xi = zoom(raw, (1, GRID/224, GRID/224), order=1)[:, :GRID, :GRID].astype(np.float32)
            X.append(xi.astype(np.float32)); Y.append(tgt.astype(np.float32))
    if not X:
        return None, None
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))


_CACHE = {}
def region_xy(region, source):
    key = (region, source)
    if key not in _CACHE:
        _CACHE[key] = load_region(region, source)
    return _CACHE[key]


def train(region, source, epochs=20):
    X, Y = region_xy(region, source)
    if X is None or len(X) < 20:
        return None
    torch.manual_seed(SEED)
    tr = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))[:int(0.85*len(X))].to(DEV)
    X, Y = X.to(DEV), Y.to(DEV)
    m = SegUNet(in_ch=X.shape[1], n=1).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator(device=DEV).manual_seed(SEED)
    for _ in range(epochs):
        m.train(); perm = tr[torch.randperm(len(tr), generator=g, device=DEV)]
        for k in range(0, len(perm), 16):
            b = perm[k:k+16]; opt.zero_grad()
            loss = nn.functional.mse_loss(torch.sigmoid(m(X[b])), Y[b]); loss.backward(); opt.step()
    m.eval(); return m


def field_corr(model, region, source):
    X, Y = region_xy(region, source)
    if X is None:
        return None
    ps, ys = [], []
    with torch.no_grad():
        for k in range(0, len(X), 16):
            p = torch.sigmoid(model(X[k:k+16].to(DEV))).cpu().numpy().reshape(-1)
            ps.append(p); ys.append(Y[k:k+16].numpy().reshape(-1))
    p = np.concatenate(ps); y = np.concatenate(ys)
    if p.std() < 1e-6 or y.std() < 1e-6:
        return 0.0
    return float(pearsonr(p, y)[0])


def run_source(source, seeds):
    global SEED
    ready = [r for r in REGIONS if region_xy(r, source)[0] is not None and len(region_xy(r, source)[0]) >= 20]
    per = []
    for sd in seeds:
        SEED = sd
        models = {r: train(r, source) for r in ready}; models = {r: m for r, m in models.items() if m}
        rr = list(models)
        mat = {s: {t: field_corr(models[s], t, source) for t in rr} for s in models}
        diag = np.mean([mat[r][r] for r in rr if mat[r].get(r) is not None])
        off = np.mean([mat[s][t] for s in rr for t in rr if s != t and mat[s].get(t) is not None])
        srcs = [s for s in rr if all(mat[s].get(t) is not None for t in rr)]
        M = np.array([[mat[s][t] for t in rr] for s in srcs])
        inv = np.mean([spearmanr(M[a], M[b]).correlation for a in range(len(srcs))
                       for b in range(a+1, len(srcs))]) if len(srcs) > 1 else float("nan")
        per.append({"in": float(diag), "out": float(off),
                    "retention": float(off/diag) if diag else None, "src_inv": float(inv)})
        print(f"  [{source} seed {sd}] in-corr={diag:.3f} out-corr={off:.3f} "
              f"retention={off/diag:.3f} src_inv={inv:.3f}", flush=True)
    agg = {k: {"mean": float(np.nanmean([p[k] for p in per])),
               "sd": float(np.nanstd([p[k] for p in per], ddof=1)) if len(per) > 1 else 0.0}
           for k in ("in", "out", "retention", "src_inv")}
    return {"ready": ready, "per_seed": per, "aggregate": agg}


def main(seeds):
    print("=== LST 2015 target; provenance dissociation (non-urban biophysical) ===", flush=True)
    res = {s: run_source(s, seeds) for s in ["hist", "raw"]}
    out = {"target": "MODIS LST 2015 (non-urban climate biophysical)", "seeds": seeds, "by_input": res}
    fn = REPO / "results/metrics/lst_provenance_matrix.json"
    json.dump(out, open(fn, "w"), indent=1)
    h, r = res["hist"]["aggregate"], res["raw"]["aggregate"]
    print(f"\n=== LST provenance dissociation, {len(seeds)} seeds ===")
    print(f"  HARMONISED input (LST history): retention {h['retention']['mean']:.3f}+/-{h['retention']['sd']:.3f}", flush=True)
    print(f"  RAW input (Landsat reflectance): retention {r['retention']['mean']:.3f}+/-{r['retention']['sd']:.3f}", flush=True)
    print(f"  dissociation = {h['retention']['mean']-r['retention']['mean']:+.3f}  "
          f"(built-up was GHSL~1.0 vs Landsat 0.56)", flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seeds", type=int, nargs="+", default=[20260525, 1, 2])
    main(p.parse_args().seeds)
