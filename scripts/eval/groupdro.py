"""Tests whether a domain-generalisation method beats single-source training.
The multi-source domain-adaptation premise
(that WHICH source you train on matters, so machinery to choose/combine sources
helps) fails for this task. That is far stronger if we actually RUN a DG method
and show it does not beat single-source. We use GroupDRO
(Sagawa et al., 2020), the standard worst-group-robust objective, the kind of
method the multi-source literature would reach for.

Leave-one-region-out. For each held-out target region R (of the eight sources):
  * single  : train on one other region alone, evaluate FoM on R (averaged over
              two reference sources; the matrix shows source-invariance, so any
              single source is representative)
  * erm-pool: train on all seven other regions pooled (standard multi-source ERM)
  * groupdro: train on the seven with GroupDRO (upweight the worst region)
If groupdro and erm-pool do not beat single on R, neither multi-source pooling
nor a DG objective buys anything the single source did not -- the premise fails.

Usage: python -m scripts.eval.groupdro --seeds 20260525 1
"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scripts.acquire.regions import CITIES
from scripts.eval.models import SimpleCNN
from scripts.eval.cross_region_train import soft_jaccard
from scripts.eval.cross_region_eval import (
    load_city_rasters, build_input_tensor, fom_metrics,
    EVAL_MASK_THRESH, TARGET_EPOCH, TRAIN_EPOCHS)
from scripts.common import TILE_PX

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
T = 0.01; PREV = TRAIN_EPOCHS[-1]
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]


def gather_region(region, stride=64, cap=64):
    X, GT, PV = [], [], []
    got = 0
    for c in [c for c in CITIES if c.region == region]:
        if got >= cap:
            break
        try:
            bu, vol, pop, _, _ = load_city_rasters(c, PROC)
        except Exception:
            continue
        H, W = bu[TARGET_EPOCH].shape
        for i in range(0, H - TILE_PX + 1, stride):
            for j in range(0, W - TILE_PX + 1, stride):
                gt = bu[TARGET_EPOCH][i:i+TILE_PX, j:j+TILE_PX]
                if gt.mean() < 0.005:
                    continue
                X.append(build_input_tensor(bu, vol, pop, i, j)[0])
                GT.append(gt.astype(np.float32))
                PV.append(bu[PREV][i:i+TILE_PX, j:j+TILE_PX].astype(np.float32))
                got += 1
    if not X:
        return None
    return (torch.stack(X), np.stack(GT), np.stack(PV))


def loss_fn(p, y):
    return nn.functional.mse_loss(p, y) + 0.5 * soft_jaccard(p, y)


def train(groups, mode, seed, epochs=12, pg_bs=8, dro_eta=0.01):
    """groups: list of (X, Y) per source region (on DEV). mode in {erm, groupdro}.
    One concatenated forward per step (per-group slices stacked) -> ~Ngroups x
    faster than a forward per group."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = SimpleCNN(input_channels=24).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    q = torch.ones(len(groups), device=DEV) / len(groups)         # GroupDRO weights
    g = torch.Generator(device=DEV).manual_seed(seed)
    perms = [torch.randperm(len(X), generator=g, device=DEV) for X, _ in groups]
    steps = max(1, max(len(X) for X, _ in groups) // pg_bs)
    for ep in range(epochs):
        m.train()
        for s in range(steps):
            xs, ys, sizes = [], [], []
            for gi, (X, Y) in enumerate(groups):
                start = (s * pg_bs) % len(X)
                idx = perms[gi][start:start + pg_bs]
                xs.append(X[idx]); ys.append(Y[idx]); sizes.append(len(idx))
            Xb, Yb = torch.cat(xs), torch.cat(ys)
            opt.zero_grad(); pred = m(Xb)                        # ONE forward
            gl, off = [], 0
            for sz in sizes:
                gl.append(loss_fn(pred[off:off+sz], Yb[off:off+sz])); off += sz
            gl = torch.stack(gl)
            if mode == "erm":
                loss = gl.mean()
            else:  # groupdro: upweight the worst region
                with torch.no_grad():
                    q = q * torch.exp(dro_eta * gl.detach()); q = q / q.sum()
                loss = (q * gl).sum()
            loss.backward(); opt.step()
    m.eval(); return m


def fom_eval(model, target, bs=32):
    """Returns both FoM (change, confounded) and extent-IoU (state, where pooling
    helps +0.16) so the GroupDRO comparison is metric-complete."""
    d = gather_region(target, stride=TILE_PX)        # non-overlap eval grid
    if d is None:
        return None
    X, GT, PV = d
    preds = []
    with torch.no_grad():
        for b0 in range(0, len(X), bs):
            preds.append(model(X[b0:b0+bs].to(DEV)).cpu().numpy()[:, 0])
    pred = np.concatenate(preds)
    B = A = C = 0; inter = union = 0
    for k in range(len(GT)):
        gt = GT[k]; mask = gt > EVAL_MASK_THRESH
        if mask.sum():
            fm = fom_metrics(gt, PV[k], pred[k], mask, T)
            B += fm["B"]; A += fm["A"]; C += fm["C"]
        pb = pred[k] > T; gb = gt > T
        inter += int((pb & gb).sum()); union += int((pb | gb).sum())
    return {"fom": (B / (A + B + C) if (A + B + C) else 0.0),
            "iou": (inter / union if union else 0.0)}


def main(seeds):
    cache = {r: gather_region(r) for r in REGIONS}
    cache = {r: d for r, d in cache.items() if d is not None}
    regions = list(cache)
    print(f"regions: {regions}  tiles={[len(cache[r][0]) for r in regions]}", flush=True)
    MODES = ("single", "erm_pool", "groupdro"); METS = ("fom", "iou")
    per_seed = []
    for sd in seeds:
        rows = {}
        for R in regions:
            others = [r for r in regions if r != R]
            grp = [(cache[r][0].to(DEV),
                    torch.from_numpy(cache[r][1])[:, None].to(DEV)) for r in others]
            erm = train(grp, "erm", sd); dro = train(grp, "groupdro", sd)
            sm = train([(cache[others[0]][0].to(DEV),
                         torch.from_numpy(cache[others[0]][1])[:, None].to(DEV))], "erm", sd)
            rows[R] = {"single": fom_eval(sm, R), "erm_pool": fom_eval(erm, R),
                       "groupdro": fom_eval(dro, R)}
            del grp
            if DEV.type == "mps": torch.mps.empty_cache()
        mean = {f"{mode}_{met}": float(np.mean([rows[R][mode][met] for R in rows]))
                for mode in MODES for met in METS}
        per_seed.append({"mean": mean, "per_region": rows})
        print(f"[seed {sd}] FoM single={mean['single_fom']:.3f} erm={mean['erm_pool_fom']:.3f} "
              f"dro={mean['groupdro_fom']:.3f} | IoU single={mean['single_iou']:.3f} "
              f"erm={mean['erm_pool_iou']:.3f} dro={mean['groupdro_iou']:.3f}", flush=True)
    agg = {k: {"mean": float(np.mean([s["mean"][k] for s in per_seed])),
               "sd": float(np.std([s["mean"][k] for s in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0}
           for k in (f"{mode}_{met}" for mode in MODES for met in METS)}
    out = {"seeds": seeds, "aggregate": agg, "per_seed": per_seed}
    fn = REPO / "results/metrics/groupdro.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"\n=== Leave-one-region-out DG baselines, {len(seeds)} seeds ===")
    for met in METS:
        s, e, d = agg[f"single_{met}"]["mean"], agg[f"erm_pool_{met}"]["mean"], agg[f"groupdro_{met}"]["mean"]
        print(f"  [{met}] single {s:.3f} | ERM-pool {e:.3f} ({e-s:+.3f}) | "
              f"GroupDRO {d:.3f} ({d-s:+.3f})", flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seeds", type=int, nargs="+", default=[20260525, 1])
    main(p.parse_args().seeds)
