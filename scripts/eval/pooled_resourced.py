"""Resourced/pooled deflation control. The deflation claim---a parameter-free linear
extrapolation beats the trained CNN on FoM (0.56 vs 0.34) -- was made with small
per-region models (~200 tiles each) that the capacity sweep shows overfit at 4x
width. This isolates whether the deflation reflects a trivial task rather than starved models.

This trains a genuinely well-resourced temporal model -- POOLED over all eight
regions (~8x the data, so no longer starved) at increasing capacity (1x..8x
width) -- and compares it to the linear-extrapolation baseline on three metrics:

  * FoM @0.01            -- the study's metric (rewards getting change QUANTITY right;
                           linext nails quantity by construction -> confounded)
  * change-AUC           -- CONFOUND-FREE: ROC-AUC of the predicted change score
                           against the actual-change label inside the eval mask;
                           rank-based, so invariant to quantity/threshold -> pure
                           spatial ALLOCATION skill
  * extent-IoU @0.01     -- IoU of the 2015 built-up state (persistence-dominated)

If even the pooled, higher-capacity model does not beat linext on FoM, the
triviality reading is not a starved-model artefact. If the model beats linext on
change-AUC (allocation) while losing on FoM (quantity), that is the honest,
sharper statement: the model is not useless, but FoM rewards the baseline's
quantity-matching -- which is the study's own confound, now turned on itself.

Usage: python -m scripts.eval.pooled_resourced --widths 1 2 4 8 --epochs 30
"""
import argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from scripts.acquire.regions import CITIES
from scripts.eval.capacity_sweep import WidthCNN
from scripts.eval.cross_region_train import soft_jaccard
from scripts.eval.cross_region_eval import (
    load_city_rasters, build_input_tensor, fom_metrics,
    EVAL_MASK_THRESH, TARGET_EPOCH, TRAIN_EPOCHS)
from scripts.common import TILE_PX

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
T = 0.01
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]
PREV, PREV2 = TRAIN_EPOCHS[-1], TRAIN_EPOCHS[-2]   # 2010, 2005


def gather(regions, stride=64, cap_per_region=140):
    """Pooled (input, gt2015, prev2010, prev2005, region) tiles across regions."""
    X, GT, PV, PV2, REG = [], [], [], [], []
    for region in regions:
        got = 0
        for c in [c for c in CITIES if c.region == region]:
            if got >= cap_per_region:
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
                    PV2.append(bu[PREV2][i:i+TILE_PX, j:j+TILE_PX].astype(np.float32))
                    REG.append(region); got += 1
    X = torch.stack(X)                              # (n,8,3,H,W) on CPU
    return X, np.stack(GT), np.stack(PV), np.stack(PV2), np.array(REG)


def train_pooled(Xd, Yd, w, epochs, seed=20260525, bs=16):
    """Xd, Yd are already resident on DEV (no per-batch host->device copy)."""
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xd); idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr = idx[:int(0.85*n)].to(DEV)
    m = WidthCNN(w).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for ep in range(epochs):
        m.train(); perm = tr[torch.randperm(len(tr), generator=g, device=DEV)]
        for b0 in range(0, len(perm), bs):
            b = perm[b0:b0+bs]
            opt.zero_grad(); p = m(Xd[b])
            loss = nn.functional.mse_loss(p, Yd[b]) + 0.5*soft_jaccard(p, Yd[b])
            loss.backward(); opt.step()
    m.eval(); return m


def predict(model, Xd, bs=32):
    out = []
    with torch.no_grad():
        for b0 in range(0, len(Xd), bs):
            out.append(model(Xd[b0:b0+bs]).cpu().numpy()[:, 0])
    return np.concatenate(out)


def metrics(pred, GT, PV, REG):
    """Per-region FoM, change-AUC, extent-IoU pooled over that region's tiles."""
    res = {}
    for region in REGIONS:
        sel = np.where(REG == region)[0]
        if len(sel) == 0:
            continue
        B = A = C = 0; inter = union = 0
        sc, lab = [], []
        for k in sel:
            gt, prev, pr = GT[k], PV[k], pred[k]
            mask = gt > EVAL_MASK_THRESH
            if mask.sum():
                fm = fom_metrics(gt, prev, pr, mask, T)
                B += fm["B"]; A += fm["A"]; C += fm["C"]
                sc.append((pr - prev)[mask]); lab.append(((gt - prev) > T)[mask])
            pb = pr > T; gb = gt > T
            inter += (pb & gb).sum(); union += (pb | gb).sum()
        fom = B/(A+B+C) if (A+B+C) else 0.0
        iou = inter/union if union else 0.0
        sc = np.concatenate(sc); lab = np.concatenate(lab).astype(int)
        auc = float(roc_auc_score(lab, sc)) if 0 < lab.sum() < len(lab) else float("nan")
        res[region] = {"fom": float(fom), "change_auc": auc, "extent_iou": float(iou)}
    return res


def linext_metrics(GT, PV, PV2, REG):
    pred = PV + (PV - PV2)                            # 2015 = 2010 + (2010-2005)
    return metrics(pred.clip(0, None), GT, PV, REG)


def summarise(per):
    return {k: float(np.nanmean([per[r][k] for r in per])) for k in ("fom", "change_auc", "extent_iou")}


def main(widths, epochs, seeds):
    X, GT, PV, PV2, REG = gather(REGIONS)
    print(f"pooled tiles: {len(X)}  (regions={[int((REG==r).sum()) for r in REGIONS]})", flush=True)
    lin = linext_metrics(GT, PV, PV2, REG); lin_s = summarise(lin)
    print(f"[linext baseline]  FoM={lin_s['fom']:.3f}  change-AUC={lin_s['change_auc']:.3f}  "
          f"extent-IoU={lin_s['extent_iou']:.3f}", flush=True)
    Xd = X.to(DEV); Yd = torch.from_numpy(GT)[:, None].float().to(DEV)   # resident on GPU
    out = {"linext": {"per_region": lin, "overall": lin_s}, "model": {}}
    for w in widths:
        runs = []
        for sd in seeds:
            m = train_pooled(Xd, Yd, w, epochs, seed=sd)
            per = metrics(predict(m, Xd), GT, PV, REG); runs.append(summarise(per))
            del m
            if DEV.type == "mps": torch.mps.empty_cache()
        agg = {k: {"mean": float(np.mean([r[k] for r in runs])),
                   "sd": float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0}
               for k in ("fom", "change_auc", "extent_iou")}
        params = sum(p.numel() for p in WidthCNN(w).parameters())
        out["model"][f"{w}x"] = {"params": params, "seeds": seeds, "aggregate": agg,
                                 "per_region_lastseed": per}
        print(f"[pooled {w}x  {params/1e3:.0f}k params]  "
              f"FoM={agg['fom']['mean']:.3f}±{agg['fom']['sd']:.3f}  "
              f"change-AUC={agg['change_auc']['mean']:.3f}±{agg['change_auc']['sd']:.3f}  "
              f"extent-IoU={agg['extent_iou']['mean']:.3f}±{agg['extent_iou']['sd']:.3f}", flush=True)
    fn = REPO / "results/metrics/pooled_resourced.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"\nlinext FoM {lin_s['fom']:.3f} vs best pooled model FoM "
          f"{max(out['model'][w]['aggregate']['fom']['mean'] for w in out['model']):.3f}", flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--widths", type=float, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260525, 1, 2])
    a = p.parse_args()
    main(a.widths, a.epochs, a.seeds)
