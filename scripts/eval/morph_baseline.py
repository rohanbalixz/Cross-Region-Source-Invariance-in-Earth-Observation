"""Deflation test: is the temporal CNN's
transferable change-skill ABOVE a parameter-free morphological heuristic, or does
it merely inherit the trivial universality of "built-up grows at the edges of
existing built-up"?

We score three NON-LEARNED baselines on the built-up FoM, identically to the CNN
(same tiles, same fom_metrics at t=0.01):
  - persistence:        pred_2015 = built-up_2010              (FoM ~ 0 by construction)
  - linear extrapolation: pred = 2010 + (2010 - 2005)         (recent growth continues)
  - morphological dilation: predict change in the dilation band of the 2010
    footprint; the existing-footprint threshold and dilation radius are SWEPT and
    the best per-region FoM is kept -- a steelman of the trivial heuristic.

Then per region we compare the best-trivial FoM to the CNN's in-region (matrix
diagonal) and out-of-region (off-diagonal mean) FoM, and report the CNN's margin
over the heuristic and the margin's retention (out-margin / in-margin). Large,
source-invariant margin => the CNN learned non-trivial transferable skill (the
mechanism claim is earned); margin ~ 0 => source-inertness is trivial morphology.

Usage: python -m scripts.eval.morph_baseline
"""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from scripts.acquire.regions import CITIES
from scripts.eval.cross_region_eval import (
    load_city_rasters, fom_metrics, EVAL_MASK_THRESH, TARGET_EPOCH, TRAIN_EPOCHS)
from scripts.common import TILE_PX, enumerate_tiles_from_grid

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"; T = 0.01
SRC = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]
MORPH_GRID = [(r, thr) for r in (1, 2, 3) for thr in (0.01, 0.05, 0.1)]


def tile_fom(pred, gt, prev):
    mask = gt > EVAL_MASK_THRESH
    if mask.sum() == 0:
        return None
    return fom_metrics(gt, prev, pred, mask, T)


def morph_pred(prev, radius, thr):
    seed = prev > thr
    grown = ndimage.binary_dilation(seed, iterations=radius)
    ring = grown & ~seed
    return prev + ring.astype(np.float32) * (2 * T + 1e-3)   # change > T on the ring only


def region_baselines(region):
    accs = {}   # name -> [B, A, C] pooled over all tiles/cities

    def add(name, fm):
        if fm is None:
            return
        a = accs.setdefault(name, [0, 0, 0]); a[0] += fm["B"]; a[1] += fm["A"]; a[2] += fm["C"]

    n_cities = 0
    for c in [c for c in CITIES if c.region == region]:
        if not (PROC / region / c.name / "eval_metrics.json").exists():
            continue
        try:
            bu, vol, pop, transform, crs = load_city_rasters(c, PROC)
        except Exception:
            continue
        n_cities += 1
        gt = bu[TARGET_EPOCH]; prev = bu[TRAIN_EPOCHS[-1]]; e2005 = bu[2005]
        refs = enumerate_tiles_from_grid(builtup_2015=gt, utm_transform=transform,
                                         city_name=c.name, region=region, utm_crs=crs)
        for ref in refs:
            i, j = ref.i, ref.j
            g = gt[i:i+TILE_PX, j:j+TILE_PX]; p = prev[i:i+TILE_PX, j:j+TILE_PX]
            p5 = e2005[i:i+TILE_PX, j:j+TILE_PX]
            add("persistence", tile_fom(p, g, p))
            add("linext", tile_fom(p + (p - p5), g, p))
            for (r, thr) in MORPH_GRID:
                add(f"morph_r{r}_t{thr}", tile_fom(morph_pred(p, r, thr), g, p))

    def fom_of(a):
        B, A, C = a; d = A + B + C; return B / d if d > 0 else 0.0
    res = {k: fom_of(v) for k, v in accs.items()}
    morphs = {k: v for k, v in res.items() if k.startswith("morph")}
    best_morph_k = max(morphs, key=morphs.get)
    best_trivial = max(res.get("persistence", 0.0), res.get("linext", 0.0), morphs[best_morph_k])
    return {"n_cities": n_cities, "persistence": res.get("persistence", 0.0),
            "linext": res.get("linext", 0.0), "best_morph": morphs[best_morph_k],
            "best_morph_cfg": best_morph_k, "best_trivial": best_trivial}


def main():
    cnn = json.load(open(REPO / "results/metrics/transfer_matrix_cnn.json"))
    out = {}
    for region in SRC:
        b = region_baselines(region)
        cnn_in = cnn[region][region]
        cnn_out = float(np.mean([cnn[s][region] for s in SRC
                                 if s != region and cnn.get(s, {}).get(region) is not None]))
        bt = b["best_trivial"]
        out[region] = {**b, "cnn_in": cnn_in, "cnn_out": cnn_out,
                       "margin_in": cnn_in - bt, "margin_out": cnn_out - bt}
        print(f"{region:11} trivial={bt:.3f}[{b['best_morph_cfg']}] "
              f"(persist={b['persistence']:.3f} linext={b['linext']:.3f}) "
              f"cnn_in={cnn_in:.3f} cnn_out={cnn_out:.3f} "
              f"margin_in={cnn_in-bt:+.3f} margin_out={cnn_out-bt:+.3f}", flush=True)
    mi = float(np.mean([out[r]["margin_in"] for r in SRC]))
    mo = float(np.mean([out[r]["margin_out"] for r in SRC]))
    summary = {"best_trivial_mean": float(np.mean([out[r]["best_trivial"] for r in SRC])),
               "cnn_in_mean": float(np.mean([out[r]["cnn_in"] for r in SRC])),
               "cnn_out_mean": float(np.mean([out[r]["cnn_out"] for r in SRC])),
               "margin_in_mean": mi, "margin_out_mean": mo,
               "margin_retention": (mo / mi if mi > 0 else None)}
    print("\n=== CNN vs best trivial heuristic (built-up FoM, 8 regions) ===")
    print(f"best-trivial {summary['best_trivial_mean']:.3f} | "
          f"CNN in {summary['cnn_in_mean']:.3f} out {summary['cnn_out_mean']:.3f}")
    print(f"CNN margin over trivial: in {mi:+.3f}, out {mo:+.3f}, "
          f"margin-retention {summary['margin_retention']}", flush=True)
    json.dump({"per_region": out, "summary": summary},
              open(REPO / "results/metrics/morph_baseline.json", "w"), indent=1)
    print("saved results/metrics/morph_baseline.json", flush=True)


if __name__ == "__main__":
    main()
