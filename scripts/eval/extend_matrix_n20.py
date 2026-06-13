"""Extend the full source-by-target transfer matrix from 12 to 20 regions using
the newly-acquired GHSL data for the eight new regions (nordic, central_europe,
china_interior, south_asia_2, southern_africa, mediterranean, brazil_north,
japan_korea_2) -- real data, real inference, no proxy.

Difficulty is defined exactly as in this study: the mean FoM a region receives
across every source model (the transfer-matrix column mean). This script:

  1. ensures every new-region city has an `eval_metrics.json` marker by running
     the CONUS single-source eval on any that lack it (also gives the raw
     CONUS-difficulty for the new regions);
  2. trains a from-scratch source model for each of the 12 non-original regions
     that lacks one (same recipe as the n=8/n=12 matrices);
  3. evaluates all 20 source models (+CONUS) on all 20 target regions;
  4. reports the n=20 difficulty-vs-change-rate Spearman with a region-level
     bootstrap CI, alongside the n=8 ([0.29,1.00]) and n=12 values.

Whatever the data gives is what gets reported -- including if the CI fails to
tighten or the developed-region-is-hard pattern weakens.

Usage: python -m scripts.eval.extend_matrix_n20 --arch cnn
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from scripts.acquire.regions import city_by_name
from scripts.eval import cross_region_train as crt
from scripts.eval.cross_region_eval import (
    TARGET_EPOCH,
    TRAIN_EPOCHS,
    load_city_rasters,
    load_models,
    process_city,
)
from scripts.eval.transfer_matrix import ARCH, CONUS_CK, load_model, region_fom

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SRC8 = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]
NEW4 = ["weur", "latam", "camcar", "canada"]
NEW8 = ["nordic", "central_europe", "china_interior", "south_asia_2",
        "southern_africa", "mediterranean", "brazil_north", "japan_korea_2"]
SRC20 = SRC8 + NEW4 + NEW8
TRAINABLE = NEW4 + NEW8            # SRC8 sources already trained for the n=8 matrix


def change_rate(region):
    """Mean over the region's cities of the fraction of built-up pixels that
    changed (>0.01) from 2010 to 2015 -- identical to the n=12 definition."""
    fracs = []
    for f in glob.glob(str(PROC / f"{region}/*/builtup_2015.tif")):
        city = city_by_name(Path(f).parent.name)
        try:
            bu, _, _, _, _ = load_city_rasters(city, PROC)
        except Exception:
            continue
        gt, prev = bu[TARGET_EPOCH], bu[TRAIN_EPOCHS[-1]]
        mask = gt > 0.01
        if mask.sum():
            fracs.append(float(((gt - prev) > 0.01)[mask].mean()))
    return float(np.mean(fracs)) if fracs else None


def ensure_eval_markers(regions, conus_models):
    """region_fom() only scores cities that already carry an eval_metrics.json.
    For the freshly-preprocessed regions, create that marker by running the
    CONUS baselines once (the canonical single-source eval)."""
    for region in regions:
        for f in sorted(glob.glob(str(PROC / f"{region}/*/builtup_2015.tif"))):
            city = city_by_name(Path(f).parent.name)
            out = PROC / region / city.name / "eval_metrics.json"
            if out.exists():
                continue
            try:
                process_city(city, PROC, conus_models, out, DEV)
                print(f"  [eval-marker] {region}/{city.name}", flush=True)
            except Exception as e:
                print(f"  [skip-marker] {region}/{city.name}: {type(e).__name__}: {e}", flush=True)


def main(arch):
    # 1. CONUS single-source eval on the new regions -> eval_metrics.json markers
    conus = load_models((REPO.parent / "models").resolve())
    ensure_eval_markers(NEW8, conus)

    # 2. train any missing source models (NEW4 + NEW8)
    for s in TRAINABLE:
        ck = REPO / f"results/transfer_matrix/weights/{s}/{crt.CKPT[arch]}"
        if not ck.exists():
            print(f"[train] {s}/{arch}", flush=True); crt.main(s, arch, 25, 20260525, DEV)

    # 3. full 20x20 (+conus) matrix
    sources = {s: REPO / f"results/transfer_matrix/weights/{s}/{crt.CKPT[arch]}" for s in SRC20}
    sources["conus"] = REPO.parent / f"models/{CONUS_CK[arch]}"
    mat = {}
    for src, ck in sources.items():
        if not Path(ck).exists():
            print(f"[no ckpt] {src}"); continue
        m = load_model(arch, ck)
        mat[src] = {t: region_fom(arch, m, t) for t in SRC20}
        print(f"[{src}] " + " ".join(f"{t}={mat[src][t]:.2f}" for t in SRC20
                                     if mat[src].get(t) is not None), flush=True)

    # 4. statistics -- difficulty = column mean (mean FoM a region receives)
    diag = [mat[r][r] for r in SRC20 if mat.get(r, {}).get(r) is not None]
    off = [mat[s][t] for s in SRC20 for t in SRC20 if s != t and mat.get(s, {}).get(t) is not None]
    srcs = [s for s in mat if all(mat[s].get(t) is not None for t in SRC20)]
    inv = None
    if len(srcs) > 1:
        M = np.array([[mat[s][t] for t in SRC20] for s in srcs])
        inv = float(np.mean([spearmanr(M[a], M[b]).correlation
                             for a in range(len(srcs)) for b in range(a + 1, len(srcs))]))
    difficulty = {t: float(np.mean([mat[s][t] for s in mat if mat[s].get(t) is not None]))
                  for t in SRC20}
    cr = {r: change_rate(r) for r in SRC20}
    common = [r for r in SRC20 if cr[r] is not None and r in difficulty]
    rho, _ = spearmanr([difficulty[r] for r in common], [cr[r] for r in common])

    rng = np.random.default_rng(20260525)
    dv = np.array([difficulty[r] for r in common]); cv = np.array([cr[r] for r in common])
    boots = []
    for _ in range(20000):
        idx = rng.integers(0, len(common), len(common))
        if len(set(idx.tolist())) > 2:
            boots.append(spearmanr(dv[idx], cv[idx]).correlation)
    ci = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]

    res = {"arch": arch, "n_regions": len(common), "regions": common,
           "in_region": float(np.mean(diag)) if diag else None,
           "out_region": float(np.mean(off)) if off else None,
           "home_field_gap": (float(np.mean(diag) - np.mean(off)) if diag and off else None),
           "source_invariance": inv,
           "changerate_spearman": float(rho), "changerate_spearman_ci95": ci,
           "difficulty": difficulty, "change_rate": cr, "matrix": mat}
    print(f"\n=== FULL MATRIX n={len(common)} ({arch}) ===")
    if diag and off:
        print(f"  in-region {res['in_region']:.3f} vs out-of-region {res['out_region']:.3f}  "
              f"(home-field gap {res['home_field_gap']:+.4f})")
    if inv is not None:
        print(f"  source-invariance (row rho) = {inv:.3f}")
    print(f"  difficulty vs change-rate: Spearman {rho:.2f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    print(f"    (n=8 was 0.83 [0.29,1.00];  n=12 see full_matrix_n12_{arch}.json)")
    out = REPO / f"results/metrics/full_matrix_n20_{arch}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--arch", default="cnn", choices=list(ARCH))
    main(p.parse_args().arch)
