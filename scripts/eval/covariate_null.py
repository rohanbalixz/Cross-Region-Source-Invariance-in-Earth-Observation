"""Covariate null with a reproducible receipt.

Two-sided result, computed from the per-tile JSONs the rest of the pipeline
writes (eval_metrics.json + terrain/settlement/transport.json, joined by
tile_id):

  WITHIN the source region (CONUS): a gradient-boosted regressor of per-tile
  FoM on the 12 covariates cross-validates at R^2 ~ 0.57 (CNN) -- covariates
  DO predict the score, but only as a restatement of the change confound
  (covariates predict built-up STATE at R^2~0.91 and CHANGE at R^2~0.49, and
  FoM tracks change at r~0.86).

  ACROSS regions: a covariate->FoM model fit on N-1 target regions predicts the
  held-out region's per-tile FoM WORSE than its mean (leave-one-region-out
  R^2 < 0 in every fold). The physical covariates give no transportable account
  of where the model performs well.

This replaces an earlier ad-hoc number (R^2=0.047) that did not reproduce.

Run:  python -m scripts.eval.covariate_null
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data/processed"
TER = ["slope_mean", "slope_var", "tri", "elev_mean"]
SET = ["builtup_frac", "fragmentation", "compactness", "contiguity"]
TRA = ["road_density", "intersection_density", "grid_entropy", "osm_completeness"]
COV = TER + SET + TRA
FOM_T = "t=0.01"
MODELS = ["cnn", "unet", "convlstm"]
REG8 = ["andes", "ssa", "south_asia", "east_asia", "mena", "sea", "eeca", "oceania"]


def load_city(city_dir):
    try:
        ev = json.load(open(city_dir / "eval_metrics.json"))
        ter = {r["tile_id"]: r for r in json.load(open(city_dir / "terrain.json"))}
        se = {r["tile_id"]: r for r in json.load(open(city_dir / "settlement.json"))}
        tr = {r["tile_id"]: r for r in json.load(open(city_dir / "transport.json"))}
    except FileNotFoundError:
        return []
    rows = []
    for r in ev:
        t = r["tile_id"]
        if t not in ter or t not in se or t not in tr:
            continue
        z = [ter[t][k] for k in TER] + [se[t][k] for k in SET] + [tr[t][k] for k in TRA]
        if any(v is None or not np.isfinite(v) for v in z):
            continue
        m = r.get("models", {})
        fom = {mo: (m.get(mo, {}).get("fom", {}).get(FOM_T, {}) or {}).get("fom")
               for mo in MODELS}
        rows.append(dict(region=r.get("region"), z=z, fom=fom,
                         state=r["gt_mean"], change=r["gt_mean"] - r["prev_mean"]))
    return rows


def gather(region):
    out = []
    for cd in sorted(glob.glob(str(PROC / region / "*"))):
        if os.path.isdir(cd):
            out += load_city(Path(cd))
    return out


def cv_r2(Z, y):
    Z, y = np.asarray(Z, float), np.asarray(y, float)
    ok = np.isfinite(y)
    return float(cross_val_score(GradientBoostingRegressor(random_state=0), Z[ok], y[ok],
                                 cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean())


def main():
    conus = gather("conus")
    n_total = len(json.load(open(PROC / "conus/conus/eval_metrics.json")))
    Zc = [r["z"] for r in conus]
    rep = {"FOM_T": FOM_T, "n_covariates": len(COV),
           "within_source": {"region": "conus", "n_tiles_total": n_total,
                             "n_tiles_used": len(conus), "n_dropped": n_total - len(conus)}}
    # within-source: covariates -> FoM (per model), and -> state / change
    for mo in MODELS:
        rep["within_source"][f"R2_fom_{mo}"] = round(cv_r2(Zc, [r["fom"][mo] for r in conus]), 3)
    rep["within_source"]["R2_state"] = round(cv_r2(Zc, [r["state"] for r in conus]), 3)
    rep["within_source"]["R2_change"] = round(cv_r2(Zc, [r["change"] for r in conus]), 3)
    fom_cnn = np.array([r["fom"]["cnn"] for r in conus], float)
    chg = np.array([r["change"] for r in conus], float)
    rep["within_source"]["corr_fom_change"] = round(float(np.corrcoef(fom_cnn, chg)[0, 1]), 3)

    # cross-region: leave-one-region-out covariate -> FoM (CNN)
    rows = []
    for reg in REG8:
        rows += gather(reg)
    ro = np.array([r["region"] for r in rows])
    Z = np.array([r["z"] for r in rows], float)
    y = np.array([r["fom"]["cnn"] for r in rows], float)
    loro = {}
    for held in REG8:
        trm, te = ro != held, ro == held
        if te.sum() < 20:
            continue
        g = GradientBoostingRegressor(random_state=0).fit(Z[trm], y[trm])
        loro[held] = round(float(r2_score(y[te], g.predict(Z[te]))), 3)
    rep["cross_region_loro"] = {"per_region_R2": loro,
                                "median_R2": round(float(np.median(list(loro.values()))), 3),
                                "n_folds_R2_positive": int(sum(v > 0 for v in loro.values())),
                                "n_folds": len(loro), "n_tiles": len(rows)}
    out = REPO / "results/metrics/covariate_null.json"
    json.dump(rep, open(out, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
