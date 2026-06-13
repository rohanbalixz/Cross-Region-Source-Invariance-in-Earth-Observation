"""Mapping-stability: an INDEPENDENT measure of how region-invariant each input's
relationship to built-up is, computed WITHOUT evaluating cross-region transfer
(so it cannot be circular with the CNN retention it is meant to predict).

For each input modality we fit ridge probes input->built-up at 16x16:
  * local R^2  = mean over regions of a per-region 3-fold-CV R^2 (each region its
                 own map);
  * global R^2 = a single pooled-over-all-regions 3-fold-CV R^2 (one map for all).
stability = global / local in [0,1]: ~1 means one global map predicts built-up as
well as per-region maps (the input->target relationship is the same everywhere);
<1 means each region needs its own map (region-specific relationship).

If stability across modalities tracks the CNN cross-region retention, the
retention ordering reflects an intrinsic property of the input->target mapping,
not the CNN -- i.e. the retention "spectrum" is measured, not asserted.

Run:  python -m scripts.eval.mapping_stability
"""
import json, glob
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr

import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.common import TILE_PX, enumerate_tiles_from_grid
from scripts.eval.cross_region_eval import load_city_rasters, TARGET_EPOCH
from scripts.acquire.regions import city_by_name

PROC = REPO / "data/processed"
GRID, CENTER_PX, BU_THRESH, DOWN = 64, 27, 0.1, 16
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania",
           "weur", "latam", "camcar", "canada", "nordic", "central_europe",
           "china_interior", "south_asia_2", "southern_africa", "mediterranean",
           "brazil_north", "japan_korea_2"]
# CNN n=20 retentions (from the dissociation receipts) for the correlation
RET = {"ghsl_history": 1.00, "sar": 0.899, "worldcover": 0.817,
       "ndvi": 0.679, "ndbi": 0.595, "raw_landsat": 0.550}


def target_patch(gt, i, j):
    c0 = TILE_PX // 2 - CENTER_PX // 2
    box = gt[i + c0:i + c0 + CENTER_PX, j + c0:j + c0 + CENTER_PX]
    if box.shape != (CENTER_PX, CENTER_PX):
        return None
    return (zoom(box, GRID / CENTER_PX, order=1) > BU_THRESH).astype(np.float32)[:GRID, :GRID]


def _resize(x, c):
    if x.shape[-1] != GRID:
        x = zoom(x, (1, GRID / x.shape[-2], GRID / x.shape[-1]), order=1)
    return x[:, :GRID, :GRID]


def aligned_loader(subdir, drop_ch=None):
    """SAR / WorldCover / NDVI: pre-aligned (n,C,64,64) npz per city."""
    def load(region):
        out = []
        for npz in glob.glob(str(REPO / "data/raw" / subdir / "*.npz")):
            city = city_by_name(Path(npz).stem)
            if city is None or city.region != region:
                continue
            d = np.load(npz, allow_pickle=True)
            P, tids = d["patches"], list(d["tile_ids"])
            try:
                bu, *_, transform, crs = load_city_rasters(city, PROC)
            except Exception:
                continue
            gt = bu[TARGET_EPOCH]
            id2ij = {r.tile_id: (r.i, r.j) for r in enumerate_tiles_from_grid(
                builtup_2015=gt, utm_transform=transform, city_name=city.name,
                region=region, utm_crs=crs)}
            for k, tid in enumerate(tids):
                if tid not in id2ij:
                    continue
                y = target_patch(gt, *id2ij[tid])
                if y is None:
                    continue
                x = _resize(P[k].astype(np.float32), P.shape[1])
                if drop_ch is not None:
                    x = np.delete(x, drop_ch, axis=0)
                out.append((x, y))
        return out
    return load


def ghsl_loader(region):
    """GHSL built-up history (2005, 2010) -> built-up 2015, the temporal input."""
    out = []
    for cd in glob.glob(str(PROC / region / "*")):
        city = city_by_name(Path(cd).name)
        if city is None:
            continue
        try:
            bu, *_, transform, crs = load_city_rasters(city, PROC)
        except Exception:
            continue
        gt = bu[TARGET_EPOCH]
        for r in enumerate_tiles_from_grid(builtup_2015=gt, utm_transform=transform,
                                           city_name=city.name, region=region, utm_crs=crs):
            i, j = r.i, r.j
            y = target_patch(gt, i, j)
            if y is None:
                continue
            chans = []
            ok = True
            for e in (2005, 2010):
                c0 = TILE_PX // 2 - CENTER_PX // 2
                b = bu[e][i + c0:i + c0 + CENTER_PX, j + c0:j + c0 + CENTER_PX]
                if b.shape != (CENTER_PX, CENTER_PX):
                    ok = False; break
                chans.append(zoom(b, GRID / CENTER_PX, order=1)[:GRID, :GRID])
            if ok:
                out.append((np.stack(chans).astype(np.float32), y))
    return out


def landsat_loader(region, ndbi=False):
    """raw multi-decade Landsat last epoch (6 bands) -> built-up; or NDBI index."""
    out = []
    for npz in glob.glob(str(REPO / "data/raw/landsat_history/*.npz")):
        city = city_by_name(Path(npz).stem)
        if city is None or city.region != region:
            continue
        d = np.load(npz, allow_pickle=True)
        P = d["patches"].astype(np.float32)          # (n,epochs,6,224,224)
        tids = list(d["tile_ids"])
        try:
            bu, *_, transform, crs = load_city_rasters(city, PROC)
        except Exception:
            continue
        gt = bu[TARGET_EPOCH]
        id2ij = {r.tile_id: (r.i, r.j) for r in enumerate_tiles_from_grid(
            builtup_2015=gt, utm_transform=transform, city_name=city.name,
            region=region, utm_crs=crs)}
        for k, tid in enumerate(tids):
            if tid not in id2ij:
                continue
            y = target_patch(gt, *id2ij[tid])
            if y is None:
                continue
            refl = np.clip(P[k, -1] * 2.75e-5 - 0.2, 0, 1)   # 6 bands, last epoch
            if ndbi:
                swir1, nir = refl[4], refl[3]
                x = ((swir1 - nir) / (swir1 + nir + 1e-6))[None]
            else:
                x = refl
            out.append((_resize(x, x.shape[0]), y))
    return out


_RNG = np.random.default_rng(0)


def featurize(pairs, per_tile=140):
    """Pixel-level: features = a 3x3 window of every input channel around each
    pixel (captures local texture a linear probe can use); target = built-up 0/1.
    A *nonlinear* mapping is fine -- we use gradient boosting -- the point is the
    global-vs-local comparison, not the probe class."""
    Xs, Ys = [], []
    for x, y in pairs:
        C, H, W = x.shape
        # 3x3 mean + the centre value per channel -> 2C features, cheap & robust
        pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
        ctx = sum(pad[:, a:a + H, b:b + W] for a in range(3) for b in range(3)) / 9.0
        feat = np.concatenate([x, ctx], 0).reshape(2 * C, -1).T          # (H*W, 2C)
        tgt = (y.reshape(-1) > 0.5).astype(int)
        idx = _RNG.choice(len(tgt), size=min(per_tile, len(tgt)), replace=False)
        Xs.append(feat[idx]); Ys.append(tgt[idx])
    return np.concatenate(Xs), np.concatenate(Ys)


def cv_auc(X, Y, k=3):
    if Y.sum() < 10 or (1 - Y).sum() < 10 or len(X) < 200:
        return np.nan
    sc = []
    for tr, te in KFold(k, shuffle=True, random_state=0).split(X):
        if Y[tr].sum() < 5 or Y[te].sum() < 5:
            continue
        m = HistGradientBoostingClassifier(max_iter=120, max_depth=3,
                                           learning_rate=0.1, random_state=0).fit(X[tr], Y[tr])
        sc.append(roc_auc_score(Y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(sc)) if sc else np.nan


def stability(loader, name):
    perreg = {}
    for reg in REGIONS:
        pairs = loader(reg)
        if len(pairs) < 12:
            continue
        perreg[reg] = featurize(pairs)
    if not perreg:
        return None
    local = [cv_auc(X, Y) for X, Y in perreg.values()]
    GX = np.concatenate([X for X, _ in perreg.values()])
    GY = np.concatenate([Y for _, Y in perreg.values()])
    glob = cv_auc(GX, GY)
    local_m = float(np.nanmean(local))
    # stability = how much of the per-region predictive SKILL a single global map keeps
    s = float(np.clip((glob - 0.5) / (local_m - 0.5), 0, 1)) if local_m > 0.52 else np.nan
    print(f"  {name:14s} local AUC={local_m:.3f}  global AUC={glob:.3f}  stability={s:.3f}  "
          f"(n_reg={len(perreg)})", flush=True)
    return dict(local_auc=round(local_m, 4), global_auc=round(glob, 4),
                stability=round(s, 4), n_regions=len(perreg))


def main():
    print("computing mapping stability (independent of CNN transfer)...", flush=True)
    loaders = {
        "ghsl_history": ghsl_loader,
        "sar": aligned_loader("sentinel1"),
        "worldcover": aligned_loader("worldcover_aligned", drop_ch=2),
        "ndvi": aligned_loader("ndvi_aligned"),
        "ndbi": lambda r: landsat_loader(r, ndbi=True),
        "raw_landsat": lambda r: landsat_loader(r, ndbi=False),
    }
    res = {}
    for name, ld in loaders.items():
        r = stability(ld, name)
        if r:
            r["cnn_retention"] = RET[name]
            res[name] = r
    names = [n for n in res if not np.isnan(res[n]["stability"])]
    S = [res[n]["stability"] for n in names]
    R = [res[n]["cnn_retention"] for n in names]
    rho, p = spearmanr(S, R)
    pr, pp = pearsonr(S, R)
    out = {"per_input": res, "n_inputs": len(names),
           "spearman_stability_vs_retention": round(float(rho), 3),
           "spearman_p": float(p),
           "pearson_stability_vs_retention": round(float(pr), 3)}
    json.dump(out, open(REPO / "results/metrics/mapping_stability.json", "w"), indent=2)
    print(f"\nstability vs CNN retention: Spearman {rho:+.3f} (p={p:.3f}), Pearson {pr:+.3f}")
    print("wrote results/metrics/mapping_stability.json")


if __name__ == "__main__":
    main()
