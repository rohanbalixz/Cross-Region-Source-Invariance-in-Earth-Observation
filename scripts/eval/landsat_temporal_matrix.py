"""Break the input-provenance confound. The temporal
tasks use GHSL -- a single globally-uniform product -- as input, and are
source-invariant; the imagery tasks use raw Sentinel-2 reflectance, and are
source-dependent. Is the split "temporal vs spectral", or "uniform-product input
vs raw-sensor input"?

Matched-input control: hold the TARGET fixed (built-up 2015) and swap only the
input from GHSL-history to RAW multi-temporal Landsat-8 reflectance (3 seasonal
frames x 6 bands = 18 channels, acquired per tile by scripts.acquire.landsat8).
The L8 patch covers the central 6.72 km of each tile; we resample GHSL built-up
2015 to that extent as the binary target, train a SegUNet per source region, and
read the 8x8 retention. If raw multi-temporal reflectance is source-DEPENDENT
(retention ~0.8, like single-date imagery), the source-invariance is a property
of the uniform GHSL product input, not of having a temporal input -- the deepest
deflation. If source-INVARIANT, temporality genuinely overcomes raw appearance.

Usage: python -m scripts.eval.landsat_temporal_matrix
"""
import glob, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
from scipy.ndimage import zoom
from scripts.acquire.regions import CITIES, city_by_name
from scripts.eval.cross_region_eval import load_city_rasters, TARGET_EPOCH
from scripts.common import TILE_PX, enumerate_tiles_from_grid
from scripts.eval.multitask_difficulty import SegUNet

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
L8DIR = REPO / "data/raw/landsat_history"   # hardened multi-decade; --data landsat8 for the 3-frame version
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED = 20260525
REGIONS = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania",
           "weur", "latam", "camcar", "canada", "nordic", "central_europe",
           "china_interior", "south_asia_2", "southern_africa", "mediterranean",
           "brazil_north", "japan_korea_2"]   # 8 original + 12 expansion (n=20)
GRID = 64                       # model in/out spatial size
CENTER_PX = 27                  # 6.72 km / 250 m ~ 27 GHSL pixels (L8 patch extent)
BU_THRESH = 0.1                 # built-up presence threshold for the binary target


def l8_to_reflectance(dn):
    # Landsat C2 L2 surface reflectance scaling, clipped to [0,1]
    return np.clip(dn.astype(np.float32) * 2.75e-5 - 0.2, 0.0, 1.0)


def load_region(region):
    X, Y = [], []
    for npz_path in glob.glob(str(L8DIR / "*.npz")):
        city = city_by_name(Path(npz_path).stem)
        if city is None or city.region != region:
            continue
        d = np.load(npz_path, allow_pickle=True)
        patches = d["patches"]                       # (n,3,6,224,224) uint16
        tile_ids = list(d["tile_ids"])
        try:
            bu, _, _, transform, crs = load_city_rasters(city, PROC)
        except Exception:
            continue
        gt = bu[TARGET_EPOCH]
        refs = enumerate_tiles_from_grid(builtup_2015=gt, utm_transform=transform,
                                         city_name=city.name, region=region, utm_crs=crs)
        id2ij = {r.tile_id: (r.i, r.j) for r in refs}
        c0 = TILE_PX // 2 - CENTER_PX // 2
        for k, tid in enumerate(tile_ids):
            if tid not in id2ij:
                continue
            i, j = id2ij[tid]
            box = gt[i + c0:i + c0 + CENTER_PX, j + c0:j + c0 + CENTER_PX]
            if box.shape != (CENTER_PX, CENTER_PX):
                continue
            tgt = (zoom(box, GRID / CENTER_PX, order=1) > BU_THRESH).astype(np.int64)[:GRID, :GRID]
            if tgt.shape != (GRID, GRID):
                continue
            l8 = l8_to_reflectance(patches[k].reshape(patches.shape[1] * 6, 224, 224))
            l8 = zoom(l8, (1, GRID / 224, GRID / 224), order=1)[:, :GRID, :GRID].astype(np.float32)
            X.append(l8); Y.append(tgt)
    if not X:
        return None, None
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))


def train(region, epochs=15):
    X, Y = region_xy(region)              # cached: data is seed-independent, reused across seeds
    if X is None or len(X) < 20:
        return None
    torch.manual_seed(SEED)
    tr = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))[:int(0.85 * len(X))].to(DEV)
    X, Y = X.to(DEV), Y.to(DEV)
    m = SegUNet(in_ch=X.shape[1], n=2).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3); ce = nn.CrossEntropyLoss()
    g = torch.Generator(device=DEV).manual_seed(SEED)
    for _ in range(epochs):
        m.train(); perm = tr[torch.randperm(len(tr), generator=g, device=DEV)]
        for k in range(0, len(perm), 8):
            b = perm[k:k+8]; opt.zero_grad(); loss = ce(m(X[b]), Y[b]); loss.backward(); opt.step()
    m.eval(); return m


_CACHE = {}
def region_xy(region):
    if region not in _CACHE:
        _CACHE[region] = load_region(region)
    return _CACHE[region]


def iou(model, region):
    X, Y = region_xy(region)
    if X is None:
        return None
    I = U = 0.0
    with torch.no_grad():
        for k in range(0, len(X), 16):
            p = model(X[k:k+16].to(DEV)).argmax(1).cpu().numpy(); y = Y[k:k+16].numpy()
            I += np.logical_and(p == 1, y == 1).sum(); U += np.logical_or(p == 1, y == 1).sum()
    return float(I / U) if U > 0 else None


def stats_for_seed(seed, ready):
    global SEED
    SEED = seed
    models = {r: train(r) for r in ready}; models = {r: m for r, m in models.items() if m is not None}
    rr = list(models)
    mat = {s: {t: iou(models[s], t) for t in rr} for s in models}
    diag = np.mean([mat[r][r] for r in rr if mat[r].get(r) is not None])
    off = np.mean([mat[s][t] for s in rr for t in rr if s != t and mat[s].get(t) is not None])
    srcs = [s for s in rr if all(mat[s].get(t) is not None for t in rr)]
    M = np.array([[mat[s][t] for t in rr] for s in srcs])
    inv = np.mean([spearmanr(M[a], M[b]).correlation
                   for a in range(len(srcs)) for b in range(a+1, len(srcs))]) if len(srcs) > 1 else float("nan")
    return {"in_region": float(diag), "out_region": float(off), "home_field_gap": float(diag - off),
            "retention": float(off / diag) if diag else None, "source_inv": float(inv),
            "diag": {r: round(float(mat[r][r]), 3) for r in rr if mat[r].get(r) is not None}}


def main(seeds, grid):
    global GRID
    GRID = grid; _CACHE.clear()           # alignment grid changed -> reload
    ready = [r for r in REGIONS if region_xy(r)[0] is not None and len(region_xy(r)[0]) >= 20]
    print(f"L8 [{L8DIR.name}] grid={grid} regions={ready} "
          f"tiles={[len(region_xy(r)[0]) for r in ready]}", flush=True)
    per = {}
    for sd in seeds:
        s = stats_for_seed(sd, ready); per[sd] = s
        print(f"[seed {sd}] in={s['in_region']:.3f} out={s['out_region']:.3f} "
              f"retention={s['retention']:.3f} gap={s['home_field_gap']:+.3f} "
              f"src_inv={s['source_inv']:.3f}", flush=True)
    agg = {}
    for k in ["in_region", "out_region", "home_field_gap", "retention", "source_inv"]:
        v = np.array([per[sd][k] for sd in seeds], dtype=float)
        agg[k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0}
    out = {"input": f"raw Landsat [{L8DIR.name}]", "target": "GHSL built-up 2015",
           "grid": grid, "seeds": list(seeds), "per_seed": {str(sd): per[sd] for sd in seeds},
           "aggregate": agg}
    suffix = "" if len(ready) <= 8 else f"_n{len(ready)}"
    fn = REPO / f"results/metrics/landsat_temporal_matrix_{L8DIR.name}_g{grid}{suffix}_multiseed.json"
    out["n_regions"] = len(ready); out["regions"] = ready
    json.dump(out, open(fn, "w"), indent=1)
    print(f"\n=== {L8DIR.name} grid={grid}, {len(seeds)} seeds ===")
    print(f"  retention {agg['retention']['mean']:.3f} +/- {agg['retention']['sd']:.3f}  |  "
          f"gap {agg['home_field_gap']['mean']:+.3f} +/- {agg['home_field_gap']['sd']:.3f}  |  "
          f"src_inv {agg['source_inv']['mean']:.2f} +/- {agg['source_inv']['sd']:.2f}", flush=True)
    print("  compare: GHSL-history ~1.0 ; single-date S2 0.82", flush=True)
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--data", default="landsat_history", choices=["landsat8", "landsat_history"])
    _p.add_argument("--seeds", type=int, nargs="+", default=[20260525, 1, 2, 3, 4])
    _p.add_argument("--grid", type=int, default=64)
    _a = _p.parse_args()
    L8DIR = REPO / f"data/raw/{_a.data}"
    GRID = _a.grid
    main(_a.seeds, _a.grid)
