"""Multi-seed robustness for the temporal transfer matrix (single-seed robustness). For each seed we retrain all eight per-region built-up models from scratch
into a seed-specific weights dir, rebuild the 8x8 source-by-target FoM matrix, and
read off the three headline quantities -- the home-field gap (in-region minus
out-of-region FoM), retention (out / in), and source-invariance (mean pairwise row
Spearman) -- then report mean +/- SD across seeds.

The seed-varying object is the 8x8 region sub-matrix (the eight per-region models
on the eight targets), which is exactly what the seed perturbs; the published
CONUS row is held fixed (we have only the released CONUS checkpoint) and is
materially identical as one extra out-of-region source. Canonical weights in
results/transfer_matrix/weights/ are never touched.

Usage:
    python -m scripts.eval.multiseed_matrix --arch cnn  --seeds 20260525 1 2 3 4
    python -m scripts.eval.multiseed_matrix --arch unet --seeds 20260525 1 2 3 4
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from scripts.eval import cross_region_train as crt
from scripts.eval.transfer_matrix import SRC_REGIONS, region_fom, ARCH

REPO = Path(__file__).resolve().parents[2]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(arch, state):
    m = ARCH[arch]().to(DEV); m.load_state_dict(state); m.eval(); return m


def matrix_for_seed(arch, seed, epochs):
    """Train (or load cached) all eight per-region models at `seed`, return the
    8x8 FoM matrix matrix[source][target]."""
    wdir = REPO / f"results/transfer_matrix/weights_seed{seed}"
    models = {}
    for s in SRC_REGIONS:
        ckpt = wdir / s / crt.CKPT[arch]
        if ckpt.exists():
            state = torch.load(ckpt, map_location=DEV)
        else:
            X, Y = crt.gather_tiles(s)
            print(f"[seed {seed}] train {s}/{arch} ({len(X)} tiles)", flush=True)
            state, _ = crt.fit(X, Y, arch, epochs, seed, DEV, tag=f"{s} ")
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, ckpt)
        models[s] = load_model(arch, state)
    mat = {s: {t: region_fom(arch, models[s], t) for t in SRC_REGIONS} for s in SRC_REGIONS}
    return mat


def stats(mat):
    diag = [mat[r][r] for r in SRC_REGIONS if mat[r].get(r) is not None]
    off = [mat[s][t] for s in SRC_REGIONS for t in SRC_REGIONS
           if s != t and mat[s].get(t) is not None]
    inr, outr = float(np.mean(diag)), float(np.mean(off))
    srcs = [s for s in SRC_REGIONS if all(mat[s].get(t) is not None for t in SRC_REGIONS)]
    M = np.array([[mat[s][t] for t in SRC_REGIONS] for s in srcs])
    inv = float(np.mean([spearmanr(M[a], M[b]).correlation
                         for a in range(len(srcs)) for b in range(a + 1, len(srcs))]))
    return {"in_region": inr, "out_region": outr, "home_field_gap": inr - outr,
            "retention": outr / inr, "source_invariance": inv}


def main(arch, seeds, epochs):
    per = {}
    for sd in seeds:
        mat = matrix_for_seed(arch, sd, epochs)
        per[sd] = {"stats": stats(mat), "matrix": mat}
        s = per[sd]["stats"]
        print(f"[seed {sd}] in={s['in_region']:.4f} out={s['out_region']:.4f} "
              f"gap={s['home_field_gap']:+.4f} retention={s['retention']:.3f} "
              f"src_inv={s['source_invariance']:.3f}", flush=True)
    agg = {}
    for k in ["in_region", "out_region", "home_field_gap", "retention", "source_invariance"]:
        v = np.array([per[sd]["stats"][k] for sd in seeds])
        agg[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                  "values": [float(x) for x in v]}
    out = {"arch": arch, "seeds": list(seeds), "n_seeds": len(seeds),
           "per_seed": {str(sd): per[sd] for sd in seeds}, "aggregate": agg}
    fn = REPO / f"results/metrics/multiseed_matrix_{arch}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"\n=== {arch}: {len(seeds)} seeds ===")
    for k in ["in_region", "out_region", "home_field_gap", "retention", "source_invariance"]:
        print(f"  {k:18} {agg[k]['mean']:+.4f} +/- {agg[k]['sd']:.4f}")
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="cnn", choices=["cnn", "unet", "convlstm"])
    p.add_argument("--seeds", type=int, nargs="+", default=[20260525, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=25)
    a = p.parse_args()
    main(a.arch, a.seeds, a.epochs)
