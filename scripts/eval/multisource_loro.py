"""Leave-one-region-out multi-source training (domain-generalisation control): does pooling
seven source regions beat training on a single foreign region? If "the source is
inert", pooling should NOT lift out-of-region performance on temporal-history
tasks. For each held-out target h we train one model on the union of the other
seven regions (identical recipe via cross_region_train.fit) and evaluate on h,
then compare to (a) the home model, in-region, and (b) the single-foreign-source
average -- the mean over s != h of the published transfer matrix -- at the same
seed. The standard multi-source domain-generalisation check the matrix study
omits.

We report, across the eight folds, multi-source retention vs single-source
retention and the pooling benefit (LORO FoM minus single-source FoM); mean +/- SD
is taken over the eight held-out regions.

Usage: python -m scripts.eval.multisource_loro --arch cnn --seed 20260525
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.eval import cross_region_train as crt
from scripts.eval.transfer_matrix import ARCH, SRC_REGIONS, region_fom

REPO = Path(__file__).resolve().parents[2]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(arch, state):
    m = ARCH[arch]().to(DEV); m.load_state_dict(state); m.eval(); return m


def main(arch, seed, epochs):
    canon = json.load(open(REPO / f"results/metrics/transfer_matrix_{arch}.json"))
    wdir = REPO / f"results/transfer_matrix/weights_loro_seed{seed}"
    rows = {}
    for h in SRC_REGIONS:
        srcs = [s for s in SRC_REGIONS if s != h]
        ckpt = wdir / h / crt.CKPT[arch]
        if ckpt.exists():
            state = torch.load(ckpt, map_location=DEV)
        else:
            X, Y = crt.gather_tiles(srcs)
            print(f"[LORO {arch}] held-out={h}: train on {len(srcs)} regions "
                  f"({len(X)} tiles)", flush=True)
            state, _ = crt.fit(X, Y, arch, epochs, seed, DEV, tag=f"-{h} ")
            ckpt.parent.mkdir(parents=True, exist_ok=True); torch.save(state, ckpt)
        m = load_model(arch, state)
        loro = region_fom(arch, m, h)
        in_region = canon[h][h]
        single_out = float(np.mean([canon[s][h] for s in srcs
                                    if canon.get(s, {}).get(h) is not None]))
        rows[h] = {"in_region": in_region, "single_source_out": single_out,
                   "multi_source_loro": loro,
                   "retention_single": single_out / in_region,
                   "retention_loro": loro / in_region,
                   "loro_minus_single": loro - single_out}
        print(f"  {h:11} in={in_region:.3f} single_out={single_out:.3f} "
              f"loro={loro:.3f} (loro-single={loro - single_out:+.3f})", flush=True)

    def agg(key):
        v = np.array([rows[h][key] for h in SRC_REGIONS])
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}

    summary = {k: agg(k) for k in ["in_region", "single_source_out", "multi_source_loro",
                                   "retention_single", "retention_loro", "loro_minus_single"]}
    out = {"arch": arch, "seed": seed, "n_folds": len(SRC_REGIONS),
           "per_region": rows, "summary": summary}
    fn = REPO / f"results/metrics/multisource_loro_{arch}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"\n=== LORO {arch} (seed {seed}), {len(SRC_REGIONS)} folds ===")
    print(f"  single-source retention: {summary['retention_single']['mean']:.3f} "
          f"+/- {summary['retention_single']['sd']:.3f}")
    print(f"  multi-source  retention: {summary['retention_loro']['mean']:.3f} "
          f"+/- {summary['retention_loro']['sd']:.3f}")
    print(f"  pooling benefit (loro-single FoM): {summary['loro_minus_single']['mean']:+.4f} "
          f"+/- {summary['loro_minus_single']['sd']:.4f}")
    print(f"saved {fn}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="cnn", choices=["cnn", "unet", "convlstm"])
    p.add_argument("--seed", type=int, default=20260525)
    p.add_argument("--epochs", type=int, default=25)
    a = p.parse_args()
    main(a.arch, a.seed, a.epochs)
