"""Confound-free cross-check of the temporal LORO result. Re-score the SAVED
leave-one-region-out models and the canonical single-source models under the
change-rate-free extent metrics (built-up extent IoU/F1) and change-F1, per
held-out region. The temporal LORO on FoM shows pooling underperforming the
single-source average; if that gap is the section-5 change-magnitude confound
(pooling flattens predicted change toward a global mean, which FoM punishes)
rather than real degradation, then on the confound-free extent IoU the pooled
model should match the single-source average -- i.e. multi-source pooling is
neutral for temporal transfer, consistent with the source being inert.

Reuses multimetric_matrix.region_metrics / load (extent IoU/F1, change-F1).
Usage: python -m scripts.eval.loro_extent_check --arch cnn --seed 20260525
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.eval.multimetric_matrix import CKPT, load, region_metrics
from scripts.eval.transfer_matrix import SRC_REGIONS

REPO = Path(__file__).resolve().parents[2]
METS = ["extent_iou", "extent_f1", "change_f1"]


def main(arch, seed):
    wl = REPO / f"results/transfer_matrix/weights_loro_seed{seed}"
    wc = REPO / "results/transfer_matrix/weights"
    rows = {}
    for h in SRC_REGIONS:
        srcs = [s for s in SRC_REGIONS if s != h]
        inm = region_metrics(load(arch, wc / h / CKPT[arch]), h)              # home model on h
        singles = [region_metrics(load(arch, wc / s / CKPT[arch]), h) for s in srcs]
        lorom = region_metrics(load(arch, wl / h / CKPT[arch]), h)            # pooled model on h
        rows[h] = {}
        for met in METS:
            inv = inm[met]
            sv = float(np.mean([s[met] for s in singles if s[met] is not None]))
            lv = lorom[met]
            rows[h][met] = {"in": inv, "single": sv, "loro": lv,
                            "ret_single": (sv / inv if inv else None),
                            "ret_loro": (lv / inv if inv else None),
                            "loro_minus_single": (lv - sv if (lv is not None and sv is not None) else None)}
        e = rows[h]["extent_iou"]
        print(f"  {h:11} extentIoU: in={e['in']:.3f} single={e['single']:.3f} "
              f"loro={e['loro']:.3f} (loro-single={e['loro'] - e['single']:+.4f})", flush=True)
    agg = {}
    for met in METS:
        d = np.array([rows[h][met]["loro_minus_single"] for h in SRC_REGIONS])
        rs = np.array([rows[h][met]["ret_single"] for h in SRC_REGIONS])
        rl = np.array([rows[h][met]["ret_loro"] for h in SRC_REGIONS])
        agg[met] = {"loro_minus_single_mean": float(d.mean()), "loro_minus_single_sd": float(d.std(ddof=1)),
                    "ret_single_mean": float(rs.mean()), "ret_loro_mean": float(rl.mean())}
        print(f"[{met:10}] ret_single={rs.mean():.3f} ret_loro={rl.mean():.3f} "
              f"(loro-single={d.mean():+.4f} +/- {d.std(ddof=1):.4f})", flush=True)
    json.dump({"arch": arch, "seed": seed, "per_region": rows, "aggregate": agg},
              open(REPO / f"results/metrics/loro_extent_{arch}.json", "w"), indent=1)
    print(f"saved results/metrics/loro_extent_{arch}.json", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="cnn", choices=["cnn", "unet", "convlstm"])
    p.add_argument("--seed", type=int, default=20260525)
    a = p.parse_args()
    main(a.arch, a.seed)
