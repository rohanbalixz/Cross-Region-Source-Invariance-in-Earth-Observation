"""Validate the freshly-acquired new regions before they enter any result.

For every city in the new regions this loads the preprocessed per-city rasters
and checks two things that catch a mis-placed bounding box:

  * 2015 built-up fraction -- a box dropped on ocean / desert / the wrong
    continent reads ~0 here;
  * 2010->2015 change fraction -- a static or empty box has nothing to predict.

A city is FLAGGED (not silently dropped) if its 2015 built-up mean is below
`MIN_BUILT` or its rasters are unreadable, so the coordinates can be fixed or
the city honestly excluded -- never faked.

Run: python -m scripts.eval.validate_new_regions
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

from scripts.acquire.regions import city_by_name
from scripts.eval.cross_region_eval import TARGET_EPOCH, TRAIN_EPOCHS, load_city_rasters

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
NEW8 = ["nordic", "central_europe", "china_interior", "south_asia_2",
        "southern_africa", "mediterranean", "brazil_north", "japan_korea_2"]
MIN_BUILT = 0.005     # below this the 2015 box is effectively unbuilt -> bad bbox


def city_stats(city):
    bu, _, _, _, _ = load_city_rasters(city, PROC)
    gt, prev = bu[TARGET_EPOCH], bu[TRAIN_EPOCHS[-1]]
    mask = gt > 0.01
    change = float(((gt - prev) > 0.01)[mask].mean()) if mask.sum() else 0.0
    return {"built2015": float(gt.mean()), "change": change,
            "shape": list(gt.shape)}


def main(regions):
    summary = {}
    for region in regions:
        cities = sorted(glob.glob(str(PROC / f"{region}/*/builtup_2015.tif")))
        rows, flagged = [], []
        for f in cities:
            city = city_by_name(Path(f).parent.name)
            try:
                st = city_stats(city)
            except Exception as e:
                flagged.append((city.name, f"unreadable:{type(e).__name__}"))
                continue
            bad = st["built2015"] < MIN_BUILT
            tag = "  FLAG" if bad else "ok"
            if bad:
                flagged.append((city.name, f"built2015={st['built2015']:.4f}"))
            rows.append((city.name, st))
            print(f"  [{region}/{city.name:14s}] built2015={st['built2015']:.4f} "
                  f"change={st['change']:.3f} {st['shape']} {tag}", flush=True)
        ok = [r for r in rows if r[1]["built2015"] >= MIN_BUILT]
        summary[region] = {
            "n_cities_present": len(cities), "n_ok": len(ok),
            "n_flagged": len(flagged), "flagged": flagged,
            "mean_built2015": float(np.mean([r[1]["built2015"] for r in ok])) if ok else None,
            "mean_change": float(np.mean([r[1]["change"] for r in ok])) if ok else None,
        }
        print(f"=== {region}: {len(ok)}/{len(cities)} cities OK, "
              f"{len(flagged)} flagged; mean built2015="
              f"{summary[region]['mean_built2015']}, mean change="
              f"{summary[region]['mean_change']}\n", flush=True)
    out = REPO / "results/metrics/new_region_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=1)
    tot_ok = sum(s["n_ok"] for s in summary.values())
    tot_flag = sum(s["n_flagged"] for s in summary.values())
    print(f"TOTAL across {len(regions)} regions: {tot_ok} cities OK, {tot_flag} flagged")
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--region", action="append", default=[])
    a = p.parse_args()
    main(a.region or NEW8)
