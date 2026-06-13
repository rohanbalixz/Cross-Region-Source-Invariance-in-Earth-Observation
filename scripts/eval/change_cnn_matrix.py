"""Can a temporal CNN trained to predict change (not the
2015 state) beat the parameter-free linear-extrapolation baseline, and does its
margin over that baseline transfer across regions?

Train one CNN per source region on Y = built-up_2015 - built-up_2010 (growth),
same input history and recipe as the state CNN. Evaluate FoM cross-region via
pred_2015 = built-up_2010 + predicted_change, scored identically to this study.
Build the 8x8 matrix, then per region compare to linext (from morph_baseline.json)
and to the state-CNN matrix. Margin > 0 and source-invariant => the temporal
model has genuine transferable change-skill (mechanism rescued); margin <= 0 =>
deflation holds even with change training.

Usage: python -m scripts.eval.change_cnn_matrix
"""
import glob, json
from pathlib import Path
import numpy as np, torch
from scripts.acquire.regions import CITIES, city_by_name
from scripts.eval.models import SimpleCNN
from scripts.eval import cross_region_train as crt
from scripts.eval.cross_region_eval import (
    load_city_rasters, build_input_tensor, fom_metrics,
    EVAL_MASK_THRESH, TARGET_EPOCH, TRAIN_EPOCHS)
from scripts.common import TILE_PX, enumerate_tiles_from_grid

REPO = Path(__file__).resolve().parents[2]; PROC = REPO / "data/processed"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
T = 0.01; SEED = 20260525
SRC = ["south_asia", "ssa", "east_asia", "andes", "mena", "sea", "eeca", "oceania"]


def gather_change(region, stride=64):
    X, Y = [], []
    for f in glob.glob(str(PROC / f"{region}/*/builtup_2015.tif")):
        city = city_by_name(Path(f).parent.name)
        bu, vol, pop, _, _ = load_city_rasters(city, PROC)
        prev, tgt = bu[TRAIN_EPOCHS[-1]], bu[TARGET_EPOCH]
        H, W = tgt.shape
        for i in range(0, H - TILE_PX + 1, stride):
            for j in range(0, W - TILE_PX + 1, stride):
                t = tgt[i:i+TILE_PX, j:j+TILE_PX]
                if t.mean() < 0.005:
                    continue
                ch = np.clip(t - prev[i:i+TILE_PX, j:j+TILE_PX], 0.0, 1.0)
                X.append(build_input_tensor(bu, vol, pop, i, j)[0])
                Y.append(torch.from_numpy(ch[None].astype(np.float32)))
    return torch.stack(X), torch.stack(Y)


def region_fom_change(model, region):
    city_foms = []
    for c in [c for c in CITIES if c.region == region]:
        if not (PROC / region / c.name / "eval_metrics.json").exists():
            continue
        try:
            bu, vol, pop, transform, crs = load_city_rasters(c, PROC)
        except Exception:
            continue
        gt, prev = bu[TARGET_EPOCH], bu[TRAIN_EPOCHS[-1]]
        refs = enumerate_tiles_from_grid(builtup_2015=gt, utm_transform=transform,
                                         city_name=c.name, region=region, utm_crs=crs)
        tile_foms = []
        for ref in refs:
            i, j = ref.i, ref.j
            g = gt[i:i+TILE_PX, j:j+TILE_PX]; p = prev[i:i+TILE_PX, j:j+TILE_PX]
            mask = g > EVAL_MASK_THRESH
            if mask.sum() == 0:
                continue
            x = build_input_tensor(bu, vol, pop, i, j).to(DEV)
            with torch.no_grad():
                dch = model(x).squeeze().cpu().numpy()
            fm = fom_metrics(g, p, p + dch, mask, T)
            tile_foms.append(fm["fom"])
        if tile_foms:
            city_foms.append(float(np.mean(tile_foms)))
    return float(np.mean(city_foms)) if city_foms else None


def main():
    linext = {r: json.load(open(REPO / "results/metrics/morph_baseline.json"))
              ["per_region"][r]["linext"] for r in SRC}
    state = json.load(open(REPO / "results/metrics/transfer_matrix_cnn.json"))
    models = {}
    for s in SRC:
        X, Y = gather_change(s)
        print(f"[change-cnn {s}] {len(X)} tiles", flush=True)
        st, _ = crt.fit(X, Y, "cnn", 25, SEED, DEV, tag=f"{s} ")
        m = SimpleCNN().to(DEV); m.load_state_dict(st); m.eval(); models[s] = m
    mat = {s: {t: region_fom_change(models[s], t) for t in SRC} for s in SRC}
    out = {}
    for r in SRC:
        ch_in = mat[r][r]
        ch_out = float(np.mean([mat[s][r] for s in SRC if s != r and mat[s].get(r) is not None]))
        st_in = state[r][r]
        out[r] = {"linext": linext[r], "change_cnn_in": ch_in, "change_cnn_out": ch_out,
                  "state_cnn_in": st_in, "margin_in": ch_in - linext[r], "margin_out": ch_out - linext[r]}
        print(f"{r:11} linext={linext[r]:.3f} change-cnn in={ch_in:.3f} out={ch_out:.3f} "
              f"state-cnn in={st_in:.3f} | margin_in={ch_in-linext[r]:+.3f} margin_out={ch_out-linext[r]:+.3f}",
              flush=True)
    mi = float(np.mean([out[r]["margin_in"] for r in SRC]))
    mo = float(np.mean([out[r]["margin_out"] for r in SRC]))
    summary = {"linext_mean": float(np.mean([linext[r] for r in SRC])),
               "change_cnn_in_mean": float(np.mean([out[r]["change_cnn_in"] for r in SRC])),
               "change_cnn_out_mean": float(np.mean([out[r]["change_cnn_out"] for r in SRC])),
               "state_cnn_in_mean": float(np.mean([out[r]["state_cnn_in"] for r in SRC])),
               "margin_in_mean": mi, "margin_out_mean": mo,
               "margin_retention": (mo / mi if mi > 0 else None)}
    print("\n=== change-trained CNN vs linext (built-up FoM) ===")
    print(f"linext {summary['linext_mean']:.3f} | change-cnn in {summary['change_cnn_in_mean']:.3f} "
          f"out {summary['change_cnn_out_mean']:.3f} | state-cnn in {summary['state_cnn_in_mean']:.3f}")
    print(f"margin over linext: in {mi:+.3f}, out {mo:+.3f}, margin-retention {summary['margin_retention']}",
          flush=True)
    json.dump({"per_region": out, "summary": summary},
              open(REPO / "results/metrics/change_cnn_matrix.json", "w"), indent=1)
    print("saved results/metrics/change_cnn_matrix.json", flush=True)


if __name__ == "__main__":
    main()
