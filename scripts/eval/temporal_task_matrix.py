"""Temporal-task confirmation: are other
temporal-extrapolation tasks also source-invariant, like built-up nowcasting?

We run the source-by-target transfer matrix for population nowcasting and
built-volume nowcasting (predict the 2015 layer from the 1975-2010 history),
using the same GHSL inputs and the same small CNN. If both are source-
invariant (home-field gap ~0), then 'temporal-history input -> source-
invariant' has three anchors against five source-dependent imagery tasks,
turning the input-representation boundary into a characterised law.

Usage: python -m scripts.eval.temporal_task_matrix
"""
import json, glob, tempfile
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
from scripts.acquire.regions import city_by_name
from scripts.eval.models import SimpleCNN
from scripts.eval.cross_region_eval import (
    load_city_rasters, fom_metrics, TILE_PX, EVAL_MASK_THRESH)
REPO = Path(__file__).resolve().parents[2]; PROC = REPO/"data/processed"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
REGIONS = ["south_asia","ssa","east_asia","andes","mena","sea","eeca","oceania"]
SEED = 20260525; T = 0.01
# Shifted window (vol/pop 2015 were never processed): predict the 2010 layer
# from 1975-2005 history. Valid temporal-extrapolation, uses only on-disk data.
IN_EPOCHS = [1975,1980,1985,1990,1995,2000,2005]; TGT_EPOCH = 2010; PREV_EPOCH = 2005
IN_CH = len(IN_EPOCHS)*3   # 21


def _input(bu, vol, pop, i, j):
    frames = [np.stack([bu[y][i:i+TILE_PX,j:j+TILE_PX], vol[y][i:i+TILE_PX,j:j+TILE_PX],
                        pop[y][i:i+TILE_PX,j:j+TILE_PX]], 0) for y in IN_EPOCHS]
    return torch.from_numpy(np.stack(frames,0).astype(np.float32))   # (7,3,H,W)


def gather(source, layer, stride=64):
    """(input 1975-2005, target = `layer` 2010 map, prev = layer 2005)."""
    X, Y, P = [], [], []
    for f in glob.glob(str(PROC/f"{source}/*/builtup_2015.tif")):
        city = city_by_name(Path(f).parent.name)
        bu, vol, pop, _, _ = load_city_rasters(city, PROC)
        tgt = {"bu": bu, "vol": vol, "pop": pop}[layer]
        H, W = bu[TGT_EPOCH].shape
        for i in range(0, H-TILE_PX+1, stride):
            for j in range(0, W-TILE_PX+1, stride):
                t = tgt[TGT_EPOCH][i:i+TILE_PX, j:j+TILE_PX]
                if t.mean() < 0.005: continue
                X.append(_input(bu, vol, pop, i, j))
                Y.append(torch.from_numpy(t.astype(np.float32))[None])
                P.append(torch.from_numpy(tgt[PREV_EPOCH][i:i+TILE_PX, j:j+TILE_PX].astype(np.float32))[None])
    if not X: return None, None, None
    return torch.stack(X), torch.stack(Y), torch.stack(P)


def soft_jacc(p, t, eps=1e-6):
    i = (p*t).sum((1,2,3)); u = (p+t-p*t).sum((1,2,3)); return 1-((i+eps)/(u+eps)).mean()


def train(source, layer, epochs=20):
    X, Y, _ = gather(source, layer)
    if X is None or len(X) < 20: return None
    torch.manual_seed(SEED); n=len(X)
    tr = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:int(0.85*n)].to(DEV)
    X, Y = X.to(DEV), Y.to(DEV)
    m = SimpleCNN(input_channels=IN_CH).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator(device=DEV).manual_seed(SEED)
    for _ in range(epochs):
        m.train(); perm = tr[torch.randperm(len(tr), generator=g, device=DEV)]
        for k in range(0, len(perm), 16):
            b = perm[k:k+16]; opt.zero_grad(); p = m(X[b])
            loss = nn.functional.mse_loss(p, Y[b]) + 0.5*soft_jacc(p, Y[b]); loss.backward(); opt.step()
    m.eval(); return m


def region_fom(model, region, layer):
    Xa, Ya, Pa = gather(region, layer, stride=TILE_PX)   # non-overlap for eval
    if Xa is None: return None
    fs=[]
    with torch.no_grad():
        for k in range(len(Xa)):
            pred = model(Xa[k:k+1].to(DEV)).squeeze().cpu().numpy()
            gt = Ya[k,0].numpy(); prev = Pa[k,0].numpy(); mask = gt > EVAL_MASK_THRESH
            if mask.sum()==0: continue
            r = fom_metrics(gt, prev, pred, mask, T); fs.append(r["fom"] if isinstance(r,dict) else r)
    fs=[x for x in fs if x is not None]
    return float(np.mean(fs)) if fs else None


def run_layer(layer):
    models={r: train(r, layer) for r in REGIONS}; models={r:m for r,m in models.items() if m is not None}
    rr=list(models)
    mat={s:{t:region_fom(models[s], t, layer) for t in rr} for s in models}
    diag=np.mean([mat[r][r] for r in models if mat[r].get(r) is not None])
    off=np.mean([mat[s][t] for s in models for t in rr if s!=t and mat[s].get(t) is not None])
    srcs=[s for s in models if all(mat[s].get(t) is not None for t in rr)]
    M=np.array([[mat[s][t] for t in rr] for s in srcs])
    inv=np.mean([spearmanr(M[a],M[b]).correlation for a in range(len(srcs)) for b in range(a+1,len(srcs))]) if len(srcs)>1 else float("nan")
    return {"task":f"{layer}_nowcast(temporal)","n_sources":len(models),
            "in_region_fom":round(float(diag),3),"out_region_fom":round(float(off),3),
            "home_field_gap":round(float(diag-off),4),"source_invariance":round(float(inv),3)}


def main():
    out=[]
    for layer in ["bu","vol","pop"]:
        print(f"[{layer} nowcasting] running matrix ...", flush=True)
        r=run_layer(layer); out.append(r); print(f"  {r}", flush=True)
    json.dump(out, open(REPO/"results/metrics/temporal_task_matrix.json","w"), indent=1)
    print("\n=== TEMPORAL-TASK CONFIRMATION ===")
    print("built-up nowcast (reference): home-field gap +0.001, source-inv 0.94")
    for r in out: print(f"{r['task']:28} gap={r['home_field_gap']:+.3f}  source-inv={r['source_invariance']:.3f}")
    gaps=[r['home_field_gap'] for r in out]
    print(f"\n=> if pop & vol gaps are ~0 (like built-up's +0.001), temporal-extrapolation is source-invariant")
    print(f"   across 3 tasks, vs imagery-segmentation gaps +0.06..+0.19 -> the input-representation law holds.")
    print("saved results/metrics/temporal_task_matrix.json")
if __name__=="__main__": main()
