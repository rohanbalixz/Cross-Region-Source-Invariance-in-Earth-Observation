"""Multi-metric robustness Re-evaluate the already-trained
transfer matrices under metrics beyond FoM: built-up EXTENT IoU and F1 (state
segmentation, no change-rate confound) and change-F1. For each (architecture,
metric) report the home-field gap and source-invariance, to test whether the
no-home-field / source-invariance conclusion is FoM-specific.
"""
import json, glob
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from scripts.acquire.regions import CITIES
from scripts.eval.models import SimpleCNN, SimpleUNet, ConvLSTMModel
from scripts.eval.cross_region_eval import (
    load_city_rasters, build_input_tensor, enumerate_tiles_from_grid,
    TILE_PX, TRAIN_EPOCHS, TARGET_EPOCH, EVAL_MASK_THRESH)
REPO = Path(__file__).resolve().parents[2]; PROC = REPO/"data/processed"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
REGIONS = ["south_asia","ssa","east_asia","andes","mena","sea","eeca","oceania"]
ARCH = {"cnn":SimpleCNN,"unet":SimpleUNet,"convlstm":ConvLSTMModel}
CKPT = {"cnn":"best_cnn_3ch.pth","unet":"best_unet_3ch.pth","convlstm":"best_3ch_mc_model.pth"}
TAU = 0.2   # built-up presence threshold for extent metrics


def f1(tp, fp, fn):
    return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else None
def iou(tp, fp, fn):
    return tp/(tp+fp+fn) if (tp+fp+fn) else None


def city_metrics(model, city):
    bu, vol, pop, transform, crs = load_city_rasters(city, PROC)
    gt, prev = bu[TARGET_EPOCH], bu[TRAIN_EPOCHS[-1]]
    refs = enumerate_tiles_from_grid(builtup_2015=gt, utm_transform=transform,
                                     city_name=city.name, region=city.region, utm_crs=crs)
    acc = {"extent_iou":[], "extent_f1":[], "change_f1":[]}
    for r in refs:
        i, j = r.i, r.j
        g = gt[i:i+TILE_PX, j:j+TILE_PX]; p0 = prev[i:i+TILE_PX, j:j+TILE_PX]
        x = build_input_tensor(bu, vol, pop, i, j).to(DEV)
        with torch.no_grad():
            pr = model(x).squeeze().cpu().numpy()
        gb, pb = g > TAU, pr > TAU                       # extent (state)
        tp=int((gb&pb).sum()); fp=int((~gb&pb).sum()); fn=int((gb&~pb).sum())
        if tp+fp+fn: acc["extent_iou"].append(iou(tp,fp,fn)); acc["extent_f1"].append(f1(tp,fp,fn))
        gc, pc = (g-p0)>0.01, (pr-p0)>0.01               # change
        tpc=int((gc&pc).sum()); fpc=int((~gc&pc).sum()); fnc=int((gc&~pc).sum())
        v=f1(tpc,fpc,fnc)
        if v is not None: acc["change_f1"].append(v)
    return {k:(float(np.mean(v)) if v else None) for k,v in acc.items()}


def region_metrics(model, region):
    vals={"extent_iou":[], "extent_f1":[], "change_f1":[]}
    for c in [c for c in CITIES if c.region==region]:
        if not (PROC/region/c.name/"eval_metrics.json").exists(): continue
        try: m=city_metrics(model, c)
        except Exception: continue
        for k in vals:
            if m[k] is not None: vals[k].append(m[k])
    return {k:(float(np.mean(v)) if v else None) for k,v in vals.items()}


def load(arch, path):
    m=ARCH[arch]().to(DEV); m.load_state_dict(torch.load(path,map_location=DEV)); m.eval(); return m


def main():
    out={}
    for arch in ["cnn","unet","convlstm"]:
        srcs={s:REPO/f"results/transfer_matrix/weights/{s}/{CKPT[arch]}" for s in REGIONS}
        srcs["conus"]=REPO.parent/f"models/{CKPT[arch]}"
        mats={m:{} for m in ("extent_iou","extent_f1","change_f1")}
        for s,ck in srcs.items():
            if not Path(ck).exists(): continue
            model=load(arch,ck)
            rm={t:region_metrics(model,t) for t in REGIONS}
            for met in mats: mats[met][s]={t:rm[t][met] for t in REGIONS}
        out[arch]={}
        for met,M in mats.items():
            ss=[s for s in M if all(M[s].get(t) is not None for t in REGIONS)]
            A=np.array([[M[s][t] for t in REGIONS] for s in ss])
            diag=np.mean([M[r][r] for r in REGIONS if M.get(r,{}).get(r) is not None])
            off=np.mean([M[s][t] for s in REGIONS for t in REGIONS if s!=t and M[s].get(t) is not None])
            rho=np.mean([spearmanr(A[a],A[b]).correlation for a in range(len(ss)) for b in range(a+1,len(ss))])
            out[arch][met]={"home_field_gap":round(float(diag-off),4),"source_invariance":round(float(rho),3),
                            "mean":round(float(np.nanmean(A)),3)}
            print(f"[{arch:8} {met:11}] gap={diag-off:+.4f}  source-inv(rho)={rho:.3f}  mean={np.nanmean(A):.3f}", flush=True)
    json.dump(out, open(REPO/"results/metrics/multimetric_matrix.json","w"), indent=1)
    print("\nsaved results/metrics/multimetric_matrix.json")
if __name__=="__main__": main()
