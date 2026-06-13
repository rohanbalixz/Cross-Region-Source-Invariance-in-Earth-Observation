"""Two-seed error bars: re-evaluate all cities with the seed-0 checkpoints
(without touching the canonical eval_metrics.json), then report per-region
CNN/UNet/ConvLSTM gaps for both seeds and their spread."""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.acquire.regions import CITIES
from scripts.eval.cross_region_eval import process_city
from scripts.eval.models import load_models

REPO=Path(__file__).resolve().parents[2]
PROC=REPO/"data/processed"; T="t=0.01"; CONUS={"cnn":0.581,"unet":0.558,"convlstm":0.391}
MODELS=["cnn","unet","convlstm"]
def region_means_from_dir(get_path):
    by={}
    for c in CITIES:
        p=get_path(c)
        if not p.exists(): continue
        per={m:[] for m in MODELS}
        for rec in json.load(open(p)):
            for m in MODELS:
                cell=((rec.get("models",{}).get(m,{}) or {}).get("fom",{}) or {}).get(T)
                if cell and cell.get("fom") is not None: per[m].append(cell["fom"])
        by.setdefault(c.region,{m:[] for m in MODELS})
        for m in MODELS:
            if per[m]: by[c.region][m].append(np.mean(per[m]))  # city-level mean
    return by
# --- default seed: canonical eval_metrics.json ---
default=region_means_from_dir(lambda c: PROC/c.region/c.name/"eval_metrics.json")
# --- seed0: stage checkpoints, eval to temp ---
stage=Path(tempfile.mkdtemp(prefix="seed0_w_"))
for tgt,src in [("best_cnn_3ch.pth","best_cnn_seed0.pth"),
                ("best_unet_3ch.pth","best_unet_3ch_seed0.pth"),
                ("best_3ch_mc_model.pth","best_3ch_mc_model_seed0.pth")]:
    shutil.copy(REPO.parent/"models"/src, stage/tgt)
dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
models=load_models(stage)
outdir=Path(tempfile.mkdtemp(prefix="seed0_eval_"))
for c in CITIES:
    if not (PROC/c.region/c.name/"eval_metrics.json").exists(): continue
    try: process_city(c, PROC, models, outdir/f"{c.name}.json", dev)
    except Exception as e: print("skip",c.name,e)
seed0=region_means_from_dir(lambda c: outdir/f"{c.name}.json")
# --- compare ---
order=["oceania","andes","ssa","sea","mena","south_asia","eeca","east_asia"]
print("\n=== Per-region CNN gap: seed-default vs seed-0 (city-mean) ===")
print(f"{'region':12}{'gap_default':>12}{'gap_seed0':>11}{'mean+/-halfrange':>18}")
for r in order:
    g0=np.mean(default[r]['cnn'])-CONUS['cnn']
    g1=np.mean(seed0[r]['cnn'])-CONUS['cnn']
    mean=(g0+g1)/2; hr=abs(g0-g1)/2
    print(f"{r:12}{g0:>+12.3f}{g1:>+11.3f}   {mean:>+.3f} +/- {hr:.3f}")
out={"models":{}}
for m in MODELS:
    out["models"][m]={r:{"gap_default":float(np.mean(default[r][m])-CONUS[m]),
                          "gap_seed0":float(np.mean(seed0[r][m])-CONUS[m])} for r in order}
json.dump(out,open(REPO/"results/metrics/multiseed_gaps.json","w"),indent=2)
print("\nsaved results/metrics/multiseed_gaps.json")
