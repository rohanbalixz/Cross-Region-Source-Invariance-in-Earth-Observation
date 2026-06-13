"""Full cross-region transfer matrix for a given architecture: train on each
region, eval every source-model on every target region. Source #9 = existing
CONUS-trained model. Usage: python -m scripts.eval.transfer_matrix --arch unet
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.acquire.regions import CITIES
from scripts.eval import cross_region_train as crt
from scripts.eval.cross_region_eval import process_city
from scripts.eval.models import ConvLSTMModel, SimpleCNN, SimpleUNet

REPO=Path(__file__).resolve().parents[2]; PROC=REPO/"data/processed"; T="t=0.01"
DEV=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SRC_REGIONS=["south_asia","ssa","east_asia","andes","mena","sea","eeca","oceania"]
ARCH={"cnn":SimpleCNN,"unet":SimpleUNet,"convlstm":ConvLSTMModel}
CONUS_CK={"cnn":"best_cnn_3ch.pth","unet":"best_unet_3ch.pth","convlstm":"best_3ch_mc_model.pth"}
def train_all(arch,epochs=25):
    for s in SRC_REGIONS:
        if (REPO/f"results/transfer_matrix/weights/{s}/{crt.CKPT[arch]}").exists():
            print(f"[skip train] {s}/{arch}",flush=True); continue
        crt.main(s,arch,epochs,20260525,DEV)
def load_model(arch,path):
    m=ARCH[arch]().to(DEV); m.load_state_dict(torch.load(path,map_location=DEV)); m.eval(); return m
def region_fom(arch, model, region):
    out=Path(tempfile.mkdtemp()); foms=[]
    for c in [c for c in CITIES if c.region==region]:
        if not (PROC/region/c.name/"eval_metrics.json").exists(): continue
        try: process_city(c, PROC, {arch:model}, out/f"{c.name}.json", DEV)
        except Exception as e: print("  skip",c.name,e); continue
        per=[]
        for rec in json.load(open(out/f"{c.name}.json")):
            cell=((rec.get("models",{}).get(arch,{}) or {}).get("fom",{}) or {}).get(T)
            if cell and cell.get("fom") is not None: per.append(cell["fom"])
        if per: foms.append(np.mean(per))
    return float(np.mean(foms)) if foms else None
def main(arch):
    train_all(arch)
    sources={s: REPO/f"results/transfer_matrix/weights/{s}/{crt.CKPT[arch]}" for s in SRC_REGIONS}
    sources["conus"]=REPO.parent/f"models/{CONUS_CK[arch]}"
    matrix={}
    for src,ck in sources.items():
        if not Path(ck).exists(): print(f"[no ckpt] {src}/{arch}");continue
        m=load_model(arch,ck)
        matrix[src]={tgt: region_fom(arch,m,tgt) for tgt in SRC_REGIONS}
        print(f"[matrix {arch}] source={src}: "+" ".join(f"{t}={matrix[src][t]:.3f}" for t in SRC_REGIONS if matrix[src][t] is not None),flush=True)
    fn=REPO/f"results/metrics/transfer_matrix_{arch}.json"
    json.dump(matrix, open(fn,"w"), indent=1); print(f"saved {fn}",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--arch",default="cnn",choices=list(ARCH)); a=p.parse_args()
    main(a.arch)
