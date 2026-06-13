"""Cross-region transfer matrix
for multi-class land-cover SEGMENTATION (Sentinel-2 -> ESA WorldCover).

This is the task where the multi-source domain-adaptation literature says the
source SHOULD matter. We train a small U-Net per source region and evaluate
it on every region, then report the same diagnostics as the built-up matrix:
source-invariance, home-field advantage, and per-region difficulty vs a
base-rate proxy (label-class entropy). If source-invariance + the confound
hold here too, this study's claim generalises; a clear home-field advantage
bounds it.

Usage: python -m scripts.eval.seg_transfer_matrix
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
REGIONS = ["south_asia","ssa","east_asia","andes","mena","sea","eeca","oceania"]
WC = [10,20,30,40,50,60,70,80,90,95,100]            # ESA WorldCover class codes
WC2IDX = {v:i for i,v in enumerate(WC)}; NCLS = len(WC)
SEED = 20260525


class SegUNet(nn.Module):
    def __init__(self, in_ch=4, n=NCLS):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                           nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.e1=blk(in_ch,32); self.e2=blk(32,64); self.bott=blk(64,128)
        self.p=nn.MaxPool2d(2)
        self.u2=nn.ConvTranspose2d(128,64,2,stride=2); self.d2=blk(128,64)
        self.u1=nn.ConvTranspose2d(64,32,2,stride=2); self.d1=blk(64,32)
        self.out=nn.Conv2d(32,n,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.p(e1)); b=self.bott(self.p(e2))
        d2=self.d2(torch.cat([self.u2(b),e2],1)); d1=self.d1(torch.cat([self.u1(d2),e1],1))
        return self.out(d1)


def load_region(region):
    Xs, Ys = [], []
    for f in glob.glob(str(REPO/f"data/hardtask/{region}/*/patches.npz")):
        d = np.load(f); Xs.append(d["s2"].astype(np.float32)); Ys.append(d["label"])
    if not Xs: return None, None
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    X = np.clip(X/3000.0, 0, 3)                       # rough S2 reflectance scaling
    Yi = np.zeros_like(Y, dtype=np.int64)
    for v,i in WC2IDX.items(): Yi[Y==v] = i
    return torch.from_numpy(X), torch.from_numpy(Yi)


def train_region(region, epochs=15):
    X, Y = load_region(region)
    if X is None or len(X) < 20: return None
    torch.manual_seed(SEED); n=len(X); idx=torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    tr=idx[:int(0.85*n)].to(DEV); X,Y=X.to(DEV),Y.to(DEV)
    m=SegUNet().to(DEV); opt=torch.optim.Adam(m.parameters(),1e-3); ce=nn.CrossEntropyLoss()
    g=torch.Generator(device=DEV).manual_seed(SEED)
    for _ in range(epochs):
        m.train(); perm=tr[torch.randperm(len(tr),generator=g,device=DEV)]
        for k in range(0,len(perm),8):
            b=perm[k:k+8]; opt.zero_grad(); loss=ce(m(X[b]),Y[b]); loss.backward(); opt.step()
    m.eval(); return m


def miou(model, region):
    X, Y = load_region(region)
    if X is None: return None
    inter=np.zeros(NCLS); union=np.zeros(NCLS)
    with torch.no_grad():
        for k in range(0,len(X),16):
            p=model(X[k:k+16].to(DEV)).argmax(1).cpu().numpy(); y=Y[k:k+16].numpy()
            for c in range(NCLS):
                pi=p==c; yi=y==c
                inter[c]+=np.logical_and(pi,yi).sum(); union[c]+=np.logical_or(pi,yi).sum()
    valid=union>0
    return float(np.mean(inter[valid]/union[valid])) if valid.any() else None


def class_entropy(region):
    _, Y = load_region(region)
    if Y is None: return None
    _,c=np.unique(Y.numpy(),return_counts=True); p=c/c.sum()
    return float(-(p*np.log(p)).sum())


def main():
    ready=[r for r in REGIONS if len(glob.glob(str(REPO/f"data/hardtask/{r}/*/patches.npz")))>=6]
    print(f"regions with >=6 cities of patches: {ready}", flush=True)
    models={r:train_region(r) for r in ready}; models={r:m for r,m in models.items() if m is not None}
    mat={s:{t:miou(models[s],t) for t in ready} for s in models}
    diag=np.mean([mat[r][r] for r in models if mat[r].get(r) is not None])
    off=np.mean([mat[s][t] for s in models for t in ready if s!=t and mat[s].get(t) is not None])
    srcs=[s for s in models if all(mat[s].get(t) is not None for t in ready)]
    M=np.array([[mat[s][t] for t in ready] for s in srcs])
    inv=np.mean([spearmanr(M[a],M[b]).correlation for a in range(len(srcs)) for b in range(a+1,len(srcs))]) if len(srcs)>1 else float("nan")
    ent={r:class_entropy(r) for r in ready}; dfc={r:float(np.mean([mat[s][r] for s in models if mat[s].get(r) is not None])) for r in ready}
    rho_ent=spearmanr([ent[r] for r in ready],[dfc[r] for r in ready]).correlation
    print(f"\n=== SEGMENTATION transfer matrix ({len(srcs)} sources x {len(ready)} targets) ===")
    print(f"home-field: in-region mIoU={diag:.3f} vs out-of-region={off:.3f} (gap {diag-off:+.3f})")
    print(f"source-invariance (row Spearman) = {inv:.3f}")
    print(f"per-region difficulty vs class-entropy (base-rate proxy): Spearman = {rho_ent:.3f}")
    json.dump({"ready":ready,"matrix":mat,"home_field_gap":round(float(diag-off),4),
               "in_region":round(float(diag),4),"out_region":round(float(off),4),
               "source_invariance":round(float(inv),3),"difficulty_vs_entropy_spearman":round(float(rho_ent),3)},
              open(REPO/"results/metrics/seg_transfer_matrix.json","w"),indent=1)
    print("saved results/metrics/seg_transfer_matrix.json")
if __name__=="__main__": main()
