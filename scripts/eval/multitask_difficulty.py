"""Multi-task difficulty sweep: does source-invariance
break PREDICTABLY as the task gets harder? We run the same 7x7 source-by-target
transfer matrix on several land-cover tasks spanning difficulty (binary
single-class segmentation of increasing ambiguity, then full 11-class), all
from the already-acquired S2+WorldCover patches. For each task we record the
home-field gap, source-invariance, and an intrinsic-difficulty proxy
(mean in-region mIoU). If invariance declines monotonically with difficulty,
the easy-vs-hard boundary becomes a characterised law, not a single point.

Usage: python -m scripts.eval.multitask_difficulty
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
REGIONS = ["south_asia","ssa","east_asia","andes","mena","eeca","oceania"]  # the 7 ready
WC = [10,20,30,40,50,60,70,80,90,95,100]
SEED = 20260525
# tasks: (name, n_classes, label_fn) -- binary tasks are 1=positive class, ordered easy->hard guess
TASKS = [
    ("water_binary",   2, lambda Y: (Y==80).astype(np.int64)),     # high spectral contrast (easy)
    ("builtup_binary", 2, lambda Y: (Y==50).astype(np.int64)),     # medium
    ("tree_binary",    2, lambda Y: (Y==10).astype(np.int64)),     # medium-hard
    ("crop_binary",    2, lambda Y: (Y==40).astype(np.int64)),     # spectrally ambiguous (hard)
    ("multiclass11",  11, None),                                    # hardest (done separately too)
]
WC2IDX = {v:i for i,v in enumerate(WC)}


class SegUNet(nn.Module):
    def __init__(self, in_ch=4, n=2):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),
                                           nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU())
        self.e1=blk(in_ch,32);self.e2=blk(32,64);self.bott=blk(64,128);self.p=nn.MaxPool2d(2)
        self.u2=nn.ConvTranspose2d(128,64,2,stride=2);self.d2=blk(128,64)
        self.u1=nn.ConvTranspose2d(64,32,2,stride=2);self.d1=blk(64,32);self.out=nn.Conv2d(32,n,1)
    def forward(self,x):
        e1=self.e1(x);e2=self.e2(self.p(e1));b=self.bott(self.p(e2))
        d2=self.d2(torch.cat([self.u2(b),e2],1));d1=self.d1(torch.cat([self.u1(d2),e1],1));return self.out(d1)


_CACHE={}
def load_raw(region):
    if region in _CACHE: return _CACHE[region]
    Xs,Ys=[],[]
    for f in glob.glob(str(REPO/f"data/hardtask/{region}/*/patches.npz")):
        d=np.load(f); Xs.append(d["s2"].astype(np.float32)); Ys.append(d["label"])
    if not Xs: _CACHE[region]=(None,None); return None,None
    X=np.clip(np.concatenate(Xs)/3000.0,0,3); Y=np.concatenate(Ys)
    _CACHE[region]=(X,Y); return X,Y

def make_labels(Yraw, task):
    name,nc,fn=task
    if fn is not None: return fn(Yraw)
    Yi=np.zeros_like(Yraw,dtype=np.int64)
    for v,i in WC2IDX.items(): Yi[Yraw==v]=i
    return Yi

def train(region, task, epochs=12):
    name,nc,_=task; X,Yraw=load_raw(region)
    if X is None or len(X)<20: return None
    Y=make_labels(Yraw,task)
    if nc==2 and Y.sum()<50: return None     # positive class essentially absent
    Xt=torch.from_numpy(X); Yt=torch.from_numpy(Y)
    g=torch.Generator().manual_seed(SEED); idx=torch.randperm(len(Xt),generator=g)
    tr=idx[:int(0.85*len(Xt))].to(DEV); Xt=Xt.to(DEV); Yt=Yt.to(DEV)
    m=SegUNet(n=nc).to(DEV); opt=torch.optim.Adam(m.parameters(),1e-3); ce=nn.CrossEntropyLoss()
    gg=torch.Generator(device=DEV).manual_seed(SEED)
    for _ in range(epochs):
        m.train(); perm=tr[torch.randperm(len(tr),generator=gg,device=DEV)]
        for k in range(0,len(perm),8):
            b=perm[k:k+8]; opt.zero_grad(); loss=ce(m(Xt[b]),Yt[b]); loss.backward(); opt.step()
    m.eval(); return m

def miou(model, region, task):
    name,nc,_=task; X,Yraw=load_raw(region)
    if X is None: return None
    Y=make_labels(Yraw,task); Xt=torch.from_numpy(X)
    inter=np.zeros(nc); union=np.zeros(nc)
    with torch.no_grad():
        for k in range(0,len(Xt),16):
            p=model(Xt[k:k+16].to(DEV)).argmax(1).cpu().numpy(); y=Y[k:k+16]
            for c in range(nc):
                inter[c]+=np.logical_and(p==c,y==c).sum(); union[c]+=np.logical_or(p==c,y==c).sum()
    if nc==2:   # report positive-class IoU for binary
        return float(inter[1]/union[1]) if union[1]>0 else None
    v=union>0; return float(np.mean(inter[v]/union[v])) if v.any() else None

def run_task(task):
    name=task[0]
    models={r:train(r,task) for r in REGIONS}; models={r:m for r,m in models.items() if m is not None}
    if len(models)<4: return None
    rr=list(models)
    mat={s:{t:miou(models[s],t,task) for t in rr} for s in models}
    diag=np.mean([mat[r][r] for r in models if mat[r].get(r) is not None])
    off=np.mean([mat[s][t] for s in models for t in rr if s!=t and mat[s].get(t) is not None])
    srcs=[s for s in models if all(mat[s].get(t) is not None for t in rr)]
    M=np.array([[mat[s][t] for t in rr] for s in srcs])
    inv=np.mean([spearmanr(M[a],M[b]).correlation for a in range(len(srcs)) for b in range(a+1,len(srcs))]) if len(srcs)>1 else float("nan")
    return {"task":name,"n_sources":len(models),"in_region_miou":round(float(diag),3),
            "out_region_miou":round(float(off),3),"home_field_gap":round(float(diag-off),4),
            "source_invariance":round(float(inv),3)}

def main():
    out=[]
    for task in TASKS:
        print(f"[task {task[0]}] running 7x7 matrix ...", flush=True)
        r=run_task(task)
        if r: out.append(r); print(f"  {r}", flush=True)
    # add the built-up-nowcasting point (the easy temporal task) from the existing matrix
    out.append({"task":"builtup_nowcast(temporal)","n_sources":9,"in_region_miou":None,
                "out_region_miou":None,"home_field_gap":0.001,"source_invariance":0.94})
    # transition: source-invariance vs difficulty (in-region mIoU; lower=harder)
    seg=[r for r in out if r["in_region_miou"] is not None]
    if len(seg)>=3:
        diff=[r["in_region_miou"] for r in seg]; inv=[r["source_invariance"] for r in seg]
        gap=[r["home_field_gap"] for r in seg]
        rho_inv=spearmanr(diff,inv).correlation; rho_gap=spearmanr(diff,gap).correlation
        print(f"\n=== DIFFICULTY TRANSITION (n={len(seg)} seg tasks) ===")
        print(f"Spearman(in-region mIoU [easier->], source-invariance) = {rho_inv:+.2f}  (expect >0: easier=more invariant)")
        print(f"Spearman(in-region mIoU, home-field gap) = {rho_gap:+.2f}  (expect <0: easier=smaller gap)")
    json.dump(out, open(REPO/"results/metrics/multitask_difficulty.json","w"), indent=1)
    print("\nsaved results/metrics/multitask_difficulty.json")
    print(f"{'task':28}{'in_mIoU':>8}{'gap':>8}{'src_inv':>9}")
    for r in out: print(f"{r['task']:28}{str(r['in_region_miou']):>8}{r['home_field_gap']:>+8.3f}{r['source_invariance']:>9.3f}")
if __name__=="__main__": main()
