"""Frozen foundation-model segmentation transfer matrix.
Does a globally-pretrained FM encoder reduce the imagery-input source-
dependence we saw with from-scratch U-Nets? We freeze a DINOv2 ViT-S encoder
(RGB, a strong general FM; GeoCrossBench reports DINOv3 ~ satellite GFMs),
train only a small decoder per source region on built-up binary segmentation,
and read the home-field gap of the 7x7 matrix. Compare to the from-scratch
U-Net gap of +0.126 on the same task: a smaller gap => pretraining reduces
source-dependence (nuances the law); a similar gap => the law holds for FMs too.
"""
import json, glob
from pathlib import Path
import numpy as np, torch, torch.nn as nn, timm, warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
REPO=Path(__file__).resolve().parents[2]; DEV=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
REGIONS=["south_asia","ssa","east_asia","andes","mena","eeca","oceania"]; SEED=20260525
IMA=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); IMS=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
ENC=timm.create_model("vit_small_patch14_dinov2.lvd142m",pretrained=True,num_classes=0,img_size=128).to(DEV).eval()
for p in ENC.parameters(): p.requires_grad=False

def load_region(region):
    Xs,Ys=[],[]
    for f in glob.glob(str(REPO/f"data/hardtask/{region}/*/patches.npz")):
        d=np.load(f); Xs.append(d["s2"].astype(np.float32)); Ys.append((d["label"]==50).astype(np.int64))
    if not Xs: return None,None
    X=np.concatenate(Xs)[:, :3]                       # B04,B03,B02 = R,G,B
    X=np.clip(X/3000.0,0,1)                            # reflectance -> [0,1]
    return torch.from_numpy(X), torch.from_numpy(np.concatenate(Ys))

def feats(x):                                         # x: (B,3,128,128) [0,1]
    x=((x-IMA.to(DEV))/IMS.to(DEV))
    with torch.no_grad():
        f=ENC.forward_features(x)[:,1:,:]             # drop CLS -> (B,81,384)
    s=int(f.shape[1]**0.5)
    return f.transpose(1,2).reshape(f.shape[0],-1,s,s).contiguous()  # (B,384,9,9)

class Dec(nn.Module):
    def __init__(s,c=384):
        super().__init__()
        s.net=nn.Sequential(nn.Conv2d(c,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
                            nn.Upsample(scale_factor=2),nn.Conv2d(128,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
                            nn.Conv2d(64,2,1))
    def forward(s,f): return nn.functional.interpolate(s.net(f),size=128,mode="bilinear",align_corners=False)

def train(region,epochs=12):
    X,Y=load_region(region)
    if X is None or len(X)<20: return None
    torch.manual_seed(SEED); tr=torch.randperm(len(X),generator=torch.Generator().manual_seed(SEED))[:int(0.85*len(X))]
    dec=Dec().to(DEV); opt=torch.optim.Adam(dec.parameters(),1e-3); ce=nn.CrossEntropyLoss()
    for _ in range(epochs):
        dec.train(); perm=tr[torch.randperm(len(tr))]
        for k in range(0,len(perm),16):
            b=perm[k:k+16]; xb=X[b].to(DEV); yb=Y[b].to(DEV)
            opt.zero_grad(); loss=ce(dec(feats(xb)),yb); loss.backward(); opt.step()
    dec.eval(); return dec

def iou(dec,region):
    X,Y=load_region(region)
    if X is None: return None
    I=U=0.0
    with torch.no_grad():
        for k in range(0,len(X),16):
            p=dec(feats(X[k:k+16].to(DEV))).argmax(1).cpu().numpy(); y=Y[k:k+16].numpy()
            I+=np.logical_and(p==1,y==1).sum(); U+=np.logical_or(p==1,y==1).sum()
    return float(I/U) if U>0 else None

def main():
    models={r:train(r) for r in REGIONS}; models={r:m for r,m in models.items() if m is not None}
    rr=list(models); mat={s:{t:iou(models[s],t) for t in rr} for s in models}
    diag=np.mean([mat[r][r] for r in models if mat[r].get(r) is not None])
    off=np.mean([mat[s][t] for s in models for t in rr if s!=t and mat[s].get(t) is not None])
    srcs=[s for s in models if all(mat[s].get(t) is not None for t in rr)]
    M=np.array([[mat[s][t] for t in rr] for s in srcs])
    inv=np.mean([spearmanr(M[a],M[b]).correlation for a in range(len(srcs)) for b in range(a+1,len(srcs))])
    print(f"=== FROZEN-FM (DINOv2) built-up segmentation, 7x7 ===")
    print(f"in-region IoU={diag:.3f}  out-of-region={off:.3f}  home-field gap={diag-off:+.4f}  source-inv={inv:.3f}")
    print(f"from-scratch U-Net same task: home-field gap=+0.126")
    print(f"=> smaller gap => pretraining reduces imagery source-dependence (nuances law); similar => law holds for FMs")
    json.dump({"encoder":"DINOv2-ViT-S frozen","task":"built-up binary seg","in_region":round(float(diag),4),
               "out_region":round(float(off),4),"home_field_gap":round(float(diag-off),4),"source_inv":round(float(inv),3),
               "fromscratch_unet_gap":0.126}, open(REPO/"results/metrics/fm_seg_matrix.json","w"),indent=1)
    print("saved results/metrics/fm_seg_matrix.json")
if __name__=="__main__": main()
