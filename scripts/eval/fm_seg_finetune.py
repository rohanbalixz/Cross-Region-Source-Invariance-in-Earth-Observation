"""Fine-tuned foundation-model segmentation transfer matrix does fine-tuning,
not just a frozen encoder, close the imagery cross-region gap?

Same protocol as the frozen control in `fm_seg_matrix.py` -- DINOv2 ViT-S, RGB
Sentinel-2, built-up binary segmentation, 7x7 source-by-target matrix -- but here
we UNFREEZE the last N transformer blocks (+ final norm) of the encoder and
fine-tune them end-to-end with the decoder, one model per source region. We then
read retention (out/in IoU) and the home-field gap and compare to the frozen
encoder (retention 0.78) and the from-scratch U-Net (0.82). If fine-tuned
retention stays ~0.8 rather than rising to ~1.0, the imagery penalty is not a
frozen-encoder artefact -- fine-tuning a 142M-image-pretrained ViT does not buy
back the lost fifth.

Usage: python -m scripts.eval.fm_seg_finetune --n-unfreeze 4 --epochs 12
"""
import argparse, copy, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
import scripts.eval.fm_seg_matrix as fz   # reuses ENC (pretrained), load_region, Dec, IMA, IMS, REGIONS, SEED

REPO = fz.REPO; DEV = fz.DEV; REGIONS = fz.REGIONS; SEED = fz.SEED


def feats_grad(enc, x):
    """Encoder features WITH gradients (the frozen module's feats() uses no_grad)."""
    x = (x - fz.IMA.to(DEV)) / fz.IMS.to(DEV)
    f = enc.forward_features(x)[:, 1:, :]            # drop CLS -> (B,81,384)
    s = int(f.shape[1] ** 0.5)
    return f.transpose(1, 2).reshape(f.shape[0], -1, s, s).contiguous()


def unfreeze_last_n(enc, n):
    for p in enc.parameters():
        p.requires_grad = False
    for blk in enc.blocks[len(enc.blocks) - n:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in enc.norm.parameters():
        p.requires_grad = True


def train_ft(region, n_unfreeze, epochs):
    X, Y = fz.load_region(region)
    if X is None or len(X) < 20:
        return None
    enc = copy.deepcopy(fz.ENC).to(DEV); unfreeze_last_n(enc, n_unfreeze)
    dec = fz.Dec().to(DEV)
    torch.manual_seed(SEED)
    tr = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))[:int(0.85 * len(X))]
    opt = torch.optim.AdamW(
        [{"params": [p for p in enc.parameters() if p.requires_grad], "lr": 1e-5},
         {"params": dec.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        enc.train(); dec.train(); perm = tr[torch.randperm(len(tr))]
        for k in range(0, len(perm), 16):
            b = perm[k:k+16]; xb = X[b].to(DEV); yb = Y[b].to(DEV)
            opt.zero_grad(); loss = ce(dec(feats_grad(enc, xb)), yb); loss.backward(); opt.step()
    enc.eval(); dec.eval()
    return (enc, dec)


def iou_ft(model, region):
    enc, dec = model
    X, Y = fz.load_region(region)
    if X is None:
        return None
    I = U = 0.0
    with torch.no_grad():
        for k in range(0, len(X), 16):
            p = dec(feats_grad(enc, X[k:k+16].to(DEV))).argmax(1).cpu().numpy(); y = Y[k:k+16].numpy()
            I += np.logical_and(p == 1, y == 1).sum(); U += np.logical_or(p == 1, y == 1).sum()
    return float(I / U) if U > 0 else None


def main(n_unfreeze, epochs):
    models = {r: train_ft(r, n_unfreeze, epochs) for r in REGIONS}
    models = {r: m for r, m in models.items() if m is not None}
    rr = list(models)
    print(f"fine-tuned sources: {rr}", flush=True)
    mat = {s: {t: iou_ft(models[s], t) for t in rr} for s in models}
    diag = np.mean([mat[r][r] for r in models if mat[r].get(r) is not None])
    off = np.mean([mat[s][t] for s in models for t in rr if s != t and mat[s].get(t) is not None])
    srcs = [s for s in models if all(mat[s].get(t) is not None for t in rr)]
    M = np.array([[mat[s][t] for t in rr] for s in srcs])
    inv = np.mean([spearmanr(M[a], M[b]).correlation
                   for a in range(len(srcs)) for b in range(a + 1, len(srcs))])
    retention = off / diag if diag else None
    print(f"=== FINE-TUNED DINOv2 (last {n_unfreeze} blocks) built-up seg, {len(rr)}x{len(rr)} ===")
    print(f"in-region IoU={diag:.3f}  out-of-region={off:.3f}  gap={diag-off:+.4f}  "
          f"retention={retention:.3f}  source-inv={inv:.3f}", flush=True)
    print(f"compare: frozen DINOv2 retention 0.78 (gap +0.142), from-scratch U-Net retention 0.82 (gap +0.126)")
    json.dump({"encoder": f"DINOv2-ViT-S fine-tuned (last {n_unfreeze} blocks)",
               "task": "built-up binary seg", "n_unfreeze_blocks": n_unfreeze, "epochs": epochs,
               "in_region": round(float(diag), 4), "out_region": round(float(off), 4),
               "home_field_gap": round(float(diag - off), 4), "retention": round(float(retention), 3),
               "source_inv": round(float(inv), 3), "matrix": mat,
               "frozen_retention": 0.78, "fromscratch_unet_retention": 0.82},
              open(REPO / "results/metrics/fm_seg_finetuned.json", "w"), indent=1)
    print("saved results/metrics/fm_seg_finetuned.json", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-unfreeze", type=int, default=4)
    p.add_argument("--epochs", type=int, default=12)
    a = p.parse_args()
    main(a.n_unfreeze, a.epochs)
