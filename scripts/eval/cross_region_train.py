"""Train a from-scratch model on ONE source region's tiles, for the
cross-region transfer matrix. Reuses the eval pipeline's rasters/normalisation
so source-trained and CONUS-trained models are directly comparable.

Usage:
    python -m scripts.eval.cross_region_train --source east_asia --arch cnn
Outputs: results/transfer_matrix/weights/<source>/best_<arch>_3ch.pth
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from scripts.acquire.regions import city_by_name
from scripts.eval.cross_region_eval import (
    TARGET_EPOCH,
    TILE_PX,
    build_input_tensor,
    load_city_rasters,
)
from scripts.eval.models import ConvLSTMModel, SimpleCNN, SimpleUNet

REPO = Path(__file__).resolve().parents[2]
ARCH = {"cnn": SimpleCNN, "unet": SimpleUNet, "convlstm": ConvLSTMModel}
CKPT = {"cnn": "best_cnn_3ch.pth", "unet": "best_unet_3ch.pth", "convlstm": "best_3ch_mc_model.pth"}


def gather_tiles(source, stride: int = 64):
    """Sliding-window (input, target) tiles. `source` is a region name or an
    iterable of region names (their union, for multi-source/LORO training)."""
    sources = [source] if isinstance(source, str) else list(source)
    X, Y = [], []
    for s in sources:
        for f in glob.glob(str(REPO / f"data/processed/{s}/*/builtup_2015.tif")):
            city = city_by_name(Path(f).parent.name)
            try:
                bu, vol, pop, _, _ = load_city_rasters(city, REPO / "data/processed")
            except Exception as e:
                print(f"  [skip {s}/{getattr(city, 'name', '?')}] unreadable raster "
                      f"({type(e).__name__})", flush=True)
                continue
            H, W = bu[TARGET_EPOCH].shape
            for i in range(0, H - TILE_PX + 1, stride):
                for j in range(0, W - TILE_PX + 1, stride):
                    tgt = bu[TARGET_EPOCH][i:i+TILE_PX, j:j+TILE_PX]
                    if tgt.mean() < 0.005:      # skip near-empty tiles
                        continue
                    X.append(build_input_tensor(bu, vol, pop, i, j)[0])  # (8,3,H,W)
                    Y.append(torch.from_numpy(tgt.astype(np.float32))[None])
    return torch.stack(X), torch.stack(Y)


def soft_jaccard(pred, tgt, eps=1e-6):
    inter = (pred * tgt).sum((1, 2, 3))
    union = (pred + tgt - pred * tgt).sum((1, 2, 3))
    return 1 - ((inter + eps) / (union + eps)).mean()


def fit(X, Y, arch, epochs, seed, device, tag=""):
    """The matrix's exact training recipe, factored out so the multi-seed and
    leave-one-region-out drivers cannot drift from `main`: batch 16, full epoch
    budget, no early stopping, seeded split + shuffle, MSE + 0.5*soft-Jaccard,
    ReduceLROnPlateau. Returns the lowest-validation state_dict (on CPU)."""
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(X); idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, va = idx[: int(0.85 * n)], idx[int(0.85 * n):]
    X = X.to(device); Y = Y.to(device); tr = tr.to(device); va = va.to(device)
    model = ARCH[arch]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = 1e9; best_state = None; bs = 16
    gen = torch.Generator(device=tr.device).manual_seed(seed)
    for ep in range(epochs):
        model.train(); perm = tr[torch.randperm(len(tr), generator=gen, device=tr.device)]
        for k in range(0, len(perm), bs):
            b = perm[k:k+bs]
            opt.zero_grad()
            pred = model(X[b])
            loss = nn.functional.mse_loss(pred, Y[b]) + 0.5 * soft_jaccard(pred, Y[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vls = []
            for k in range(0, len(va), bs):
                vb = va[k:k+bs]; vp = model(X[vb])
                vls.append((nn.functional.mse_loss(vp, Y[vb]) + 0.5 * soft_jaccard(vp, Y[vb])).item())
            vl = float(np.mean(vls))
        sched.step(vl)
        if vl < best:
            best = vl; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  {tag}ep{ep:02d} val={vl:.4f} best={best:.4f}", flush=True)
    return best_state, best


def main(source, arch, epochs, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    X, Y = gather_tiles(source)
    print(f"[{source}/{arch}] {len(X)} training tiles", flush=True)
    n = len(X); idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr, va = idx[: int(0.85 * n)], idx[int(0.85 * n):]
    # Keep the whole (small) dataset resident on the GPU — avoids per-batch
    # CPU->MPS copies, the dominant cost for these tiny models.
    X = X.to(device); Y = Y.to(device); tr = tr.to(device); va = va.to(device)
    model = ARCH[arch]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best = 1e9; out = REPO / f"results/transfer_matrix/weights/{source}"; out.mkdir(parents=True, exist_ok=True)
    # IDENTICAL recipe for every (source, arch): batch 16, full `epochs`, no
    # early stopping, seeded shuffle. GPU-residency is the only speed change and
    # does not alter the math. This keeps the transfer matrix free of any
    # trainer-version confound between sources.
    bs = 16
    gen = torch.Generator(device=tr.device).manual_seed(seed)
    for ep in range(epochs):
        model.train(); perm = tr[torch.randperm(len(tr), generator=gen, device=tr.device)]
        for k in range(0, len(perm), bs):
            b = perm[k:k+bs]
            opt.zero_grad()
            pred = model(X[b])
            loss = nn.functional.mse_loss(pred, Y[b]) + 0.5 * soft_jaccard(pred, Y[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():   # chunk validation to bound memory
            vls = []
            for k in range(0, len(va), bs):
                vb = va[k:k+bs]; vp = model(X[vb])
                vls.append((nn.functional.mse_loss(vp, Y[vb]) + 0.5 * soft_jaccard(vp, Y[vb])).item())
            vl = float(np.mean(vls))
        sched.step(vl)
        if vl < best:
            best = vl; torch.save(model.state_dict(), out / CKPT[arch])
        print(f"  ep{ep:02d} val={vl:.4f} best={best:.4f}", flush=True)
    print(f"[{source}/{arch}] done, best val={best:.4f} -> {out/CKPT[arch]}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--arch", default="cnn", choices=list(ARCH))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260525)
    a = p.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    main(a.source, a.arch, a.epochs, a.seed, dev)
