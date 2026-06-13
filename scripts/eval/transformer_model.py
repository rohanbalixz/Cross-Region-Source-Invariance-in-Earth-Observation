"""Divided space-time attention model (TimeSformer-style) for built-up
nowcasting on the same 8-epoch input / 1-epoch target protocol as the
existing CNN / U-Net / ConvLSTM baselines.

Architecture (matched in parameter count to SimpleUNet at ~500k params):

    Input  (B, T=8, C=3, H=128, W=128)
      -> per-timestep patch embed (16x16 patches -> 64 spatial tokens of dim 64)
      -> add separate temporal (T=8) and spatial (8x8) positional embeddings
      -> 4x { divided temporal attention; divided spatial attention; MLP }
      -> take last-timestep tokens, fold back to (64, 8, 8), upsample 16x
      -> 1x1 conv to (1, 128, 128), sigmoid
    Output (B, 1, 128, 128)

The "divided" space-time attention factorisation (Bertasius et al., 2021)
keeps the attention quadratic in T+HW instead of (T*HW)^2, which is the
load-bearing trick for running on MPS at our tile size.

Training entry point:  python -m scripts.eval.transformer_model train
Loading checkpoint:    use `load_transformer(weights_root)` mirroring
                        `scripts.eval.models.load_models`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PATCH = 16
TILE = 128
N_PATCHES = (TILE // PATCH) ** 2  # 64
DIM = 64
N_HEADS = 4
N_LAYERS = 4
T = 8


class DividedAttentionBlock(nn.Module):
    """One divided space-time block: temporal attn, then spatial attn, then MLP."""

    def __init__(self, dim: int = DIM, n_heads: int = N_HEADS):
        super().__init__()
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # x: (B, T, N, D)
        B, Tt, N, D = x.shape
        # Temporal attention: each spatial token attends across time.
        xt = x.permute(0, 2, 1, 3).reshape(B * N, Tt, D)
        xt = xt + self.attn_t(self.norm_t(xt), self.norm_t(xt), self.norm_t(xt))[0]
        xt = xt.reshape(B, N, Tt, D).permute(0, 2, 1, 3)  # back to (B,T,N,D)
        # Spatial attention: each temporal slice attends across space.
        xs = xt.reshape(B * Tt, N, D)
        xs = xs + self.attn_s(self.norm_s(xs), self.norm_s(xs), self.norm_s(xs))[0]
        xs = xs.reshape(B, Tt, N, D)
        # MLP.
        return xs + self.mlp(self.norm_m(xs))


class TimeSformer(nn.Module):
    def __init__(self, in_channels: int = 3, n_layers: int = N_LAYERS):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, DIM, kernel_size=PATCH, stride=PATCH)
        # Separable temporal + spatial positional embeddings.
        self.pos_t = nn.Parameter(torch.zeros(1, T, 1, DIM))
        self.pos_s = nn.Parameter(torch.zeros(1, 1, N_PATCHES, DIM))
        nn.init.trunc_normal_(self.pos_t, std=0.02)
        nn.init.trunc_normal_(self.pos_s, std=0.02)
        self.blocks = nn.ModuleList(
            [DividedAttentionBlock() for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(DIM)
        # Decoder: tokens (DIM at 8x8 spatial) -> upsample 16x to TILExTILE.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(DIM, 32, 4, stride=2, padding=1),  # 16x16
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),   # 32x32
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),    # 64x64
            nn.GELU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),     # 128x128
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, Tt, C, H, W = x.shape
        # Patch embed each timestep.
        x = x.reshape(B * Tt, C, H, W)
        tokens = self.patch_embed(x)        # (B*T, DIM, 8, 8)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B*T, N, DIM)
        tokens = tokens.reshape(B, Tt, N_PATCHES, DIM)
        tokens = tokens + self.pos_t + self.pos_s
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        # Take last timestep tokens (the temporal-context-aware encoding for
        # the prediction year).
        last = tokens[:, -1]                # (B, N, DIM)
        H8 = W8 = int(N_PATCHES ** 0.5)
        last = last.transpose(1, 2).reshape(B, DIM, H8, W8)  # (B, DIM, 8, 8)
        return self.decoder(last)            # (B, 1, 128, 128)


# --------------------------------------------------------------------------- #
#                              loader for eval                                  #
# --------------------------------------------------------------------------- #


def load_transformer(weights_root: Path):
    """Returns {'transformer': model} if checkpoint exists, else {}."""
    weights_root = Path(weights_root)
    ckpt = weights_root / "best_transformer.pth"
    if not ckpt.exists():
        return {}
    model = TimeSformer()
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    return {"transformer": model}


# --------------------------------------------------------------------------- #
#                                  training                                    #
# --------------------------------------------------------------------------- #


def build_input_arr(bu, vol, pop, i: int, j: int, train_epochs, tile_px: int) -> np.ndarray:
    frames = []
    for yr in train_epochs:
        frames.append(np.stack([
            bu[yr][i:i + tile_px, j:j + tile_px],
            vol[yr][i:i + tile_px, j:j + tile_px],
            pop[yr][i:i + tile_px, j:j + tile_px],
        ], axis=0))
    return np.stack(frames, axis=0).astype(np.float32)


class CONUSTileDataset(torch.utils.data.Dataset):
    def __init__(self, bu, vol, pop, train_epochs, target_epoch,
                 tile_indices, tile_px: int):
        self.bu = bu
        self.vol = vol
        self.pop = pop
        self.train_epochs = train_epochs
        self.target_epoch = target_epoch
        self.tile_indices = tile_indices
        self.tile_px = tile_px

    def __len__(self):
        return len(self.tile_indices)

    def __getitem__(self, idx):
        i, j = self.tile_indices[idx]
        x = build_input_arr(self.bu, self.vol, self.pop, i, j,
                            self.train_epochs, self.tile_px)
        y = self.bu[self.target_epoch][i:i + self.tile_px, j:j + self.tile_px]
        y = (y >= 0.01).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y[None])


def train_transformer(
    geotiff_root: Path, out_path: Path,
    n_train_tiles: int, n_val_tiles: int, batch_size: int, n_epochs: int,
    lr: float, seed: int, device: str,
) -> None:
    from scripts.eval.cross_region_eval import TRAIN_EPOCHS, TARGET_EPOCH
    from scripts.common import TILE_PX, TILE_STRIDE, TILE_BUILTUP_THRESHOLD
    from scripts.eval.conus_baseline import load_conus_rasters

    print(f"loading CONUS rasters from {geotiff_root} ...", flush=True)
    bu, vol, pop, _, _ = load_conus_rasters(geotiff_root)
    print(f"  grid: {bu[TARGET_EPOCH].shape}", flush=True)

    # Enumerate valid tiles (have built-up in target year).
    h, w = bu[TARGET_EPOCH].shape
    valid = []
    gt = bu[TARGET_EPOCH]
    print("enumerating valid tiles ...", flush=True)
    for i in range(0, h - TILE_PX + 1, TILE_STRIDE):
        for j in range(0, w - TILE_PX + 1, TILE_STRIDE):
            patch = gt[i:i + TILE_PX, j:j + TILE_PX]
            if patch.mean() > TILE_BUILTUP_THRESHOLD:
                valid.append((i, j))
    print(f"  {len(valid)} valid tiles", flush=True)

    rng = np.random.default_rng(seed)
    rng.shuffle(valid)
    train_idx = valid[:n_train_tiles]
    val_idx = valid[n_train_tiles:n_train_tiles + n_val_tiles]
    print(f"  train: {len(train_idx)}, val: {len(val_idx)}", flush=True)

    train_ds = CONUSTileDataset(bu, vol, pop, TRAIN_EPOCHS, TARGET_EPOCH, train_idx, TILE_PX)
    val_ds = CONUSTileDataset(bu, vol, pop, TRAIN_EPOCHS, TARGET_EPOCH, val_idx, TILE_PX)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

    dev = torch.device(device)
    model = TimeSformer().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TimeSformer params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    best_val = float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_v = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()
                n_v += 1
        val_loss /= max(n_v, 1)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict()}, str(out_path))
            marker = "  ** saved"
        print(f"epoch {epoch+1:>2}/{n_epochs}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}{marker}", flush=True)

    print(f"\nbest val loss: {best_val:.4f}  checkpoint: {out_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--geotiff-root", type=Path,
                   default=Path("data/processed/conus").resolve())
    t.add_argument("--out", type=Path,
                   default=Path("../models/best_transformer.pth").resolve())
    t.add_argument("--n-train-tiles", type=int, default=4000)
    t.add_argument("--n-val-tiles", type=int, default=400)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--n-epochs", type=int, default=12)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", type=str,
                   default="mps" if torch.backends.mps.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        train_transformer(
            args.geotiff_root, args.out,
            args.n_train_tiles, args.n_val_tiles, args.batch_size,
            args.n_epochs, args.lr, args.seed, args.device,
        )


if __name__ == "__main__":
    main()
