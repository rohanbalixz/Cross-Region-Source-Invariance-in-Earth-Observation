"""Prithvi-EO-1.0-100M MAE encoder, re-implemented in PyTorch to load
the public IBM-NASA checkpoint at
`ibm-nasa-geospatial/Prithvi-EO-1.0-100M:Prithvi_EO_V1_100M.pt`.

Architecture (inferred directly from the state_dict shapes):

  Input:  (B, C=6 bands, T=3 timesteps, H=224, W=224)
          The 6 bands are the HLS surface-reflectance subset:
          B02 (blue), B03 (green), B04 (red), B8A (NIR), B11 (SWIR1),
          B12 (SWIR2) -- the same channel order Prithvi pretraining used.
  Patch:  (1, 16, 16) spatial-temporal patch via Conv3d
          -> 588 patches per input (T*Hp*Wp = 3*14*14)
  Tokens: 588 patch tokens + 1 [CLS] = 589, embed_dim 768
  Encoder: 12 transformer blocks (12 heads, MLP ratio 4), MAE-style
          (decoder dropped; we only need encoder features for transfer).

For downstream built-up segmentation, we attach a lightweight CNN head
that upsamples the 14x14 patch grid back to the original tile size and
predicts per-pixel probability of built-up. The encoder stays frozen
during head training so the comparison to the other four architectures
is "what does Prithvi's pre-trained representation predict, given a
linear-probe-style head trained on CONUS?"

Run forward-pass smoke test:
    python -m scripts.eval.prithvi_encoder
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Architecture constants matched to the public Prithvi-EO-1.0-100M checkpoint
PRITHVI_BANDS = 6           # HLS B02/B03/B04/B8A/B11/B12
PRITHVI_FRAMES = 3          # 3 temporal timesteps
PRITHVI_IMG_SIZE = 224
PRITHVI_PATCH_SIZE = 16     # spatial patch
PRITHVI_TUBELET = 1         # temporal patch (each frame is its own patch in T)
PRITHVI_EMBED_DIM = 768
PRITHVI_DEPTH = 12
PRITHVI_HEADS = 12
PRITHVI_MLP_RATIO = 4.0


def _grid_count(img_size=PRITHVI_IMG_SIZE, patch=PRITHVI_PATCH_SIZE,
                frames=PRITHVI_FRAMES, tubelet=PRITHVI_TUBELET):
    return (frames // tubelet) * (img_size // patch) ** 2


# --------------------------------------------------------------------------- #
#                         Vanilla ViT block components                          #
# --------------------------------------------------------------------------- #


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, H, N, Dh)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
#                              Prithvi encoder                                  #
# --------------------------------------------------------------------------- #


class PrithviPatchEmbed3D(nn.Module):
    """Conv3d patch embedding: input (B, C, T, H, W) -> tokens (B, N, D)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(
            PRITHVI_BANDS, PRITHVI_EMBED_DIM,
            kernel_size=(PRITHVI_TUBELET, PRITHVI_PATCH_SIZE, PRITHVI_PATCH_SIZE),
            stride=(PRITHVI_TUBELET, PRITHVI_PATCH_SIZE, PRITHVI_PATCH_SIZE),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)                       # (B, D, T', Hp, Wp)
        return x.flatten(2).transpose(1, 2)    # (B, N, D)


class PrithviEncoder(nn.Module):
    """Prithvi MAE encoder. Loadable from Prithvi_EO_V1_100M.pt by
    renaming `encoder.*` keys with `_load_official_checkpoint`."""

    def __init__(self):
        super().__init__()
        self.patch_embed = PrithviPatchEmbed3D()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, PRITHVI_EMBED_DIM))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + _grid_count(), PRITHVI_EMBED_DIM),
            requires_grad=False,   # MAE uses sin-cos pos embed, fixed
        )
        self.blocks = nn.ModuleList(
            [Block(PRITHVI_EMBED_DIM, PRITHVI_HEADS, PRITHVI_MLP_RATIO)
             for _ in range(PRITHVI_DEPTH)]
        )
        self.norm = nn.LayerNorm(PRITHVI_EMBED_DIM)

    def forward(self, x):
        # x: (B, C=6, T=3, H=224, W=224)
        tokens = self.patch_embed(x)                       # (B, 588, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)    # (B, 1, D)
        x = torch.cat([cls, tokens], dim=1)                # (B, 589, D)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)                                 # (B, 589, D)


def load_prithvi_encoder(checkpoint_path: Path,
                          device: str = "cpu") -> PrithviEncoder:
    """Build PrithviEncoder and populate from the official checkpoint.
    The checkpoint stores encoder weights under `encoder.*`; we strip
    the prefix. Decoder weights are ignored.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"unexpected checkpoint type: {type(ckpt)}")

    encoder_state = {}
    for k, v in ckpt.items():
        if not k.startswith("encoder."):
            continue
        new_k = k[len("encoder."):]
        # The official code uses `norm` as the final layer norm, which we
        # have under the same name. blocks.N.* maps 1:1.
        encoder_state[new_k] = v

    model = PrithviEncoder()
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    if unexpected:
        print(f"warning: unexpected keys in checkpoint (will be ignored): {unexpected[:5]}"
              f" (and {len(unexpected)-5} more)" if len(unexpected) > 5 else
              f"warning: unexpected keys: {unexpected}", flush=True)
    if missing:
        # Some keys may be MAE-decoder-only (not in our encoder); these
        # are listed as "missing" because we declared them in our model.
        # But since we only declared encoder layers, missing keys here
        # would be a real load failure.
        truly_missing = [k for k in missing if not k.startswith("decoder")]
        if truly_missing:
            print(f"warning: missing keys in our model: {truly_missing[:5]}"
                  f" (and {len(truly_missing)-5} more)" if len(truly_missing) > 5 else
                  f"warning: missing keys: {truly_missing}", flush=True)
    model.eval()
    model.to(device)
    return model


# --------------------------------------------------------------------------- #
#                       Segmentation head (built-up)                            #
# --------------------------------------------------------------------------- #


class PrithviBuiltupHead(nn.Module):
    """Map Prithvi encoder tokens to a built-up probability map.

    Input: (B, 589, 768) = [cls] + 588 patch tokens (3 timesteps * 14 * 14)
    We average over the temporal axis to get a single (Hp=14, Wp=14, D)
    token grid (matching the spatial layout that the bound's covariate
    space is anchored on), then upsample 16x to the original 224x224.
    """

    def __init__(self, embed_dim: int = PRITHVI_EMBED_DIM,
                 hidden: int = 64):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, 32, 4, stride=2, padding=1),  # 28x28
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),       # 56x56
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),        # 112x112
            nn.GELU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),         # 224x224
            nn.Sigmoid(),
        )

    def forward(self, tokens):
        # tokens: (B, 589, D)  -> drop cls -> (B, 588, D)
        x = tokens[:, 1:, :]
        B, N, D = x.shape
        Hp = Wp = int(math.sqrt(N // PRITHVI_FRAMES))
        # Reshape to (B, T, Hp, Wp, D), mean over T to collapse temporal
        x = x.reshape(B, PRITHVI_FRAMES, Hp, Wp, D).mean(dim=1)
        x = self.proj(x)                                # (B, Hp, Wp, hidden)
        x = x.permute(0, 3, 1, 2).contiguous()          # (B, hidden, Hp, Wp)
        return self.decoder(x)                          # (B, 1, 224, 224)


class PrithviSegmentationModel(nn.Module):
    """Frozen-Prithvi-encoder + trainable built-up head; matches the
    `(B, C, T, H, W) -> (B, 1, H, W)` interface used by the other
    cross_region_eval models."""

    def __init__(self, encoder: PrithviEncoder, head: PrithviBuiltupHead):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = head

    def forward(self, x):
        # x: (B, C=6 bands, T=3, 224, 224)
        with torch.no_grad():
            tokens = self.encoder(x)
        return self.head(tokens)                        # (B, 1, 224, 224)


# --------------------------------------------------------------------------- #
#                                smoke test                                     #
# --------------------------------------------------------------------------- #


def _smoke_test():
    ckpt_path = Path(__file__).resolve().parents[2].parent / "models" / "foundation" / "Prithvi_EO_V1_100M.pt"
    print(f"loading checkpoint from {ckpt_path} ...")
    enc = load_prithvi_encoder(ckpt_path)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"encoder params: {n_params:,}")

    x = torch.randn(2, PRITHVI_BANDS, PRITHVI_FRAMES, PRITHVI_IMG_SIZE, PRITHVI_IMG_SIZE)
    with torch.no_grad():
        tok = enc(x)
    print(f"encoder forward: input {tuple(x.shape)} -> tokens {tuple(tok.shape)}")
    assert tok.shape == (2, 1 + _grid_count(), PRITHVI_EMBED_DIM), tok.shape

    head = PrithviBuiltupHead()
    n_head = sum(p.numel() for p in head.parameters())
    print(f"head params: {n_head:,}")
    with torch.no_grad():
        y = head(tok)
    print(f"head forward: tokens {tuple(tok.shape)} -> seg {tuple(y.shape)}, "
          f"range [{y.min().item():.3f}, {y.max().item():.3f}]")

    full = PrithviSegmentationModel(enc, head)
    n_total = sum(p.numel() for p in full.parameters())
    n_trainable = sum(p.numel() for p in full.parameters() if p.requires_grad)
    print(f"full model: total {n_total:,} params, trainable {n_trainable:,}")


if __name__ == "__main__":
    _smoke_test()
