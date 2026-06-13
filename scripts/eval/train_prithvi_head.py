"""Train a built-up segmentation head on top of a frozen Prithvi-EO-1.0
encoder, using CONUS Landsat-8 patches as input and CONUS GHSL built-up
2015 as the target.

Protocol design choices:

  - **Encoder is frozen.** This is a linear-probe-style evaluation: how
    well does the pretrained representation predict downstream built-up,
    given only a small head trained on the source data? This mirrors
    the standard FM evaluation protocol used by e.g. SatMAE,
    Prithvi-Sen1Floods11, etc., and isolates the question "does the
    pretrained representation know about built-up structure?".
  - **L2 surface reflectance scaling**: Landsat C2 L2 raw uint16 values
    are converted to reflectance via `raw * 2.75e-5 - 0.2` before being
    fed to Prithvi (matching the HLS preprocessing Prithvi expects).
    Zero pixels (cloud/nodata) are left at -0.2 (the offset).
  - **Target alignment**: For each CONUS tile, the GHSL built-up 2015
    raster is sampled at the same UTM centroid, in a 6.72km x 6.72km
    window matching the L8 patch (224 px at 30m). GHSL native 250m
    resolution becomes ~27 px which is upsampled to 224 with
    nearest-neighbour to give a per-pixel binary target at 224x224.
  - **Loss**: BCE on per-pixel built-up probability.

Run:
    python -m scripts.eval.train_prithvi_head
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.eval.prithvi_encoder import (
    PrithviEncoder, PrithviBuiltupHead, PrithviSegmentationModel,
    load_prithvi_encoder, PRITHVI_BANDS, PRITHVI_FRAMES,
    PRITHVI_IMG_SIZE,
)


REPO = Path(__file__).resolve().parents[2]
GEOTIFF = (REPO / "data/processed/conus").resolve()


# --------------------------------------------------------------------------- #
#                              data plumbing                                    #
# --------------------------------------------------------------------------- #


def _l2_reflectance(raw: np.ndarray) -> np.ndarray:
    """Landsat C2 L2 surface reflectance scaling, clipped to [0,1].
    Cloud / saturation / nodata pixels can push raw uint16 beyond the
    valid 0-10000 reflectance band (which scales to 0.075-0.475); we
    clip to [0,1] for a sane physical reflectance prior."""
    r = raw.astype(np.float32) * 2.75e-5 - 0.2
    return np.clip(r, 0.0, 1.0)


def _conus_target_for_tile(tile_centroid_utm, ghsl_2015_path: Path,
                            patch_size_m: float = 224 * 30,
                            out_px: int = PRITHVI_IMG_SIZE,
                            ghsl_crs: str = "EPSG:5070"):
    """Read GHSL built-up 2015 for a 6.72km x 6.72km window centered on
    `tile_centroid_utm` (assumed to be in CONUS EPSG:5070), resampled
    via nearest-neighbour to `out_px` x `out_px` and binarised at 0.5.
    """
    cx, cy = tile_centroid_utm
    half = patch_size_m / 2.0
    with rasterio.open(str(ghsl_2015_path)) as src:
        # Compute pixel window for the geographic window centered on (cx, cy)
        ul_col, ul_row = ~src.transform * (cx - half, cy + half)
        lr_col, lr_row = ~src.transform * (cx + half, cy - half)
        col_off = int(round(min(ul_col, lr_col)))
        row_off = int(round(min(ul_row, lr_row)))
        width = max(1, int(round(abs(lr_col - ul_col))))
        height = max(1, int(round(abs(lr_row - ul_row))))
        col_off = max(0, min(src.width - width, col_off))
        row_off = max(0, min(src.height - height, row_off))
        window = Window(col_off, row_off, width, height)
        arr = src.read(1, window=window)
    if arr.shape != (out_px, out_px):
        # nearest-neighbour resample to out_px x out_px
        from PIL import Image
        arr = np.array(Image.fromarray(arr).resize((out_px, out_px),
                                                    Image.NEAREST))
    arr = (arr >= 0.5).astype(np.float32)
    return arr


class CONUSL8Dataset(torch.utils.data.Dataset):
    """One sample = (L8 patch (6, 3, 224, 224) reflectance, target (1, 224, 224) binary)."""

    def __init__(self, l8_npz_path: Path, eval_metrics_path: Path,
                 ghsl_2015_path: Path,
                 indices: list[int] | None = None):
        d = np.load(str(l8_npz_path), allow_pickle=True)
        self.patches = d["patches"]                  # (n, 3, 6, 224, 224) uint16
        self.tile_ids = d["tile_ids"]
        # Build centroid lookup from eval_metrics
        ev = json.loads(eval_metrics_path.read_text())
        self.tile_id_to_centroid = {}
        for rec in ev:
            bb = rec["bbox_utm"]
            cx = 0.5 * (bb[0] + bb[2])
            cy = 0.5 * (bb[1] + bb[3])
            self.tile_id_to_centroid[rec["tile_id"]] = (cx, cy, rec["utm_crs"])
        self.ghsl_path = ghsl_2015_path
        self.indices = indices if indices is not None else list(range(len(self.tile_ids)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        raw = self.patches[idx]                       # (3 frames, 6 bands, 224, 224)
        # Transpose to (6 bands, 3 frames, 224, 224) which Prithvi expects.
        raw_btf = np.transpose(raw, (1, 0, 2, 3))     # (6, 3, 224, 224)
        x = _l2_reflectance(raw_btf)
        tile_id = str(self.tile_ids[idx])
        if tile_id not in self.tile_id_to_centroid:
            # Fall back to zeros if no centroid lookup (shouldn't happen)
            y = np.zeros((PRITHVI_IMG_SIZE, PRITHVI_IMG_SIZE), dtype=np.float32)
        else:
            cx, cy, crs = self.tile_id_to_centroid[tile_id]
            if crs != "EPSG:5070":
                # convert to EPSG:5070 for CONUS GHSL lookup
                xs, ys = warp_transform(crs, "EPSG:5070", [cx], [cy])
                cx, cy = xs[0], ys[0]
            y = _conus_target_for_tile((cx, cy), self.ghsl_path)
        return torch.from_numpy(x), torch.from_numpy(y[None])


# --------------------------------------------------------------------------- #
#                                training                                       #
# --------------------------------------------------------------------------- #


class FocalLoss(nn.Module):
    """Focal loss for sparse-positive binary segmentation
    (Lin et al., 2017, "Focal Loss for Dense Object Detection").
    Default gamma=2.0, alpha=0.25 matches the FAIR RetinaNet recipe and
    is robust to the heavy class imbalance in our 6.72km L8 patches
    centered on city tile centroids (only ~0.1-1% pixels are built-up
    at this scale because the patch extends well past the urban core)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        pt = pred * target + (1.0 - pred) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        return -(alpha_t * (1.0 - pt).pow(self.gamma) * pt.log()).mean()


def train(args):
    dev = torch.device(args.device)
    print(f"loading Prithvi encoder ...", flush=True)
    enc = load_prithvi_encoder(args.encoder_ckpt, device=dev)
    head = PrithviBuiltupHead().to(dev)
    model = PrithviSegmentationModel(enc, head).to(dev)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"model: total {n_total:,} params, trainable {n_trainable:,}", flush=True)

    print(f"loading CONUS L8 dataset from {args.conus_npz} ...", flush=True)
    full_ds = CONUSL8Dataset(
        args.conus_npz,
        REPO / "data" / "processed" / "conus" / "conus" / "eval_metrics.json",
        args.ghsl_2015,
    )
    n_total_tiles = len(full_ds)
    print(f"  {n_total_tiles} CONUS tile patches", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total_tiles).tolist()
    n_val = int(0.1 * n_total_tiles)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_ds = CONUSL8Dataset(args.conus_npz,
                              REPO / "data" / "processed" / "conus" / "conus" / "eval_metrics.json",
                              args.ghsl_2015, indices=train_idx)
    val_ds = CONUSL8Dataset(args.conus_npz,
                            REPO / "data" / "processed" / "conus" / "conus" / "eval_metrics.json",
                            args.ghsl_2015, indices=val_idx)
    print(f"  train={len(train_ds)} val={len(val_ds)}", flush=True)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    best_val = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.n_epochs):
        model.train()
        total_t = 0.0; n_b = 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total_t += loss.item()
            n_b += 1
        train_loss = total_t / max(n_b, 1)

        model.eval()
        total_v = 0.0; n_v = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                pred = model(x)
                total_v += loss_fn(pred, y).item()
                n_v += 1
        val_loss = total_v / max(n_v, 1)
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"head_state_dict": head.state_dict()}, str(args.out))
            marker = "  ** saved"
        print(f"epoch {epoch+1:>2}/{args.n_epochs}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}{marker}", flush=True)

    print(f"\nbest val: {best_val:.4f}  ->  {args.out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder-ckpt", type=Path,
                   default=REPO.parent / "models" / "foundation" / "Prithvi_EO_V1_100M.pt")
    p.add_argument("--conus-npz", type=Path,
                   default=REPO / "data" / "raw" / "landsat8" / "conus.npz")
    p.add_argument("--ghsl-2015", type=Path,
                   default=GEOTIFF / "CONUS_builtup_2015.tif")
    p.add_argument("--out", type=Path,
                   default=REPO.parent / "models" / "foundation" / "best_prithvi_head.pth")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="mps" if torch.backends.mps.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
