"""Model definitions used by `cross_region_eval`.

Architectures
are fixed: the benchmark evaluates the *same weights* on every region.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        return torch.sigmoid(self.net(x.reshape(B, T * C, H, W)))


class SimpleUNet(nn.Module):
    def __init__(self, input_channels: int = 24):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B, T * C, H, W)
        e1 = self.enc1(x_flat)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            input_channels + hidden_channels, 4 * hidden_channels,
            kernel_size, padding=kernel_size // 2,
        )

    def forward(self, x, states):
        h, c = states
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels: int = 3, hidden_channels: int = 64,
                 num_layers: int = 2, mc_dropout: float = 0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            layers.append(ConvLSTMCell(
                input_channels if i == 0 else hidden_channels,
                hidden_channels,
            ))
        self.convlstm_layers = nn.ModuleList(layers)
        self.mc_dropouts = nn.ModuleList(
            [nn.Dropout2d(p=mc_dropout) for _ in range(num_layers)]
        )
        self.skip_proj = nn.Conv2d(hidden_channels * num_layers,
                                   hidden_channels, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        states = [
            (torch.zeros(B, self.hidden_channels, H, W, device=x.device),
             torch.zeros(B, self.hidden_channels, H, W, device=x.device))
            for _ in range(self.num_layers)
        ]
        for t in range(T):
            x_t = x[:, t]
            hiddens = []
            for i, layer in enumerate(self.convlstm_layers):
                h, c = states[i]
                h, c = layer(x_t, (h, c))
                h = self.mc_dropouts[i](h)
                states[i] = (h, c)
                x_t = h
                hiddens.append(h)
        fused = self.skip_proj(torch.cat(hiddens, dim=1))
        return self.decoder(fused)


def load_models(weights_root):
    """Load CNN, U-Net, ConvLSTM, and TimeSformer
    with whichever checkpoints are present in `weights_root`.

    Falls back gracefully if a checkpoint is missing — the caller will report
    which models were available. TimeSformer is the fourth
    transformer-family baseline to test the architecture-invariance claim of
    the slope-bounded bound.
    """
    from pathlib import Path
    weights_root = Path(weights_root)
    loaded = {}

    for name, cls, ckpt in [
        ("cnn",      lambda: SimpleCNN(input_channels=24),
            weights_root / "best_cnn_3ch.pth"),
        ("unet",     lambda: SimpleUNet(input_channels=24),
            weights_root / "best_unet_3ch.pth"),
        ("convlstm", lambda: ConvLSTMModel(
            input_channels=3, hidden_channels=64,
            num_layers=2, mc_dropout=0.1),
            weights_root / "best_3ch_mc_model.pth"),
    ]:
        if not ckpt.exists():
            print(f"  missing checkpoint: {ckpt}")
            continue
        model = cls()
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        loaded[name] = model

    # Add the TimeSformer baseline if its checkpoint exists.
    # Imported lazily so cross_region_eval still works when the file is absent.
    try:
        from scripts.eval.transformer_model import load_transformer
        loaded.update(load_transformer(weights_root))
    except ImportError:
        pass

    return loaded
