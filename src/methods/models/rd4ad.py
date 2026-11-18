"""RD4AD  Deng and Li, "Anomaly Detection via Reverse Distillation from One-Class Embedding" (CVPR 2022).

A frozen pretrained network (the teacher) encodes a tile. A small bottleneck squeezes the teacher's
features into a compact one-class embedding, and a decoder (the student) must rebuild the teacher's
features from that embedding. It is trained on GOOD tiles only, so it rebuilds good fabric well and
defects badly. Anomaly = how different the rebuilt features are from the teacher's (1 - cosine
similarity, summed over three scales).

Unlike PatchCore and PaDiM this model has a real training loop (`steps`). It is much faster on a GPU.

    fitted = fit_rd4ad(good_tiles, steps=2000)
    amap   = rd4ad_map(tile, fitted)

Differences from the paper (compact version, so it stays short and runs on any tile size):
the bottleneck and decoder are plain convolution blocks, not the ResNet-style OCBE and reversed
ResNet of the paper; the default teacher is resnet18 (the paper uses wide_resnet50_2: pass
backbone="wide_resnet50_2", about 4x heavier).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from ._train import sample, seed_all
from .backbone import DEVICE, WEIGHTS, _backbone, _features, _finish

LAYERS = ("layer1", "layer2", "layer3")


def _cbr(i, o, k=3, s=1, p=1):
    """Conv + batch norm + ReLU."""
    return [nn.Conv2d(i, o, k, s, p, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


def _resize(x, size):
    return x if x.shape[-2:] == tuple(size) else Fn.interpolate(x, size=size, mode="bilinear", align_corners=False)


class _Student(nn.Module):
    """Bottleneck (fuses the three teacher scales into one embedding) + decoder (rebuilds the three scales)."""

    def __init__(self, c1, c2, c3, width):
        super().__init__()
        # bottleneck: bring layer1 and layer2 down to layer3's grid, join, compress
        self.down1 = nn.Sequential(*_cbr(c1, c2, 3, 2, 1), *_cbr(c2, c3, 3, 2, 1))
        self.down2 = nn.Sequential(*_cbr(c2, c3, 3, 2, 1))
        self.embed = nn.Sequential(*_cbr(3 * c3, width, 1, 1, 0), *_cbr(width, width, 3, 1, 1))
        # decoder: layer3 grid -> layer2 grid -> layer1 grid
        self.b3 = nn.Sequential(*_cbr(width, c3))
        self.o3 = nn.Conv2d(c3, c3, 3, 1, 1)
        self.up2 = nn.Sequential(nn.ConvTranspose2d(c3, c2, 2, 2), nn.BatchNorm2d(c2), nn.ReLU(inplace=True), *_cbr(c2, c2))
        self.o2 = nn.Conv2d(c2, c2, 3, 1, 1)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(c2, c1, 2, 2), nn.BatchNorm2d(c1), nn.ReLU(inplace=True), *_cbr(c1, c1))
        self.o1 = nn.Conv2d(c1, c1, 3, 1, 1)

    def forward(self, f1, f2, f3):
        size3 = f3.shape[-2:]
        z = self.embed(torch.cat([_resize(self.down1(f1), size3), _resize(self.down2(f2), size3), f3], 1))
        h3 = self.b3(z)
        h2 = _resize(self.up2(h3), f2.shape[-2:])
        h1 = _resize(self.up1(h2), f1.shape[-2:])
        return [self.o1(h1), self.o2(h2), self.o3(h3)]            # same order and shapes as (f1, f2, f3)


def _cos_maps(teacher, student):
    """1 - cosine similarity per position, one map [B, h, w] per scale."""
    return [1 - Fn.cosine_similarity(t, s, dim=1) for t, s in zip(teacher, student)]


def fit_rd4ad(good_tiles, backbone="resnet18", width=256, steps=2000, batch_size=8, lr=1e-3, smooth=4.0,
              seed=0, weights=WEIGHTS, device=None, verbose=True):
    """Train the bottleneck + decoder on good tiles only. Returns the fitted model (a dict).

    backbone    frozen teacher: "resnet18" (fast, default) or "wide_resnet50_2" (paper, 4x heavier).
    width       channels of the one-class embedding (the bottleneck). Smaller = stronger squeeze, so
                defects are harder to rebuild, but good fabric also gets harder.
    steps       training steps (one step = one batch). 1000-3000 is usually enough for ~100 good tiles;
                on a CPU expect a few minutes with resnet18.
    batch_size  tiles per step.
    lr          Adam learning rate (the last 10 % of the steps use lr / 10).
    smooth      Gaussian sigma (px) of the final map.
    verbose     print the loss every 10 % of the steps.
    """
    device = device or DEVICE
    rng = seed_all(seed)
    tiles = list(good_tiles)
    cfg = dict(model="rd4ad", backbone=backbone, layers=LAYERS, weights=weights, device=device,
               tile_norm=False, smooth=smooth)
    _backbone(backbone, LAYERS, weights, device)                                # load the frozen teacher once
    with torch.no_grad():
        c1, c2, c3 = [f.shape[1] for f in _features(tiles[:1], cfg)]
    net = _Student(c1, c2, c3, width).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
    net.train()
    history = []
    for step in range(steps):
        if step == int(0.9 * steps):
            for g in opt.param_groups:
                g["lr"] = lr / 10
        with torch.no_grad():
            feats = _features(sample(tiles, rng, batch_size), cfg)
        out = net(*feats)
        loss = sum(m.mean() for m in _cos_maps(feats, out))
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss))
        if verbose and (step + 1) % max(1, steps // 10) == 0:
            print(f"rd4ad step {step + 1}/{steps}  loss {np.mean(history[-max(1, steps // 10):]):.4f}")
    cfg["net"] = net.eval()
    cfg["loss"] = history
    return cfg


@torch.no_grad()
def rd4ad_map(tile, fitted):
    """Anomaly map: how badly the student rebuilds the teacher's features at every position."""
    feats = _features([tile], fitted)
    out = fitted["net"](*feats)
    h, w = np.shape(tile)
    m = sum(_resize(d.unsqueeze(1), (h, w)) for d in _cos_maps(feats, out)) / len(feats)
    return _finish(m[0, 0].cpu().numpy(), (h, w), fitted["smooth"])
