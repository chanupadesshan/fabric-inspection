"""Shared part of the models: the frozen pretrained backbone and the helpers that turn tiles into
feature maps (used by PatchCore and PaDiM)."""
import os

import cv2
import numpy as np
import torch
import torchvision

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = ("layer1", "layer2", "layer3", "layer4")
# The papers use the original ImageNet weights (V1). torchvision's newer "DEFAULT" (V2) weights for
# wide_resnet50_2 are trained differently and are not what PatchCore / PaDiM were designed around.
WEIGHTS = "IMAGENET1K_V1"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_BACKBONES = {}


def device_info():
    return f"torch {torch.__version__}, device: {DEVICE}" + (
        f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else " (no GPU found: slower, still works)")


# =====================================================================
# Shared: frozen pretrained backbone -> feature maps
# =====================================================================
class _Backbone(torch.nn.Module):
    """ResNet-family network cut after the deepest layer that is needed. Returns the chosen layers."""

    def __init__(self, name, layers, weights):
        super().__init__()
        from_file = isinstance(weights, (str, os.PathLike)) and os.path.isfile(str(weights))
        net = getattr(torchvision.models, name)(weights=None if from_file else weights)
        if from_file:  # a state_dict saved on disk (for machines that cannot download weights)
            net.load_state_dict(torch.load(str(weights), map_location="cpu"))
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layers = tuple(layers)
        self.names = LAYERS[:max(LAYERS.index(l) for l in layers) + 1]
        self.blocks = torch.nn.ModuleList([getattr(net, n) for n in self.names])

    def forward(self, x):
        out = {}
        x = self.stem(x)
        for n, block in zip(self.names, self.blocks):
            x = block(x)
            if n in self.layers:
                out[n] = x
        return [out[l] for l in self.layers]


def _backbone(name, layers, weights, device):
    key = (name, tuple(layers), str(weights), device)
    if key not in _BACKBONES:
        _BACKBONES[key] = _Backbone(name, layers, weights).to(device).eval()
    return _BACKBONES[key]


def _to_batch(tiles, device):
    """Grayscale uint8 tiles -> [B, 3, H, W] float tensor, scaled the way the pretrained network expects.
    The gray channel is repeated 3 times. No resize and no crop: small defects must survive."""
    x = torch.from_numpy(np.stack([np.asarray(t, dtype=np.float32) for t in tiles]) / 255.0)
    return ((x.unsqueeze(1).repeat(1, 3, 1, 1) - _MEAN) / _STD).to(device)


@torch.no_grad()
def _features(tiles, cfg):
    """Feature maps of the chosen layers for a list of tiles (one list entry per layer, on cfg device)."""
    net = _backbone(cfg["backbone"], cfg["layers"], cfg["weights"], cfg["device"])
    feats = net(_to_batch(tiles, cfg["device"]))
    if cfg.get("tile_norm"):
        # every feature channel relative to this tile's own average patch (and spread)
        feats = [(f - f.mean((2, 3), keepdim=True)) / (f.std((2, 3), keepdim=True) + 1e-6) for f in feats]
    return feats


def _batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _finish(dist, shape, smooth):
    """Patch-grid distances [h, w] -> full-size smoothed anomaly map."""
    m = cv2.resize(dist.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(m, (0, 0), smooth) if smooth else m
