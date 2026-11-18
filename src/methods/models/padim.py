"""PaDiM  Defard et al., "PaDiM: a Patch Distribution Modeling Framework" (ICPR 2021).

Fits a Gaussian to the good-patch features (one per patch position); anomaly = Mahalanobis distance.
Trained on GOOD tiles only.

    fitted = fit_padim(good_tiles, ...)
    amap   = padim_map(tile, fitted)
"""
import numpy as np
import torch
import torch.nn.functional as Fn

from .backbone import DEVICE, WEIGHTS, _batches, _features, _finish


def _padim_embed(feats, dims):
    """Features of every layer on the finest layer's grid, concatenated, random subset of channels.
    Returns [B, d, h, w]."""
    size = feats[0].shape[-2:]
    feats = [feats[0]] + [Fn.interpolate(f, size=size, mode="nearest") for f in feats[1:]]
    return torch.cat(feats, 1)[:, dims]


def fit_padim(good_tiles, backbone="resnet18", layers=("layer1", "layer2", "layer3"), n_dims=100,
              per_position=True, eps=0.01, smooth=4.0, tile_norm=False, seed=0, weights=WEIGHTS, device=None,
              batch_size=8):
    """Fit the PaDiM Gaussian(s) to patch features of good tiles only.

    backbone      "resnet18" (paper's small model) or "wide_resnet50_2" (bigger, slower).
    layers        feature maps used; the patch grid is that of the first one (layer1: 64 x 64 on 256 px).
    n_dims        number of randomly chosen feature channels kept (paper: 100 for resnet18, 550 for
                  wide_resnet50_2). More = slower; it must stay well below the number of training patches.
    per_position  True  = the paper's version: one Gaussian per patch position (needs test tiles of the
                          same size as the training tiles).
                  False = one Gaussian shared by all positions (far less memory, any tile size).
                  On fabric 02 the paper's version was clearly better.
    eps           added to the covariance diagonal so it can be inverted (paper: 0.01).
    smooth        Gaussian sigma (px) of the final map.
    tile_norm     same option as in fit_patchcore.
    """
    cfg = dict(model="padim", backbone=backbone, layers=tuple(layers), weights=weights,
               device=device or DEVICE, smooth=smooth, per_position=bool(per_position), tile_norm=tile_norm)
    g = torch.Generator().manual_seed(seed)
    emb = []
    for batch in _batches(list(good_tiles), batch_size):
        feats = _features(batch, cfg)
        if "dims" not in cfg:
            c = sum(f.shape[1] for f in feats)
            cfg["dims"] = torch.randperm(c, generator=g)[:min(n_dims, c)].sort().values.to(cfg["device"])
        e = _padim_embed(feats, cfg["dims"])
        emb.append(e.flatten(2).permute(0, 2, 1).cpu().double())           # [B, hw, d]
    emb = torch.cat(emb)                                                     # [N, hw, d]
    n, hw, d = emb.shape
    if per_position and hw * d * d * 8 > 4e9:
        raise ValueError(f"per_position PaDiM would need {hw * d * d * 8 / 1e9:.0f} GB for {hw} positions x "
                         f"{d} dims: lower n_dims or start at a coarser layer (e.g. layers=('layer2', 'layer3'))")
    eye = eps * torch.eye(d, dtype=torch.float64)
    if per_position:
        mean = emb.mean(0)                                                   # [hw, d]
        x = emb - mean
        cov = torch.einsum("npi,npj->pij", x, x) / max(n - 1, 1) + eye       # [hw, d, d]
    else:
        flat = emb.reshape(-1, d)
        mean = flat.mean(0)                                                  # [d]
        x = flat - mean
        cov = x.T @ x / max(len(flat) - 1, 1) + eye                          # [d, d]
    cfg["mean"] = mean.float().to(cfg["device"])
    # cov is symmetric positive definite (eps on the diagonal): invert through its Cholesky factor.
    # torch.linalg.inv (batched LU) fails with multi-threaded MKL ("Pivots given to lu_solve ...").
    cfg["inv"] = torch.cholesky_inverse(torch.linalg.cholesky(cov)).float().to(cfg["device"])
    return cfg


@torch.no_grad()
def padim_map(tile, fitted):
    """Anomaly map: Mahalanobis distance of every patch from the good-fabric Gaussian."""
    e = _padim_embed(_features([tile], fitted), fitted["dims"])[0]
    h, w = e.shape[-2:]
    x = e.flatten(1).T - fitted["mean"]                                      # [hw, d]
    if fitted["per_position"]:
        if x.shape[0] != fitted["inv"].shape[0]:
            raise ValueError("per_position PaDiM needs tiles of the same size as the training tiles")
        m = torch.einsum("pi,pij,pj->p", x, fitted["inv"], x)
    else:
        m = ((x @ fitted["inv"]) * x).sum(1)
    return _finish(m.clamp_min(0).sqrt().reshape(h, w).cpu().numpy(), np.shape(tile), fitted["smooth"])
