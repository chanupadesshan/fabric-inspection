"""PatchCore  Roth et al., "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022).

Keeps a memory bank of good-patch features (reduced to a coreset); anomaly = distance of a patch to the
nearest good patches. Trained on GOOD tiles only.

    fitted = fit_patchcore(good_tiles, ...)
    amap   = patchcore_map(tile, fitted)
"""
import numpy as np
import torch
import torch.nn.functional as Fn

from .backbone import DEVICE, WEIGHTS, _batches, _features, _finish


def _patchcore_embed(feats, pool):
    """Locally averaged features of every layer, brought to the finest layer's grid and concatenated.
    Returns [B, C, h, w]."""
    if pool and pool > 1:
        feats = [Fn.avg_pool2d(f, pool, 1, pool // 2) for f in feats]
    size = feats[0].shape[-2:]
    feats = [feats[0]] + [Fn.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in feats[1:]]
    return torch.cat(feats, 1)


def _coreset(x, n, seed, device):
    """Greedy k-centre selection: n patches that cover the whole set as evenly as possible
    (distances are measured in a random 128-d projection to keep it fast)."""
    g = torch.Generator().manual_seed(seed)
    z = x.to(device)
    if z.shape[1] > 128:
        z = z @ (torch.randn(x.shape[1], 128, generator=g) / 128 ** 0.5).to(device)
    first = int(torch.randint(len(z), (1,), generator=g))
    idx = [first]
    d = ((z - z[first]) ** 2).sum(1)
    for _ in range(n - 1):
        i = int(torch.argmax(d))
        idx.append(i)
        d = torch.minimum(d, ((z - z[i]) ** 2).sum(1))
    return torch.tensor(idx)


def fit_patchcore(good_tiles, backbone="wide_resnet50_2", layers=("layer2", "layer3"), coreset=0.01,
                  n_neighbors=1, pool=3, smooth=4.0, tile_norm=False, max_patches=200_000, seed=0,
                  weights=WEIGHTS, device=None, batch_size=8):
    """Build the PatchCore memory bank from good tiles only.

    backbone     pretrained network: "wide_resnet50_2" (paper, best, slow on CPU) or "resnet18" (fast).
    layers       which feature maps to use. ("layer2", "layer3") is the paper's choice: 32 x 32 patch
                 grid on a 256 px tile. ("layer1", "layer2") gives a finer 64 x 64 grid (smaller
                 defects) but 4x more patches to compare.
    coreset      share of the good patches kept in the memory bank (0.01 - 0.1). Larger = a little more
                 accurate, slower to score.
    n_neighbors  anomaly = mean distance to this many nearest good patches (1 = paper).
    pool         local averaging of the features before comparing (3 = paper).
    smooth       Gaussian sigma (px) of the final map.
    tile_norm    True = every feature channel is taken relative to the tile's own average patch, so a
                 whole strip that looks slightly different (contrast, weave tightness) is not scored as
                 abnormal everywhere. Not in the paper; an option for the strip-to-strip bias.
    max_patches  cap on patches considered before coreset selection (memory and time limit).
    """
    cfg = dict(model="patchcore", backbone=backbone, layers=tuple(layers), weights=weights,
               device=device or DEVICE, pool=pool, smooth=smooth, n_neighbors=n_neighbors, tile_norm=tile_norm)
    bank = []
    for batch in _batches(list(good_tiles), batch_size):
        e = _patchcore_embed(_features(batch, cfg), pool)
        bank.append(e.permute(0, 2, 3, 1).reshape(-1, e.shape[1]).cpu())
    bank = torch.cat(bank)
    g = torch.Generator().manual_seed(seed)
    if len(bank) > max_patches:
        bank = bank[torch.randperm(len(bank), generator=g)[:max_patches]]
    n = max(1, int(round(coreset * len(bank))))
    cfg["bank"] = bank[_coreset(bank, n, seed, cfg["device"])].to(cfg["device"])
    return cfg


@torch.no_grad()
def patchcore_map(tile, fitted):
    """Anomaly map: distance of every patch of the tile to its nearest good patches in the memory bank."""
    e = _patchcore_embed(_features([tile], fitted), fitted["pool"])[0]
    h, w = e.shape[-2:]
    q = e.permute(1, 2, 0).reshape(h * w, -1)
    k = min(fitted["n_neighbors"], len(fitted["bank"]))
    d = torch.cat([torch.cdist(part, fitted["bank"]).topk(k, largest=False).values.mean(1)
                   for part in q.split(1024)])
    return _finish(d.reshape(h, w).cpu().numpy(), np.shape(tile), fitted["smooth"])
