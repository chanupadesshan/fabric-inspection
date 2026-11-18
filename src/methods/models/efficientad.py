"""EfficientAD  Batzner et al., "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level
Latencies" (WACV 2024).

Three small convolutional networks look at a tile:
  teacher      a "patch description network" (PDN) that turns every patch into a feature vector. Frozen.
  student      same network, trained on GOOD tiles only to copy the teacher. On defects it copies badly.
  autoencoder  rebuilds the teacher's features from a squeezed code. Also trained on good tiles only.
The student has two outputs: the first half imitates the teacher, the second half imitates the autoencoder.
Anomaly = (teacher vs student difference) and (autoencoder vs student difference), each scaled with
quantiles measured on good tiles, then averaged.

This model has a real training loop (`steps`). It is built to be fast at inference.

    fitted = fit_efficientad(good_tiles, steps=3000)
    amap   = efficientad_map(tile, fitted)

Differences from the paper (so it runs here without extra downloads):
* The paper distills the teacher PDN once on ImageNet from a WideResNet-101 and ships the weights. Here the
  teacher is distilled from the frozen torchvision wide_resnet50_2 (layer2 + layer3) on YOUR good tiles
  (`teacher_steps`). A teacher trained only on one fabric generalises less than the paper's.
* The paper's penalty term needs ImageNet images. It is optional here: pass any other images (other
  fabrics, ordinary photos) as `penalty_images`; without them the student may also copy the teacher on
  defects, which lowers the detection.
* The autoencoder is fully convolutional (spatial code, any tile size), not the paper's global code.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from ._train import sample, seed_all
from .backbone import DEVICE, WEIGHTS, _features, _finish, _to_batch


def _pdn(size, out):
    """Patch description network: output grid is 1/4 of the input size, `out` channels."""
    if size == "small":
        body = [nn.Conv2d(3, 128, 4, padding=3), nn.ReLU(True), nn.AvgPool2d(2, 2, padding=1),
                nn.Conv2d(128, 256, 4, padding=3), nn.ReLU(True), nn.AvgPool2d(2, 2, padding=1),
                nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True), nn.Conv2d(256, out, 4), nn.ReLU(True)]
    elif size == "medium":
        body = [nn.Conv2d(3, 256, 4, padding=3), nn.ReLU(True), nn.AvgPool2d(2, 2, padding=1),
                nn.Conv2d(256, 512, 4, padding=3), nn.ReLU(True), nn.AvgPool2d(2, 2, padding=1),
                nn.Conv2d(512, 512, 1), nn.ReLU(True), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(512, out, 4), nn.ReLU(True)]
    else:
        raise ValueError("size must be 'small' or 'medium'")
    return nn.Sequential(*body, nn.Conv2d(out, out, 1))


class _AutoEncoder(nn.Module):
    """Encoder: five stride-2 convolutions (1/32 of the input). Decoder: three upsampling blocks, then a
    resize to the teacher's grid (1/4). Dropout in the decoder as in the paper."""

    def __init__(self, out):
        super().__init__()
        enc, c = [], 3
        for o in (32, 32, 64, 64, 64):
            enc += [nn.Conv2d(c, o, 4, 2, 1), nn.ReLU(True)]
            c = o
        self.enc = nn.Sequential(*enc, nn.Conv2d(64, 64, 3, 1, 1))
        self.dec = nn.ModuleList([nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Dropout(0.2))
                                  for _ in range(3)])
        self.head = nn.Conv2d(64, out, 3, 1, 1)

    def forward(self, x, size):
        z = self.enc(x)
        for block in self.dec:
            z = block(Fn.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False))
        return self.head(Fn.interpolate(z, size=size, mode="bilinear", align_corners=False))


def _hard_mse(d, frac=0.001):
    """Mean of the largest `frac` share of the squared differences (the paper's hard-feature loss)."""
    flat = d.flatten()
    k = max(1, int(round(frac * flat.numel())))
    thr = torch.topk(flat, k).values[-1]
    return flat[flat >= thr].mean()


def _distill_teacher(tiles, cfg, size, out, steps, batch_size, lr, rng, verbose):
    """Teacher PDN trained to reproduce (a fixed random subset of) the frozen backbone's layer2 + layer3
    features, normalised per channel. Returns the frozen teacher (its outputs are already normalised)."""
    device = cfg["device"]
    bcfg = dict(backbone=cfg["backbone"], layers=("layer2", "layer3"), weights=cfg["weights"], device=device,
                tile_norm=False)
    teacher = _pdn(size, out).to(device)

    def targets(batch, grid):
        feats = _features(batch, bcfg)
        return torch.cat([Fn.interpolate(f, size=grid, mode="bilinear", align_corners=False) for f in feats], 1)

    with torch.no_grad():
        grid = teacher(_to_batch(tiles[:1], device)).shape[-2:]
        total = targets(tiles[:1], grid).shape[1]
        if total < out:
            raise ValueError(f"the backbone gives {total} feature channels (layer2 + layer3), fewer than out_channels={out}: "
                             "lower out_channels or use backbone='wide_resnet50_2'")
        dims = torch.randperm(total, generator=torch.Generator().manual_seed(0))[:out].sort().values.to(device)
        n, s1, s2 = 0, 0, 0                                   # per-channel mean / std from running sums (low memory)
        for i in range(0, len(tiles), 8):
            f = targets(tiles[i:i + 8], grid)[:, dims].double()
            n += f.shape[0] * f.shape[2] * f.shape[3]
            s1 = s1 + f.sum((0, 2, 3))
            s2 = s2 + (f ** 2).sum((0, 2, 3))
        mean = (s1 / n).float().view(1, -1, 1, 1)
        std = ((s2 / n - (s1 / n) ** 2).clamp_min(0).sqrt().float() + 1e-6).view(1, -1, 1, 1)
    opt = torch.optim.Adam(teacher.parameters(), lr=lr, weight_decay=1e-5)
    teacher.train()
    run = []
    for step in range(steps):
        batch = sample(tiles, rng, batch_size)
        with torch.no_grad():
            target = (targets(batch, grid)[:, dims] - mean) / std
        loss = Fn.mse_loss(teacher(_to_batch(batch, device)), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run.append(float(loss))
        if verbose and (step + 1) % max(1, steps // 5) == 0:
            print(f"efficientad teacher step {step + 1}/{steps}  loss {np.mean(run[-max(1, steps // 5):]):.4f}")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def fit_efficientad(good_tiles, size="small", out_channels=384, steps=3000, teacher_steps=2000, batch_size=4,
                    lr=1e-4, backbone="wide_resnet50_2", penalty_images=None, smooth=4.0, seed=0,
                    weights=WEIGHTS, device=None, verbose=True):
    """Distill the teacher, then train student + autoencoder on good tiles only. Returns the fitted model.

    size            "small" or "medium" PDN (medium: more capacity, 2x slower).
    out_channels    feature channels of the PDN (paper: 384).
    steps           student + autoencoder training steps (paper: 70 000; 2000-5000 is a short run).
    teacher_steps   steps to distill the teacher from the backbone on the good tiles.
    batch_size      tiles per step.
    lr              Adam learning rate (x0.1 for the last 5 % of the steps).
    backbone        frozen network the teacher learns from ("wide_resnet50_2" or the faster "resnet18").
    penalty_images  optional list of grayscale images (>= 256 px, NOT good tiles of this fabric): the student's
                    output on them is pushed to zero, so it does not learn to copy the teacher everywhere.
    smooth          Gaussian sigma (px) of the final map.
    """
    device = device or DEVICE
    rng = seed_all(seed)
    tiles = list(good_tiles)
    cfg = dict(model="efficientad", backbone=backbone, weights=weights, device=device, smooth=smooth,
               out=out_channels)
    teacher = _distill_teacher(tiles, cfg, size, out_channels, teacher_steps, batch_size, 1e-4, rng, verbose)
    student = _pdn(size, 2 * out_channels).to(device)
    ae = _AutoEncoder(out_channels).to(device)
    opt = torch.optim.Adam(list(student.parameters()) + list(ae.parameters()), lr=lr, weight_decay=1e-5)
    student.train()
    ae.train()
    run = []
    for step in range(steps):
        if step == int(0.95 * steps):
            for g in opt.param_groups:
                g["lr"] = lr / 10
        x = _to_batch(sample(tiles, rng, batch_size), device)
        with torch.no_grad():
            t = teacher(x)
        s = student(x)
        a = ae(x, t.shape[-2:])
        loss = _hard_mse((t - s[:, :out_channels]) ** 2) + Fn.mse_loss(a, t) + Fn.mse_loss(a, s[:, out_channels:])
        if penalty_images:
            img = penalty_images[rng.randint(len(penalty_images))]
            y, x0 = rng.randint(img.shape[0] - 255), rng.randint(img.shape[1] - 255)
            loss = loss + (student(_to_batch([img[y:y + 256, x0:x0 + 256]], device))[:, :out_channels] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        run.append(float(loss))
        if verbose and (step + 1) % max(1, steps // 10) == 0:
            print(f"efficientad step {step + 1}/{steps}  loss {np.mean(run[-max(1, steps // 10):]):.4f}")
    cfg.update(teacher=teacher, student=student.eval(), ae=ae.eval(), loss=run)
    # scale both maps with quantiles measured on good tiles (paper: 0.9 and 0.995)
    st_vals, ae_vals = [], []
    with torch.no_grad():
        for i in range(0, min(len(tiles), 64), 8):
            m_st, m_ae = _raw_maps(tiles[i:i + 8], cfg)
            st_vals.append(m_st.flatten().cpu())
            ae_vals.append(m_ae.flatten().cpu())
    for name, vals in (("st", st_vals), ("ae", ae_vals)):
        v = torch.cat(vals).numpy()
        cfg["q_" + name] = (float(np.quantile(v, 0.9)), float(np.quantile(v, 0.995)))
    return cfg


def _raw_maps(tiles, cfg):
    """Teacher-student and autoencoder-student difference maps [B, h, w] at 1/4 of the tile size."""
    x = _to_batch(tiles, cfg["device"])
    t = cfg["teacher"](x)
    s = cfg["student"](x)
    a = cfg["ae"](x, t.shape[-2:])
    out = cfg["out"]
    return ((t - s[:, :out]) ** 2).mean(1), ((a - s[:, out:]) ** 2).mean(1)


@torch.no_grad()
def efficientad_map(tile, fitted):
    """Anomaly map: mean of the scaled teacher-student and autoencoder-student differences."""
    m_st, m_ae = _raw_maps([tile], fitted)
    (s0, s1), (a0, a1) = fitted["q_st"], fitted["q_ae"]
    m = 0.5 * ((m_st - s0) / max(s1 - s0, 1e-8) + (m_ae - a0) / max(a1 - a0, 1e-8))
    return _finish(m[0].cpu().numpy(), np.shape(tile), fitted["smooth"])
