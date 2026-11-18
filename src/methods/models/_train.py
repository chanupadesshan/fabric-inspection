"""Small helpers shared by the models that have a training loop (RD4AD, EfficientAD)."""
import numpy as np
import torch


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.RandomState(seed)


def augment(tile, rng):
    """Random horizontal / vertical flip. No rotation: it would change the direction of the weave,
    which is part of what 'normal' looks like."""
    t = np.asarray(tile)
    if rng.rand() < 0.5:
        t = t[:, ::-1]
    if rng.rand() < 0.5:
        t = t[::-1, :]
    return np.ascontiguousarray(t)


def sample(tiles, rng, n):
    """n random augmented tiles."""
    return [augment(tiles[i], rng) for i in rng.randint(len(tiles), size=n)]
