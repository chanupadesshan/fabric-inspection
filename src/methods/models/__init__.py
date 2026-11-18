"""Learned anomaly models for Layer 1 (PyTorch).

All of them are trained on GOOD tiles only (no defect labels) and follow the same contract as filters.py:
a fitted model turns a grayscale tile into a float32 anomaly map of the same size (higher = more
abnormal), so evaluate.py scores them exactly like the classical methods.

    fitted = fit_patchcore(good_tiles, ...)      # or fit_padim / fit_rd4ad / fit_efficientad
    amap   = anomaly_map(tile, fitted)

Files:
    backbone.py     frozen pretrained network and feature extraction (shared)
    patchcore.py    PatchCore: memory bank of good patches; no training loop
    padim.py        PaDiM: Gaussian per patch position; no training loop
    rd4ad.py        RD4AD: reverse distillation; trains a bottleneck + decoder
    efficientad.py  EfficientAD: student-teacher + autoencoder; trains student and autoencoder

Needs torch + torchvision (the classical code in filters.py does not). Runs on a GPU when one is
available, otherwise on the CPU.
"""
from .backbone import DEVICE, LAYERS, WEIGHTS, device_info
from .patchcore import fit_patchcore, patchcore_map
from .padim import fit_padim, padim_map
from .rd4ad import fit_rd4ad, rd4ad_map
from .efficientad import fit_efficientad, efficientad_map

_MAPS = {"patchcore": patchcore_map, "padim": padim_map, "rd4ad": rd4ad_map, "efficientad": efficientad_map}


def anomaly_map(tile, fitted):
    """Anomaly map from a model returned by any fit_* function."""
    return _MAPS[fitted["model"]](tile, fitted)
