from __future__ import annotations

import math

try:
    import torch
except Exception as e:  # pragma: no cover
    torch = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def mae(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    return (pred - target).abs().mean()


def mse(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    return ((pred - target) ** 2).mean()


def psnr(pred: "torch.Tensor", target: "torch.Tensor") -> float:
    m = float(mse(pred, target).item())
    if m <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / m)
