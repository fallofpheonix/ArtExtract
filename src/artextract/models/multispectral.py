from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

from artextract.reconstruction import ReconstructionUNet


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(f"torch unavailable: {_IMPORT_ERROR}")
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stage1 = EncoderBlock(in_channels, c1, stride=1)
        self.stage2 = EncoderBlock(c1, c2, stride=2)
        self.stage3 = EncoderBlock(c2, c3, stride=2)
        self.stage4 = EncoderBlock(c3, c4, stride=2)
        self.out_channels = c4

    def forward(self, x):
        e1 = self.stage1(x)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)
        return {
            "e1": e1,
            "e2": e2,
            "e3": e3,
            "bottleneck": e4,
        }


@dataclass(frozen=True)
class TaskFlags:
    properties: bool
    hidden: bool
    reconstruction: bool


class MultiSpectralMultiTaskModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_pigments: int,
        enable_properties: bool = True,
        enable_hidden: bool = True,
        enable_reconstruction: bool = False,
        base_channels: int = 32,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(f"torch unavailable: {_IMPORT_ERROR}")
        super().__init__()

        self.in_channels = int(in_channels)
        self.flags = TaskFlags(
            properties=bool(enable_properties),
            hidden=bool(enable_hidden),
            reconstruction=bool(enable_reconstruction),
        )

        self.encoder = SharedEncoder(in_channels=self.in_channels, base_channels=base_channels)
        head_in = self.encoder.out_channels + self.in_channels

        if self.flags.properties:
            self.pigments_head = nn.Linear(head_in, num_pigments)
            self.damage_head = nn.Linear(head_in, 1)
            self.restoration_head = nn.Linear(head_in, 1)

        if self.flags.hidden:
            self.hidden_head = nn.Linear(head_in, 1)

        if self.flags.reconstruction:
            self.reconstruction_head = ReconstructionUNet(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                base_channels=base_channels,
            )

    def forward(self, x, channel_mask):
        if channel_mask.dim() != 2:
            raise ValueError("channel_mask must have shape [B, C]")
        if channel_mask.shape[1] != self.in_channels:
            raise ValueError(
                f"channel_mask channels mismatch: expected {self.in_channels}, got {channel_mask.shape[1]}"
            )

        x_masked = x * channel_mask.unsqueeze(-1).unsqueeze(-1)
        feats = self.encoder(x_masked)
        pooled = torch.nn.functional.adaptive_avg_pool2d(feats["bottleneck"], 1).flatten(1)
        shared = torch.cat([pooled, channel_mask], dim=1)

        out = {
            "embedding": shared,
        }

        if self.flags.properties:
            out["pigments_logits"] = self.pigments_head(shared)
            out["damage_logits"] = self.damage_head(shared).squeeze(1)
            out["restoration_logits"] = self.restoration_head(shared).squeeze(1)

        if self.flags.hidden:
            out["hidden_logits"] = self.hidden_head(shared).squeeze(1)

        if self.flags.reconstruction:
            out["reconstruction"] = self.reconstruction_head(x_masked)

        return out
