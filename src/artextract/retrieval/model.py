from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetRetrieval(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 32) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(f"torch unavailable: {_IMPORT_ERROR}")
        super().__init__()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8

        self.down1 = _ConvBlock(in_ch, c1)
        self.down2 = _ConvBlock(c1, c2)
        self.down3 = _ConvBlock(c2, c3)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _ConvBlock(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(c1 + c1, c1)

        self.head = nn.Conv2d(c1, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.down1(x)
        e2 = self.down2(self.pool(e1))
        e3 = self.down3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        y = torch.sigmoid(self.head(d1))
        return y
