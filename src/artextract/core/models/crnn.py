from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
except Exception as e:  # pragma: no cover
    torch = None
    nn = None
    tvm = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class CRNNMultiTask(nn.Module):
    """Dual-branch CNN+RNN multi-task classifier.

    Inputs:
    - x: [B, 3, H, W]
    Outputs:
    - dict with logits for style/artist/genre
    """

    def __init__(
        self,
        style_classes: int,
        artist_classes: int,
        genre_classes: int,
        patch_grid: int = 4,
        cnn_backbone: str = "resnet18",
        pretrained_backbone: bool = False,
        global_dim: int = 512,
        patch_dim: int = 256,
        rnn_hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(f"torch/torchvision unavailable: {_IMPORT_ERROR}")
        super().__init__()

        self.patch_grid = patch_grid
        self.global_backbone = self._build_backbone(cnn_backbone, pretrained_backbone)
        backbone_dim = int(self.global_backbone.fc.in_features)
        self.global_backbone.fc = nn.Identity()
        self.global_proj = (
            nn.Identity() if backbone_dim == global_dim else nn.Linear(backbone_dim, global_dim)
        )

        self.patch_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.patch_proj = nn.Linear(64, patch_dim)
        self.rnn = nn.GRU(
            input_size=patch_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        fusion_dim = global_dim + (2 * rnn_hidden)
        self.dropout = nn.Dropout(dropout)
        self.style_head = nn.Linear(fusion_dim, style_classes)
        self.artist_head = nn.Linear(fusion_dim, artist_classes)
        self.genre_head = nn.Linear(fusion_dim, genre_classes)

    def _build_backbone(self, name: str, pretrained: bool):
        n = (name or "resnet18").lower()
        ctor = {
            "resnet18": tvm.resnet18,
            "resnet34": tvm.resnet34,
            "resnet50": tvm.resnet50,
        }.get(n)
        if ctor is None:
            raise ValueError(f"unsupported cnn_backbone: {name}")
        if pretrained:
            try:
                return ctor(weights="DEFAULT")
            except Exception:
                return ctor(weights=None)
        return ctor(weights=None)

    def _extract_patch_sequence(self, x):
        b, c, h, w = x.shape
        gh = gw = self.patch_grid
        ph = h // gh
        pw = w // gw
        x = x[:, :, : ph * gh, : pw * gw]

        patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
        patches = patches.contiguous().view(b, c, gh * gw, ph, pw)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(b * gh * gw, c, ph, pw)

        emb = self.patch_encoder(patches).flatten(1)
        emb = self.patch_proj(emb)
        emb = emb.view(b, gh * gw, -1)
        return emb

    def forward(self, x):
        g = self.global_backbone(x)
        g = self.global_proj(g)
        seq = self._extract_patch_sequence(x)
        _, h = self.rnn(seq)
        h = h.transpose(0, 1).reshape(x.size(0), -1)

        z = torch.cat([g, h], dim=1)
        z = self.dropout(z)

        return {
            "style": self.style_head(z),
            "artist": self.artist_head(z),
            "genre": self.genre_head(z),
            "embedding": z,
        }
