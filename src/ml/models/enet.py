# src/ml/models/enet.py
from __future__ import annotations

import torch
import torch.nn as nn


class InitialBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        conv_out = out_ch - in_ch
        self.conv = nn.Conv2d(in_ch, conv_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.pool(x)
        x = torch.cat([x_conv, x_pool], dim=1)
        x = self.bn(x)
        x = self.act(x)
        return x


class DownBottleneck(nn.Module):
    """
    Downsample bottleneck that returns (out, indices) where indices come from pooling the main branch.
    Indices channel count == in_ch (important for MaxUnpool2d later).
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

        mid = out_ch // 4
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act1 = nn.PReLU(mid)

        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.act2 = nn.PReLU(mid)

        self.conv3 = nn.Conv2d(mid, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.drop = nn.Dropout2d(dropout)
        self.act_out = nn.PReLU(out_ch)

    def forward(self, x):
        identity, indices = self.pool(x)  # identity: (B, in_ch, H/2, W/2)

        x = self.conv1(x)
        x = self.bn1(x); x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x); x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.drop(x)

        # pad identity channels up to out_ch if needed
        if identity.shape[1] != x.shape[1]:
            pad_ch = x.shape[1] - identity.shape[1]
            identity = nn.functional.pad(identity, (0, 0, 0, 0, 0, pad_ch))

        out = x + identity
        out = self.act_out(out)
        return out, indices


class RegularBottleneck(nn.Module):
    def __init__(self, ch: int, dropout: float):
        super().__init__()
        mid = ch // 4
        self.conv1 = nn.Conv2d(ch, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act1 = nn.PReLU(mid)

        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.act2 = nn.PReLU(mid)

        self.conv3 = nn.Conv2d(mid, ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch)

        self.drop = nn.Dropout2d(dropout)
        self.act_out = nn.PReLU(ch)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x); x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x); x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.drop(x)

        x = x + identity
        x = self.act_out(x)
        return x


class ENet(nn.Module):
    """
    Simple, stable ENet-like model with correct MaxUnpool channel handling.
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.initial = InitialBlock(3, 16)  # -> 1/2

        # Encoder
        self.down1 = DownBottleneck(16, 64, dropout=0.01)  # -> 1/4, indices have 16 ch
        self.enc1 = nn.Sequential(
            RegularBottleneck(64, dropout=0.01),
            RegularBottleneck(64, dropout=0.01),
            RegularBottleneck(64, dropout=0.01),
            RegularBottleneck(64, dropout=0.01),
        )

        self.down2 = DownBottleneck(64, 128, dropout=0.1)  # -> 1/8, indices have 64 ch
        self.enc2 = nn.Sequential(
            RegularBottleneck(128, dropout=0.1),
            RegularBottleneck(128, dropout=0.1),
            RegularBottleneck(128, dropout=0.1),
            RegularBottleneck(128, dropout=0.1),
            RegularBottleneck(128, dropout=0.1),
            RegularBottleneck(128, dropout=0.1),
            RegularBottleneck(128, dropout=0.1),
            RegularBottleneck(128, dropout=0.1),
        )

        # Decoder: must unpool with matching channel counts
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dec2_reduce = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.dec2 = nn.Sequential(
            RegularBottleneck(64, dropout=0.1),
            RegularBottleneck(64, dropout=0.1),
        )

        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.dec1_reduce = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
        )
        self.dec1 = nn.Sequential(
            RegularBottleneck(16, dropout=0.1),
        )

        # Final upsample to input size
        self.head = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        x0 = self.initial(x)        # (B,16,H/2,W/2)

        x1, ind1 = self.down1(x0)   # (B,64,H/4,W/4), ind1: (B,16,H/4,W/4)
        x1 = self.enc1(x1)

        x2, ind2 = self.down2(x1)   # (B,128,H/8,W/8), ind2: (B,64,H/8,W/8)
        x2 = self.enc2(x2)

        # Decode 1: 1/8 -> 1/4
        x = self.dec2_reduce(x2)    # (B,64,H/8,W/8)  (matches ind2 channels)
        x = self.unpool2(x, ind2, output_size=(x1.size(0), 64, x1.size(2), x1.size(3)))  # -> (B,64,H/4,W/4)
        x = self.dec2(x)

        # Decode 2: 1/4 -> 1/2
        x = self.dec1_reduce(x)     # (B,16,H/4,W/4) (matches ind1 channels)
        x = self.unpool1(x, ind1, output_size=(x0.size(0), 16, x0.size(2), x0.size(3)))  # -> (B,16,H/2,W/2)
        x = self.dec1(x)

        # Head: 1/2 -> 1/1
        x = self.head(x)            # (B,C,H,W)
        return x
