# src/ml/models/enet.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # ENet initial block: conv + maxpool concat
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


class Bottleneck(nn.Module):
    """
    ENet bottleneck.
    - regular, downsample, upsample variants
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        downsample: bool = False,
        upsample: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert not (downsample and upsample), "downsample and upsample cannot both be True"

        self.downsample = downsample
        self.upsample = upsample

        mid_ch = out_ch // 4

        if downsample:
            # main branch: maxpool + indices
            self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

            # ext branch
            self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=2, stride=2, padding=0, bias=False)
        elif upsample:
            # main branch: 1x1 conv + upsample
            self.conv_main = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

            # ext branch
            self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.act1 = nn.PReLU(mid_ch)

        if upsample:
            self.conv2 = nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=2, stride=2, bias=False)
        else:
            self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.act2 = nn.PReLU(mid_ch)

        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.dropout = nn.Dropout2d(p=dropout)
        self.act_out = nn.PReLU(out_ch)

        # projection if channels differ on residual
        self.need_proj = (in_ch != out_ch) and (not downsample) and (not upsample)
        if self.need_proj:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            self.bn_proj = nn.BatchNorm2d(out_ch)

    def forward(self, x, pool_indices=None, output_size=None):
        identity = x

        if self.downsample:
            identity, indices = self.pool(identity)
        elif self.upsample:
            # unpool not done here; handled in ENet using maxunpool
            indices = None

        # ext branch
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.dropout(x)

        if self.downsample:
            # identity already pooled, but channels may differ
            if identity.shape[1] != x.shape[1]:
                pad_ch = x.shape[1] - identity.shape[1]
                identity = F.pad(identity, (0, 0, 0, 0, 0, pad_ch))
            out = x + identity
            out = self.act_out(out)
            return out, indices

        if self.upsample:
            # identity is upsampled outside; x is out_ch already
            out = x + identity
            out = self.act_out(out)
            return out

        # regular residual
        if self.need_proj:
            identity = self.proj(identity)
            identity = self.bn_proj(identity)

        out = x + identity
        out = self.act_out(out)
        return out


class ENet(nn.Module):
    """
    Lightweight ENet for semantic segmentation.
    Output stride ~8.
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.initial = InitialBlock(3, 16)

        # Encoder Stage 1 (down to 1/4)
        self.down1 = Bottleneck(16, 64, downsample=True, dropout=0.01)
        self.reg1 = nn.Sequential(
            Bottleneck(64, 64, dropout=0.01),
            Bottleneck(64, 64, dropout=0.01),
            Bottleneck(64, 64, dropout=0.01),
            Bottleneck(64, 64, dropout=0.01),
        )

        # Encoder Stage 2 (down to 1/8)
        self.down2 = Bottleneck(64, 128, downsample=True, dropout=0.1)
        self.reg2 = nn.Sequential(
            Bottleneck(128, 128, dropout=0.1),
            Bottleneck(128, 128, dropout=0.1),
            Bottleneck(128, 128, dropout=0.1),
            Bottleneck(128, 128, dropout=0.1),
            Bottleneck(128, 128, dropout=0.1),
            Bottleneck(128, 128, dropout=0.1),
            Bottleneck(128, 128, dropout=0.1),
            Bottleneck(128, 128, dropout=0.1),
        )

        # Decoder: up to 1/4
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.up2_proj = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.dec2 = nn.Sequential(
            Bottleneck(64, 64, dropout=0.1),
            Bottleneck(64, 64, dropout=0.1),
        )

        # Decoder: up to 1/2
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.up1_proj = nn.Conv2d(64, 16, kernel_size=1, bias=False)
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
        )
        self.dec1 = Bottleneck(16, 16, dropout=0.1)

        # Final upsample to input size
        self.fullconv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, bias=True)

    def forward(self, x):
        x0 = self.initial(x)  # 1/2

        x1, ind1 = self.down1(x0)  # 1/4
        x1 = self.reg1(x1)

        x2, ind2 = self.down2(x1)  # 1/8
        x2 = self.reg2(x2)

        # Decoder up: 1/8 -> 1/4 using stored indices ind2
        # Prepare identity branch for residual: unpool + channel proj
        id2 = self.unpool2(x2, ind2, output_size=x1.size())
        id2 = self.up2_proj(id2)

        x = self.up2(x2)
        x = self.unpool2(x, ind2, output_size=x1.size())
        # residual add in Bottleneck-like manner
        x = x + id2
        x = F.prelu(x, torch.tensor(0.25, device=x.device))  # lightweight activation
        x = self.dec2(x)

        # Decoder up: 1/4 -> 1/2 using stored indices ind1
        id1 = self.unpool1(x, ind1, output_size=x0.size())
        id1 = self.up1_proj(id1)

        x = self.up1(x)
        x = self.unpool1(x, ind1, output_size=x0.size())
        x = x + id1
        x = F.prelu(x, torch.tensor(0.25, device=x.device))
        x = self.dec1(x)

        # 1/2 -> 1/1
        x = self.fullconv(x)
        return x
