import torch
from torch import nn
from .vmamba import SS2D

class VisionMambaEncoder(nn.Module):
    """Simple Vision Mamba encoder using 1D state space convolution."""

    def __init__(self, in_chans=3, embed_dim=128, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.mamba = SS2D(d_model=embed_dim, dropout=0, d_state=16)
        self.out_conv = nn.Conv2d(embed_dim, 128, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        x = self.bn(x)
        x = x.permute(0, 2, 3, 1)
        x = self.mamba(x)
        x = x.permute(0, 3, 1, 2)
        x = self.out_conv(x)
        return x


class ECA(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class PIDNetDetailBlock(nn.Module):
    """Depthwise separable conv block with ECA attention."""

    def __init__(self, in_chans, out_chans, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_chans, in_chans, kernel_size=3, stride=stride, padding=1, groups=in_chans, bias=False)
        self.pw = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.eca = ECA(out_chans)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.eca(x)
        x = self.relu(x)
        return x


class PIDNetDetailEncoder(nn.Module):
    """PIDNet detail branch encoder with consistent patch size."""

    def __init__(self, in_chans=3, embed_dim=128, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.block1 = PIDNetDetailBlock(embed_dim, embed_dim)
        self.block2 = PIDNetDetailBlock(embed_dim, embed_dim)
        self.out_conv = nn.Conv2d(embed_dim, 128, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        x = self.bn(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.out_conv(x)
        return x


class DualBackboneEncoder(nn.Module):
    """Combine Vision Mamba and PIDNet detail encoders."""

    def __init__(self, in_chans=3, embed_dim=128, patch_size=4):
        super().__init__()
        self.vmamba = VisionMambaEncoder(in_chans, embed_dim, patch_size)
        self.pid_detail = PIDNetDetailEncoder(in_chans, embed_dim, patch_size)

    def forward(self, x):
        feat_vmamba = self.vmamba(x)
        feat_pid = self.pid_detail(x)
        return feat_vmamba, feat_pid
