import torch
from torch import nn
import torch.nn.functional as F

class SoftGate(nn.Module):
    """Channel-wise gating via a sigmoid."""
    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.sigmoid(self.conv(self.avg_pool(x)))
        return x * gate


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Compute simple Sobel edges for each channel."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])
    return torch.sqrt(gx ** 2 + gy ** 2)


class EdgeAwareCrossAttention(nn.Module):
    """Cross attention between feature and edge map."""
    def __init__(self, channels: int):
        super().__init__()
        inter = max(1, channels // 8)
        self.query = nn.Conv2d(channels, inter, 1)
        self.key = nn.Conv2d(channels, inter, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        q = self.query(feat).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(edge).view(B, -1, H * W)
        attn = self.softmax(torch.bmm(q, k))
        v = self.value(edge).view(B, C, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return out + feat


class BiFusionNeck(nn.Module):
    """Fuse global and detail branches with optional edge attention."""
    def __init__(self, channels: int, out_channels: int = None, use_edge_attention: bool = False):
        super().__init__()
        out_channels = out_channels or channels
        # group convolution with groups=C to reduce operations
        self.dwconv = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels,
        )
        self.pwconv = nn.Conv2d(channels * 2, out_channels, kernel_size=1)
        self.gate = SoftGate(out_channels)
        self.use_edge_attention = use_edge_attention
        if use_edge_attention:
            self.edge_proj = nn.Conv2d(channels, out_channels, 1)
            self.edge_att = EdgeAwareCrossAttention(out_channels)
        self.gamma = nn.Parameter(torch.ones(out_channels) * 0.1)

    def forward(self, global_feat: torch.Tensor, detail_feat: torch.Tensor) -> torch.Tensor:
        global_feat = F.interpolate(global_feat, size=detail_feat.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([global_feat, detail_feat], dim=1)
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.gate(x)
        if self.use_edge_attention:
            edge = sobel_edges(detail_feat)
            edge = self.edge_proj(edge)
            x = self.edge_att(x, edge)
        x = x * self.gamma.view(1, -1, 1, 1)
        return x
