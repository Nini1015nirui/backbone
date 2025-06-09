import torch
from torch import nn
import torch.nn.functional as F

class MultiTaskHead(nn.Module):
    """Multi-task prediction head.

    Takes a 128-channel feature map and outputs segmentation, edge map
    and thickness estimation with uncertainty.
    """

    def __init__(self, in_channels: int = 128, seg_classes: int = 1, hidden: int = 64):
        super().__init__()
        # segmentation branch: 1x1 conv then pixel shuffle up x4 (factor 4)
        self.seg_conv = nn.Conv2d(in_channels, seg_classes * 16, kernel_size=1)
        self.seg_shuffle = nn.PixelShuffle(4)

        # edge branch: 1x1 conv -> sigmoid -> bilinear upsample x4
        self.edge_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

        # thickness branch: global average pooling -> MLP
        self.thick_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2)  # [thickness, log_sigma2]
        )

    def forward(self, x: torch.Tensor):
        # segmentation output
        seg = self.seg_conv(x)
        seg = self.seg_shuffle(seg)

        # edge output
        edge = torch.sigmoid(self.edge_conv(x))
        edge = F.interpolate(edge, scale_factor=4, mode="bilinear", align_corners=False)

        # thickness output
        gap = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        thick_params = self.thick_mlp(gap)
        thickness = thick_params[:, :1]
        log_sigma2 = thick_params[:, 1:2]

        return seg, edge, thickness, log_sigma2
