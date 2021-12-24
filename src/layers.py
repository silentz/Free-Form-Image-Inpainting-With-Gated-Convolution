import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


class GatedConv2d(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: Union[int, Tuple[int, int]],
                       stride: Union[int, Tuple[int, int]],
                       padding: Union[int, Tuple[int, int]],
                       dilation: Union[int, Tuple[int, int]],
                       batch_norm: bool = True,
                       activation: nn.Module = nn.LeakyReLU(negative_slope=0.2),
                ):
        super().__init__()

        self._image_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                bias=True,
            )

        self._mask_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                bias=True,
            )

        self._batch_norm = nn.BatchNorm2d(
                num_features=out_channels,
            )

        self._activation = activation
        self._batch_norm_flag = batch_norm

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        image_out = self._image_conv(input)
        mask_out = self._mask_conv(input)
        gated_out = self._activation(image_out) * torch.sigmoid(mask_out)

        if self._batch_norm_flag:
            gated_out = self._batch_norm(gated_out)

        return gated_out


class GatedUpsampleConv2d(GatedConv2d):

    def __init__(self, *args, scale_factor: float = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._scale_factor = scale_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scaled = F.interpolate(input, scale_factor=self._scale_factor)
        result = super().forward(scaled)
        return result
