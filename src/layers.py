import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple


class GatedConv2d(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: Union[int, Tuple[int, int]],
                       stride: Union[int, Tuple[int, int]] = 1,
                       padding: Union[int, str, Tuple[int, int]] = 0,
                       dilation: Union[int, Tuple[int, int]] = 1,
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

        self._activation = activation if (activation is not None) else (lambda x: x)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        image_out = self._image_conv(input)
        mask_out = self._mask_conv(input)
        gated_out = self._activation(image_out) * torch.sigmoid(mask_out)
        return gated_out


class GatedUpsampleConv2d(GatedConv2d):

    def __init__(self, *args, scale_factor: float = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._scale_factor = scale_factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scaled = F.interpolate(input, scale_factor=self._scale_factor)
        result = super().forward(scaled)
        return result


class SpectralNormConv2d(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int,
                       kernel_size: Union[int, Tuple[int, int]],
                       stride: Union[int, Tuple[int, int]] = 1,
                       padding: Union[int, str, Tuple[int, int]] = 0,
                       dilation: Union[int, Tuple[int, int]] = 1,
                       activation: nn.Module = nn.LeakyReLU(negative_slope=0.2),
                ):
        super().__init__()

        self._conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                bias=True,
            )

        self._conv = nn.utils.spectral_norm(self._conv)
        self._activation = activation

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self._conv(input)
        out = self._activation(out)
        return out


class SelfAttention(nn.Module):

    def __init__(self, channels: int, attention_map: bool = False):
        super().__init__()

        self._in_channels  = channels
        self._out_channels = channels // 8
        self._attn_map = attention_map

        self._conv_query = nn.Conv2d(
                in_channels=self._in_channels,
                out_channels=self._out_channels,
                kernel_size=1,
            )

        self._conv_key = nn.Conv2d(
                in_channels=self._in_channels,
                out_channels=self._out_channels,
                kernel_size=1,
            )

        self._conv_value = nn.Conv2d(
                in_channels=self._in_channels,
                out_channels=self._in_channels,
                kernel_size=1,
            )

        self._gamma = nn.Parameter(
                data=torch.zeros(1),
            )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = input.shape

        # query, key, value
        query = self._conv_query(input) # [batch, ch / 8, h, w]
        key   = self._conv_key(input)   # [batch, ch / 8, h, w]
        value = self._conv_value(input) # [batch, ch, h, w]

        # reshape query, key, value
        query = query.view(batch_size, -1, height * width) # [batch, ch / 8, h * w]
        key   = key.view(batch_size, -1, height * width)   # [batch, ch / 8, h * w]
        value = value.view(batch_size, -1, height * width) # [batch, ch, h * w]

        # attention map
        query_T = query.permute(0, 2, 1)          # [batch, h * w, ch / 8]
        energy = torch.bmm(query_T, key)          # [batch, h * w, h * w]
        attention = torch.softmax(energy, dim=-1) # [batch, h * w, h * w]

        # self-attention output
        attention_T = attention.permute(0, 2, 1)
        result = torch.bmm(value, attention_T)
        result = result.view(batch_size, channels, height, width)

        # gamma parameter
        output = self._gamma * result + input

        if not self._attn_map:
            return output

        return output, attention
