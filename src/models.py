import torch
import torch.nn as nn
from typing import Tuple

from .layers import (
        GatedConv2d,
        GatedUpsampleConv2d,
        SelfAttention,
        SpectralNormConv2d,
    )


class Generator(nn.Module):

    def __init__(self, channels: int = 4):
        super().__init__()

        self._in_channels = channels
        self._channels = 32

        self._coarse_net = nn.Sequential(
                # hw: 256
                GatedConv2d(in_channels=self._in_channels, out_channels=self._channels,
                            kernel_size=5, padding='same'),
                # hw: 128
                GatedConv2d(in_channels=self._channels, out_channels=self._channels * 2,
                            kernel_size=4, padding=1, stride=2),
                GatedConv2d(in_channels=self._channels * 2, out_channels=self._channels * 2,
                            kernel_size=3, padding='same'),
                #  hw: 64
                GatedConv2d(in_channels=self._channels * 2, out_channels=self._channels * 3,
                            kernel_size=4, padding=1, stride=2),
                GatedConv2d(in_channels=self._channels * 3, out_channels=self._channels * 3,
                            kernel_size=3, padding='same'),
                GatedConv2d(in_channels=self._channels * 3, out_channels=self._channels * 3,
                            kernel_size=3, padding='same'),
                # dilated conv track start
                GatedConv2d(in_channels=3 * self._channels, out_channels=3 * self._channels,
                            kernel_size=3, dilation=2, padding='same'),
                GatedConv2d(in_channels=3 * self._channels, out_channels=3 * self._channels,
                            kernel_size=3, dilation=4, padding='same'),
                GatedConv2d(in_channels=3 * self._channels, out_channels=3 * self._channels,
                            kernel_size=3, dilation=8, padding='same'),
                GatedConv2d(in_channels=3 * self._channels, out_channels=3 * self._channels,
                            kernel_size=3, dilation=16, padding='same'),
                # dilation conv track finish
                GatedConv2d(in_channels=3 * self._channels, out_channels=3 * self._channels,
                            kernel_size=3, padding='same'),
                GatedConv2d(in_channels=3 * self._channels, out_channels=3 * self._channels,
                            kernel_size=3, padding='same'),
                # hw: 128
                GatedUpsampleConv2d(in_channels=3 * self._channels, out_channels=2 * self._channels,
                                    kernel_size=3, padding='same'),
                GatedConv2d(in_channels=2 * self._channels, out_channels=2 * self._channels,
                            kernel_size=3, padding='same'),
                # hw 256
                GatedUpsampleConv2d(in_channels=2 * self._channels, out_channels=3,
                                    kernel_size=3, padding='same', activation=None),
            )

        self._refine_head = nn.Sequential(
                # hw: 256
                GatedConv2d(in_channels=self._in_channels, out_channels=self._channels,
                            kernel_size=5, padding='same'),
                # hw: 128
                GatedConv2d(in_channels=self._channels, out_channels=self._channels * 2,
                            kernel_size=4, padding=1, stride=2),
                GatedConv2d(in_channels=self._channels * 2, out_channels=self._channels * 2,
                            kernel_size=3, padding='same'),
                #  hw: 64
                GatedConv2d(in_channels=self._channels * 2, out_channels=self._channels * 3,
                            kernel_size=4, padding=1, stride=2),
                GatedConv2d(in_channels=self._channels * 3, out_channels=self._channels * 3,
                            kernel_size=3, padding='same'),
            )

        self._refine_attn = SelfAttention(channels=self._channels * 3)

        self._refine_attn_tail = nn.Sequential(
                GatedConv2d(in_channels=3 * self._channels, out_channels=3 * self._channels,
                            kernel_size=3, padding='same'),
                GatedConv2d(in_channels=3 * self._channels, out_channels=3 * self._channels,
                            kernel_size=3, padding='same'),
                GatedConv2d(in_channels=3 * self._channels, out_channels=3 * self._channels,
                            kernel_size=3, padding='same'),
                GatedConv2d(in_channels=3 * self._channels, out_channels=3 * self._channels,
                            kernel_size=3, padding='same'),
                GatedConv2d(in_channels=3 * self._channels, out_channels=3 * self._channels,
                            kernel_size=3, padding='same'),
                # hw: 128
                GatedUpsampleConv2d(in_channels=3 * self._channels, out_channels=2 * self._channels,
                                    kernel_size=3, padding='same'),
                GatedConv2d(in_channels=2 * self._channels, out_channels=2 * self._channels,
                            kernel_size=3, padding='same'),
                # hw 256
                GatedUpsampleConv2d(in_channels=2 * self._channels, out_channels=3,
                                    kernel_size=3, padding='same', activation=None),
            )

    def _normalize(self, input: torch.Tensor) -> torch.Tensor:
        return (input - 127.5) / 127.5

    def _denormalize(self, input: torch.Tensor) -> torch.Tensor:
        return (input * 127.5) + 127.5

    def forward(self, image: torch.Tensor, mask: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # preprocess
        image = self._normalize(image)
        masked_image = image * (1 - mask) + mask

        # stage 1
        X_coarse_input = torch.cat([masked_image, mask], dim=1)
        X_coarse = self._coarse_net(X_coarse_input)
        X_coarse = torch.clamp(X_coarse, -1, 1)

        # stage 2
        X_refine_image = image * (1 - mask) + X_coarse * mask
        X_refine_input = torch.cat([X_refine_image, mask], dim=1)
        X_refine_head = self._refine_head(X_refine_input)
        X_refine_attn, attn_map = self._refine_attn(X_refine_head)
        X_refine_attn = self._refine_attn_tail(X_refine_attn)
        X_refine = torch.clamp(X_refine_attn, -1, 1)

        return self._denormalize(X_coarse), self._denormalize(X_refine), attn_map


class Discriminator(nn.Module):

    def __init__(self, channels: int = 4):
        super().__init__()

        self._in_channels = channels
        self._channels = 64

        self._layers = nn.Sequential(
                SpectralNormConv2d(in_channels=self._in_channels, out_channels=self._channels,
                                   kernel_size=4, stride=2, padding=1),
                SpectralNormConv2d(in_channels=self._channels, out_channels=self._channels * 2,
                                   kernel_size=4, stride=2, padding=1),
                SpectralNormConv2d(in_channels=self._channels * 2, out_channels=self._channels * 3,
                                   kernel_size=4, stride=2, padding=1),
                SpectralNormConv2d(in_channels=self._channels * 3, out_channels=self._channels * 3,
                                   kernel_size=4, stride=2, padding=1),
                SpectralNormConv2d(in_channels=self._channels * 3, out_channels=self._channels * 3,
                                   kernel_size=4, stride=2, padding=1),
                SpectralNormConv2d(in_channels=self._channels * 3, out_channels=self._channels * 3,
                                   kernel_size=4, stride=2, padding=1),
            )

    def _normalize(self, input: torch.Tensor) -> torch.Tensor:
        return (input - 127.5) / 127.5

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        image = self._normalize(image)
        X = torch.cat([image, mask], dim=1)
        return self._layers(X)
