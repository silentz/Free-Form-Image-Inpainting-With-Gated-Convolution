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
                nn.Tanh(),
            )

        self._refine = nn.Sequential(
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

                # attention
                SelfAttention(channels=self._channels * 3),

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
                nn.Tanh(),
            )

    def _normalize(self, input: torch.Tensor) -> torch.Tensor:
        return (input - 127.5) / 127.5

    def _denormalize(self, input: torch.Tensor) -> torch.Tensor:
        return (input * 127.5) + 127.5

    def forward(self, image: torch.Tensor, mask: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        # preprocess
        image = self._normalize(image)
        masked_image = image * (1 - mask)

        # stage 1
        X_coarse_input = torch.cat([masked_image, mask], dim=1)
        X_coarse = self._coarse_net(X_coarse_input)

        # stage 2
        X_refine_image = image * (1 - mask) + X_coarse * mask
        X_refine_input = torch.cat([X_refine_image, mask], dim=1)
        X_refine = self._refine(X_refine_input)

        return self._denormalize(X_coarse), self._denormalize(X_refine)


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
