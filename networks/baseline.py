from typing import List

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from networks.layers import DoubleConv, Down, OutConv, UpSimple


class Baseline(nn.Module):
    def __init__(
        self,
        hidden_dims_encoder: List[int],
        hidden_dims_decoder: List[int],
        n_channels: int = 3,
        bilinear: bool = False,
        **kwargs
    ):
        super(Baseline, self).__init__()
        self.hidden_dims_encoder = hidden_dims_encoder
        self.hidden_dims_decoder = hidden_dims_decoder
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.encoder = nn.Sequential(
            DoubleConv(n_channels, self.hidden_dims_encoder[0]),
            Down(self.hidden_dims_encoder[0], self.hidden_dims_encoder[1]),
            Down(self.hidden_dims_encoder[1], self.hidden_dims_encoder[2]),
            Down(self.hidden_dims_encoder[2], self.hidden_dims_encoder[3]),
            Down(self.hidden_dims_encoder[3], self.hidden_dims_encoder[4] // factor),
        )

        self.decoder = nn.Sequential(
            UpSimple(self.hidden_dims_decoder[0], self.hidden_dims_decoder[1] // factor, bilinear),
            UpSimple(self.hidden_dims_decoder[1], self.hidden_dims_decoder[2] // factor, bilinear),
            UpSimple(self.hidden_dims_decoder[2], self.hidden_dims_decoder[3] // factor, bilinear),
            UpSimple(self.hidden_dims_decoder[3], self.hidden_dims_decoder[4], bilinear),
            OutConv(self.hidden_dims_decoder[4], 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
