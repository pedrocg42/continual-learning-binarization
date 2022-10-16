from typing import List

import torch.nn as nn
from networks.layers import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(
        self,
        hidden_dims_encoder: List[int],
        hidden_dims_decoder: List[int],
        n_channels: int = 3,
        bilinear: bool = False,
        **kwargs
    ):
        super(UNet, self).__init__()
        self.hidden_dims_encoder = hidden_dims_encoder
        self.hidden_dims_decoder = hidden_dims_decoder
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, self.hidden_dims_encoder[0])
        self.down1 = Down(self.hidden_dims_encoder[0], self.hidden_dims_encoder[1])
        self.down2 = Down(self.hidden_dims_encoder[1], self.hidden_dims_encoder[2])
        self.down3 = Down(self.hidden_dims_encoder[2], self.hidden_dims_encoder[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(self.hidden_dims_encoder[3], self.hidden_dims_encoder[4] // factor)
        self.up1 = Up(self.hidden_dims_decoder[0], self.hidden_dims_decoder[1] // factor, bilinear)
        self.up2 = Up(self.hidden_dims_decoder[1], self.hidden_dims_decoder[2] // factor, bilinear)
        self.up3 = Up(self.hidden_dims_decoder[2], self.hidden_dims_decoder[3] // factor, bilinear)
        self.up4 = Up(self.hidden_dims_decoder[3], self.hidden_dims_decoder[4], bilinear)
        self.outc = OutConv(self.hidden_dims_decoder[4], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = self.sigmoid(logits)
        return out
