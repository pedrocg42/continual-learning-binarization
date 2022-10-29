from typing import List, Union

import torch
import torch.nn as nn
from networks.layers import DoubleConv, Down, OutConv, UpSimple


class Baseline(nn.Module):
    def __init__(
        self,
        hidden_dims_decoder: List[int],
        hidden_dims_encoder: List[int] = None,
        pretrained_encoder: str = None,
        freeze_encoder: bool = True,
        n_channels: int = 3,
        bilinear: bool = False,
        **kwargs
    ):
        super(Baseline, self).__init__()
        self.pretrained_encoder = pretrained_encoder
        self.freeze_encoder = freeze_encoder
        self.hidden_dims_encoder = hidden_dims_encoder
        self.hidden_dims_decoder = hidden_dims_decoder
        self.n_channels = n_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Building encoder
        if self.pretrained_encoder is None:
            # Custom encoder

            self.encoder = nn.Sequential(
                DoubleConv(n_channels, self.hidden_dims_encoder[0]),
                Down(self.hidden_dims_encoder[0], self.hidden_dims_encoder[1]),
                Down(self.hidden_dims_encoder[1], self.hidden_dims_encoder[2]),
                Down(self.hidden_dims_encoder[2], self.hidden_dims_encoder[3]),
                Down(self.hidden_dims_encoder[3], self.hidden_dims_encoder[4] // factor),
            )

        elif self.pretrained_encoder == "dino_resnet_50":
            # Pretrainedd encoder
            encoder_raw = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")

            # Getting all the model up to the second to last bottleneck block
            self.encoder = nn.Sequential(
                encoder_raw.conv1,
                encoder_raw.bn1,
                encoder_raw.relu,
                encoder_raw.maxpool,
                encoder_raw.layer1,
                encoder_raw.layer2,
                encoder_raw.layer3,
            )

        if self.freeze_encoder:
            # Freezing the encoder
            for param in self.encoder.parameters():
                param.requires_grad = False

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
