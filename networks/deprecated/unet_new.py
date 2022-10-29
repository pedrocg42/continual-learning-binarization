from typing import List, Union

import torch
import torch.nn as nn

from networks.layers import OutConv, Up


class UnetDKVB(nn.Module):
    def __init__(
        self,
        pretrained_encoder: str,
        hidden_dims_decoder: List[int],
        embedding_dim: int = 1024,
        freeze_encoder: bool = True,
        freeze_decoder: bool = False,
        **kwargs,
    ):
        super(UnetDKVB, self).__init__()

        self.pretrained_encoder = pretrained_encoder
        self.hidden_dims_decoder = hidden_dims_decoder
        self.embedding_dim = embedding_dim
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder

        if self.pretrained_encoder == "dino_resnet_50":
            encoder_raw = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")

        if self.freeze_encoder:
            # Freezing the encoder
            for param in encoder_raw.parameters():
                param.requires_grad = False

        # Getting all the model up to the second to last bottleneck block
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    encoder_raw.conv1,
                    encoder_raw.bn1,
                    encoder_raw.relu,
                    encoder_raw.maxpool,
                ),
                encoder_raw.layer1,
                encoder_raw.layer2,
                encoder_raw.layer3,
            ]
        )

        # Building decoder
        self.decoder = nn.ModuleList()
        self.decoder.append(Up(self.embedding_dim, self.hidden_dims_decoder[0]))
        for i in range(len(self.hidden_dims_decoder) - 1):
            self.decoder.append(Up(self.hidden_dims_decoder[i], self.hidden_dims_decoder[i + 1]))
        self.decoder_out = nn.Sequential(OutConv(self.hidden_dims_decoder[-1], 1), nn.Sigmoid())

        if self.freeze_decoder:
            # Freezing the encoder
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, x):

        encoder_results = []
        if self.freeze_encoder:
            with torch.no_grad():
                for module in self.encoder:
                    encoder_output = encoder_results[-1] if len(encoder_results) != 0 else x
                    encoder_results.append(module(encoder_output))
        else:
            for module in self.encoder:
                if encoder_output == None:
                    encoder_output = x
                else:
                    encoder_output = encoder_results[-1]
                encoder_results.append(module(encoder_output))

        decoder_output = None
        for i in range(len(self.decoder)):
            if decoder_output == None:
                decoder_output = encoder_results[-1]
            decoder_output = self.decoder[i](decoder_output, encoder_results[-(i + 2)])

        return self.decoder_out(decoder_output)
