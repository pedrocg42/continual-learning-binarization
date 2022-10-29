from typing import Dict, List
import os

import torch
import torch.nn as nn
from discrete_key_value_bottleneck_pytorch import DiscreteKeyValueBottleneck
from copy import deepcopy

from networks.layers import OutConv, UpSimple
import config


class DKVBBin(nn.Module):
    def __init__(
        self,
        encoder_model_experiment: Dict,
        hidden_dims_decoder: List[int],
        n_channels: int = 3,
        bilinear: bool = False,
        codebook_size: int = 512,
        num_codebooks: int = 1,
        commitment_weight: float = 0.0,
        decay: float = 0.99,
        threshold_ema_dead_code: float = 2,
        **kwargs,
    ):
        super(DKVBBin, self).__init__()

        self.encoder_model_experiment = encoder_model_experiment
        self.hidden_dims_decoder = hidden_dims_decoder
        self.n_channels = n_channels
        self.bilinear = bilinear

        # Vector Quantizer Config
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.threshold_ema_dead_code = threshold_ema_dead_code

        factor = 2 if bilinear else 1

        # Loading encoder weights
        model = self.encoder_model_experiment["architecture"](**self.encoder_model_experiment)
        model_file_path = os.path.join(config.models_path, f"{self.encoder_model_experiment['name']}.pt")
        model.load_state_dict(torch.load(model_file_path))

        self.encoder = deepcopy(model.encoder)
        del model
        # Freezing the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.key_value_bottleneck = DiscreteKeyValueBottleneck(
            dim=self.hidden_dims_decoder[0],  # input dimension
            codebook_dim=self.hidden_dims_decoder[0] // self.num_codebooks,
            num_memory_codebooks=self.num_codebooks,  # number of memory codebook
            num_memories=self.codebook_size,  # number of memories
            dim_memory=self.hidden_dims_decoder[0] // self.num_codebooks,  # dimension of the output memories
            decay=self.decay,  # the exponential moving average decay, lower means the keys will change faster
            threshold_ema_dead_code=self.threshold_ema_dead_code,  # (0.8·batch-size·h·w·mz/num-pairs)
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
        embedding = self.encoder(x)

        original_shape = embedding.shape

        embedding = torch.reshape(
            embedding, (original_shape[0], original_shape[1], original_shape[2] * original_shape[3])
        )  # B, Dim, H, W -> B, Dim, N
        embedding = torch.permute(embedding, (0, 2, 1))  # B, Dim, N -> B, N, Dim

        memories = self.key_value_bottleneck(embedding)  # quantized, indices, commitment loss

        memories = torch.permute(memories, (0, 2, 1))  # B, N, Dim -> B, Dim, N
        memories = torch.reshape(
            memories, (original_shape[0], original_shape[1], original_shape[2], original_shape[3])
        )  # B, Dim, N -> B, Dim, H, W

        return self.decoder(memories)
