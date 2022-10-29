from typing import List

import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

from networks.layers import DoubleConv, Down, OutConv, UpSimple


class VQBin(nn.Module):
    def __init__(
        self,
        hidden_dims_encoder: List[int],
        hidden_dims_decoder: List[int],
        n_channels: int = 3,
        bilinear: bool = False,
        codebook_size: int = 512,
        num_codebooks: int = 1,
        commitment_weight: float = 0.0,
        decay: float = 0.99,
        **kwargs
    ):
        super(VQBin, self).__init__()

        self.hidden_dims_encoder = hidden_dims_encoder
        self.hidden_dims_decoder = hidden_dims_decoder
        self.n_channels = n_channels
        self.bilinear = bilinear

        # Vector Quantizer Config
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.commitment_weight = commitment_weight
        self.decay = decay

        factor = 2 if bilinear else 1

        self.encoder = nn.Sequential(
            DoubleConv(n_channels, self.hidden_dims_encoder[0]),
            Down(self.hidden_dims_encoder[0], self.hidden_dims_encoder[1]),
            Down(self.hidden_dims_encoder[1], self.hidden_dims_encoder[2]),
            Down(self.hidden_dims_encoder[2], self.hidden_dims_encoder[3]),
            Down(self.hidden_dims_encoder[3], self.hidden_dims_encoder[4] // factor),
        )

        self.vector_quantizer = VectorQuantize(
            dim=self.hidden_dims_encoder[4] // factor,
            codebook_dim=self.hidden_dims_encoder[4] // factor // self.num_codebooks,
            codebook_size=self.codebook_size,
            heads=self.num_codebooks,
            separate_codebook_per_head=True,
            commitment_weight=self.commitment_weight,
            decay=self.decay,
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

        quantized, _, _ = self.vector_quantizer(embedding)  # quantized, indices, commitment loss

        quantized = torch.permute(quantized, (0, 2, 1))  # B, N, Dim -> B, Dim, N
        quantized = torch.reshape(
            quantized, (original_shape[0], original_shape[1], original_shape[2], original_shape[3])
        )  # B, Dim, N -> B, Dim, H, W

        return self.decoder(quantized)
