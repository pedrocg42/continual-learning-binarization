from typing import List, Union

import torch
import torch.nn as nn
from discrete_key_value_bottleneck_pytorch import DiscreteKeyValueBottleneck
from vector_quantize_pytorch import VectorQuantize

from networks.layers import OutConv, UpSimple


class DKVBBin(nn.Module):
    def __init__(
        self,
        architecture_type: str,
        pretrained_encoder: str,
        hidden_dims_decoder: List[int],
        embedding_dim: int = 1024,
        codebook_size: int = 8192,
        num_codebooks: int = 1,
        vq_decay: float = 0.99,
        threshold_ema_dead_code: int = 2,
        value_dimension: Union[int, str] = "same",
        freeze_encoder: bool = True,
        freeze_decoder: bool = False,
        **kwargs,
    ):
        super(DKVBBin, self).__init__()

        self.architecture_type = architecture_type
        self.pretrained_encoder = pretrained_encoder
        self.hidden_dims_decoder = hidden_dims_decoder
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.value_dimension = value_dimension
        self.vq_decay = vq_decay
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder

        # embedding dimension and number of key-value pairs must be divisible by number of codes
        assert (self.embedding_dim % num_codebooks) == 0
        assert (self.codebook_size & num_codebooks) == 0

        if self.pretrained_encoder == "dino_resnet_50":
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

        if self.architecture_type == "discrete_key_value_bottleneck":

            if isinstance(self.value_dimension, str):
                self.value_dimension = self.embedding_dim // self.num_codebooks

            self.key_value_bottleneck = DiscreteKeyValueBottleneck(
                dim=self.embedding_dim,  # input dimension
                codebook_dim=self.embedding_dim // self.num_codebooks,
                num_memory_codebooks=self.num_codebooks,  # number of memory codebook
                num_memories=self.codebook_size,  # number of memories
                dim_memory=self.embedding_dim // self.num_codebooks,  # dimension of the output memories
                decay=self.vq_decay,  # the exponential moving average decay, lower means the keys will change faster
                threshold_ema_dead_code=self.threshold_ema_dead_code,  # (0.8·batch-size·h·w·mz/num-pairs)
            )

        elif self.architecture_type == "vector_quantizer":
            self.vector_quantizer = VectorQuantize(
                dim=self.embedding_dim,
                codebook_dim=self.embedding_dim // self.num_codebooks,
                codebook_size=self.codebook_size,
                heads=self.num_codebooks,
                separate_codebook_per_head=True,
                decay=self.vq_decay,
                threshold_ema_dead_code=self.threshold_ema_dead_code,  # (0.8·batch-size·h·w·mz/num-pairs)
            )

        # Building decoder
        decoder_module_list = nn.ModuleList()
        decoder_module_list.append(UpSimple(self.embedding_dim, self.hidden_dims_decoder[0]))
        for i in range(len(self.hidden_dims_decoder) - 1):
            decoder_module_list.append(UpSimple(self.hidden_dims_decoder[i], self.hidden_dims_decoder[i + 1]))
        decoder_module_list.append(OutConv(self.hidden_dims_decoder[-1], 1))
        decoder_module_list.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_module_list)

        if self.freeze_decoder:
            # Freezing the encoder
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, x):

        if self.freeze_encoder:
            with torch.no_grad():
                embeddings = self.encoder(x)
                embeddings.detach_()
        else:
            embeddings = self.encoder(x)

        if self.architecture_type == "discrete_key_value_bottleneck" or self.architecture_type == "vector_quantizer":

            encoder_output_size = embeddings.shape[-1]
            batch_size = x.size()[0]

            embeddings = torch.reshape(
                embeddings, (embeddings.shape[0], self.embedding_dim, encoder_output_size**2)
            )  # B, Dim, H, W -> B, Dim, N
            embeddings = torch.permute(embeddings, (0, 2, 1))  # B, Dim, N -> B, N, Dim

            if self.architecture_type == "discrete_key_value_bottleneck":
                memories = self.key_value_bottleneck(embeddings)
            else:
                memories, _, _ = self.vector_quantizer(embeddings)  # quantized, indices, commitment loss

            memories = torch.permute(memories, (0, 2, 1))  # B, N, Dim -> B, Dim, N
            memories = torch.reshape(
                memories, (batch_size, self.embedding_dim, encoder_output_size, encoder_output_size)
            )  # B, Dim, N -> B, Dim, H, W

            # Processing final output
            return self.decoder(memories)

        elif self.architecture_type == "baseline":  # baseline classifier
            output = self.decoder(embeddings)

        return output
