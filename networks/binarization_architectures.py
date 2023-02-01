from typing import List, Union

import torch
import torch.nn as nn
from discrete_key_value_bottleneck_pytorch import DiscreteKeyValueBottleneck
from vector_quantize_pytorch import VectorQuantize

from networks.layers import OutConv, UpSimple


class BinarizationBaseModel(nn.Module):
    def __init__(
        self,
        pretrained_encoder: str,
        hidden_dims_decoder: List[int],
        embedding_dim: int = 1024,
        freeze_encoder: bool = True,
        freeze_decoder: bool = False,
        **kwargs,
    ):
        super(BinarizationBaseModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.pretrained_encoder = pretrained_encoder
        self.hidden_dims_decoder = hidden_dims_decoder
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder

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

        # Building decoder
        decoder_module_list = nn.ModuleList()
        decoder_module_list.append(
            UpSimple(self.embedding_dim, self.hidden_dims_decoder[0])
        )
        for i in range(len(self.hidden_dims_decoder) - 1):
            decoder_module_list.append(
                UpSimple(self.hidden_dims_decoder[i], self.hidden_dims_decoder[i + 1])
            )
        decoder_module_list.append(OutConv(self.hidden_dims_decoder[-1], 1))
        decoder_module_list.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_module_list)

    def on_train_epoch_start(self, *args, **kwargs):
        self.train()

    def on_val_epoch_start(self, *args, **kwargs):
        self.eval()

    @property
    def device(self):
        return next(self.parameters()).device


class BinarizationBaseline(BinarizationBaseModel):
    def forward(self, x):

        # Encoder inference
        if self.freeze_encoder:
            with torch.no_grad():
                embeddings = self.encoder(x)
                embeddings.detach_()
        else:
            embeddings = self.encoder(x)

        # Decoder inference
        output = self.decoder(embeddings)

        return output


class BottleneckBinarizer(BinarizationBaseModel):
    def __init__(
        self,
        codebook_size: int = 8192,
        num_codebooks: int = 1,
        vq_decay: float = 0.99,
        threshold_ema_dead_code: int = 1024,
        value_dimension: Union[int, str] = "same",
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.value_dimension = value_dimension
        self.vq_decay = vq_decay
        self.threshold_ema_dead_code = threshold_ema_dead_code

        if isinstance(self.value_dimension, str):
            self.value_dimension = self.embedding_dim // self.num_codebooks

    def forward(self, x):

        # Encoder inference
        if self.freeze_encoder:
            with torch.no_grad():
                embeddings = self.encoder(x)
                embeddings.detach_()
        else:
            embeddings = self.encoder(x)

        # Bottleneck
        memories = self._process_bottleneck(embeddings=embeddings, x=x)

        # Decoder inference
        return self.decoder(memories)

    def _bottleneck_inference(embeddings: torch.Tensor):
        raise NotImplementedError("Child classes should implement this method.")

    def _process_bottleneck(self, embeddings: torch.Tensor, x: torch.Tensor):
        encoder_output_size = embeddings.shape[-1]
        batch_size = x.size()[0]

        embeddings = torch.reshape(
            embeddings,
            (embeddings.shape[0], self.embedding_dim, encoder_output_size**2),
        )  # B, Dim, H, W -> B, Dim, N
        embeddings = torch.permute(embeddings, (0, 2, 1))  # B, Dim, N -> B, N, Dim

        memories = self._bottleneck_inference(
            embeddings
        )  # quantized, indices, commitment loss

        memories = torch.permute(memories, (0, 2, 1))  # B, N, Dim -> B, Dim, N
        memories = torch.reshape(
            memories,
            (
                batch_size,
                self.embedding_dim,
                encoder_output_size,
                encoder_output_size,
            ),
        )  # B, Dim, N -> B, Dim, H, W

        return memories

    def on_key_initialization_start(self):
        self.train()
        self.encoder.train(False)
        self.decoder.train(False)


class VectorQuantizerBinarizer(BottleneckBinarizer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.bottleneck = VectorQuantize(
            dim=self.embedding_dim,
            codebook_dim=self.embedding_dim // self.num_codebooks,
            codebook_size=self.codebook_size,
            heads=self.num_codebooks,
            separate_codebook_per_head=True,
            decay=self.vq_decay,
            threshold_ema_dead_code=self.threshold_ema_dead_code,  # (0.8·batch-size·h·w·mz/num-pairs)
        )

    def _bottleneck_inference(self, embeddings: torch.Tensor):
        memories, _, _ = self.bottleneck(embeddings)
        return memories

    def on_train_epoch_start(self, *args, **kwargs):
        # Only training decoder after key initialization
        self.eval()
        self.decoder.train()


class DkvbBinarizer(BottleneckBinarizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.bottleneck = DiscreteKeyValueBottleneck(
            dim=self.embedding_dim,  # input dimension
            codebook_dim=self.embedding_dim // self.num_codebooks,
            num_memory_codebooks=self.num_codebooks,  # number of memory codebook
            num_memories=self.codebook_size,  # number of memories
            dim_memory=self.embedding_dim
            // self.num_codebooks,  # dimension of the output memories
            decay=self.vq_decay,  # the exponential moving average decay, lower means the keys will change faster
            threshold_ema_dead_code=self.threshold_ema_dead_code,  # (0.8·batch-size·h·w·mz/num-pairs)
        )

    def _bottleneck_inference(self, embeddings: torch.Tensor):
        return self.bottleneck(embeddings)

    def on_train_epoch_start(self, i_dataset: int):
        # Only training decoder and values after key initialization
        self.eval()
        if i_dataset == 0:
            # Training values and decoder for the first dataset
            self.decoder.train()
            for param in self.decoder.parameters():
                param.requires_grad = True
        else:
            # Only training values next datasets
            for param in self.decoder.parameters():
                param.requires_grad = False
