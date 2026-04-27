import torch
import torch.nn as nn

from transformers.models.roformer.modeling_roformer import RoFormerEmbeddings
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

# you can increase this if you plan to support other protocols beyond 1, 6, 17
HIGHEST_TRANSPORT_LAYER_PROTOCOL = 18


class netFoundEmbeddingsWithMeta:
    def __init__(self, config):
        self.metaEmbeddingLayer1 = nn.Linear(config.metaFeatures, 1024)
        self.metaEmbeddingLayer2 = nn.Linear(1024, config.hidden_size)
        self.activation = nn.GELU()
        self.no_meta = config.no_meta
        self.protoEmbedding = nn.Embedding(HIGHEST_TRANSPORT_LAYER_PROTOCOL, config.hidden_size)
        self.compressEmbeddings = nn.Linear(config.hidden_size * 3, config.hidden_size)

    def addMetaEmbeddings(self,
                          embeddings,
                          direction=None,
                          iats=None,
                          bytes=None,
                          pkt_count=None,
                          protocol=None):
        if self.no_meta:
            metaEmbeddings = torch.zeros_like(embeddings)
        else:
            linearLayerDtype = self.metaEmbeddingLayer1.weight.dtype
            metaEmbeddings = torch.concat(
                [
                    direction.unsqueeze(2).to(linearLayerDtype),
                    bytes.unsqueeze(2).to(linearLayerDtype) / 1000,
                    pkt_count.unsqueeze(2).to(linearLayerDtype),
                    iats.unsqueeze(2).to(linearLayerDtype),
                ],
                dim=-1,
            )
            metaEmbeddings = self.metaEmbeddingLayer1(metaEmbeddings)
            metaEmbeddings = self.activation(metaEmbeddings)
            metaEmbeddings = self.metaEmbeddingLayer2(metaEmbeddings)

        embeddings = torch.concat([embeddings, metaEmbeddings], dim=-1)
        protoEmbeddings = self.protoEmbedding(protocol).unsqueeze(1).repeat(1, embeddings.shape[1], 1)
        return self.compressEmbeddings(torch.concat([embeddings, protoEmbeddings], dim=-1))


class netFoundRobertaEmbeddings(RobertaEmbeddings, netFoundEmbeddingsWithMeta):
    def __init__(self, config):
        RobertaEmbeddings.__init__(self, config)
        netFoundEmbeddingsWithMeta.__init__(self, config)

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            direction=None,
            iats=None,
            bytes=None,
            pkt_count=None,
            protocol=None,
    ):
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, self.position_ids)
        embeddings = self.word_embeddings(input_ids)
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.addMetaEmbeddings(embeddings, direction, iats, bytes, pkt_count, protocol)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    @staticmethod
    def create_position_ids_from_input_ids(input_ids, padding_idx, position_ids):
        mask = input_ids.ne(padding_idx).int()
        position_ids = (
                position_ids.repeat(
                    input_ids.shape[0], input_ids.shape[1] // position_ids.shape[1]
                )
                * mask
        )
        return position_ids


class netFoundRoformerEmbeddings(RoFormerEmbeddings, netFoundEmbeddingsWithMeta):
    def __init__(self, config):
        RoFormerEmbeddings.__init__(self, config)
        netFoundEmbeddingsWithMeta.__init__(self, config)
        self.roformer = config.roformer

    def forward(
            self,
            input_ids=None,
            position_ids=None,
            direction=None,
            iats=None,
            bytes=None,
            pkt_count=None,
            protocol=None,
    ):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.addMetaEmbeddings(embeddings, direction, iats, bytes, pkt_count, protocol)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
