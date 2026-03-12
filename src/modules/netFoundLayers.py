import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Tuple

from modules.netFoundAttentions import FlashSelfAttention, RobertaAttention, RoFormerAttention

from transformers.utils import ModelOutput
from transformers.activations import gelu_new
from transformers.models.modernbert.modeling_modernbert import ModernBertMLP
from transformers.models.roformer.modeling_roformer import RoFormerSinusoidalPositionalEmbedding


@dataclass
class BaseModelOutputWithFlowAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    flow_attentions: Optional[Tuple[torch.FloatTensor]] = None


def transform_tokens2bursts(hidden_states, num_bursts, max_burst_length):
    # transform sequence into segments
    seg_hidden_states = torch.reshape(
        hidden_states,
        (hidden_states.size(0), num_bursts, max_burst_length, hidden_states.size(-1)),
    )
    # squash segments into sequence into a single axis (samples * segments, max_segment_length, hidden_size)
    hidden_states_reshape = seg_hidden_states.contiguous().view(
        hidden_states.size(0) * num_bursts, max_burst_length, seg_hidden_states.size(-1)
    )

    return hidden_states_reshape


def transform_masks2bursts(hidden_states, num_bursts, max_burst_length):
    # transform sequence into segments
    seg_hidden_states = torch.reshape(
        hidden_states, (hidden_states.size(0), 1, 1, num_bursts, max_burst_length)
    )
    # squash segments into sequence into a single axis (samples * segments, 1, 1, max_segment_length)
    hidden_states_reshape = seg_hidden_states.contiguous().view(
        hidden_states.size(0) * num_bursts, 1, 1, seg_hidden_states.size(-1)
    )

    return hidden_states_reshape


def transform_bursts2tokens(seg_hidden_states, num_bursts, max_burst_length):
    # transform squashed sequence into segments
    hidden_states = seg_hidden_states.contiguous().view(
        seg_hidden_states.size(0) // num_bursts,
        num_bursts,
        max_burst_length,
        seg_hidden_states.size(-1),
    )
    # transform segments into sequence
    hidden_states = hidden_states.contiguous().view(
        hidden_states.size(0), num_bursts * max_burst_length, hidden_states.size(-1)
    )
    return hidden_states


class TransformerLayer(nn.Module):
    """
    Defines a typical transformer layer with attention and feed-forward network, see ModernBERT for design
    """
    def __init__(self, config):
        super().__init__()
        self.roformer = config.roformer
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        if config.use_flash_attn:
            self.attention = FlashSelfAttention(config, use_rotary=self.roformer)
        else:
            self.attention = RoFormerAttention(config) if self.roformer else RobertaAttention(config)
        # HF RoFormerAttention/RobertaAttention already include residual + LayerNorm
        # inside their SelfOutput sublayer; FlashSelfAttention does not.
        self._attn_has_internal_residual = not config.use_flash_attn
        self.is_decoder = config.is_decoder
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            seqNo=None
    ):
        kwargs = {}
        if self.roformer:
            kwargs = {"sinusoidal_pos": seqNo}
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self._attn_has_internal_residual:
            # RoFormerAttention / RobertaAttention output already contains
            # residual connection + LayerNorm (Post-LN), so do NOT add again.
            hidden_states = attention_outputs[0]
        else:
            hidden_states = hidden_states + attention_outputs[0]
        mlp_output = self.mlp(self.mlp_norm(hidden_states))
        hidden_states = hidden_states + mlp_output
        return (hidden_states,) + attention_outputs[1:]


class netFoundLayer(nn.Module):
    """
    Defines a netFound Transformer layer which consists of two TransformerLayers - for bursts and flows
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.burst_encoder = TransformerLayer(config)
        self.flow_encoder = TransformerLayer(config)
        self.position_embeddings = nn.Embedding(
            config.max_bursts + 1, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.roformer = config.roformer

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            num_bursts=None,
            output_attentions=False,
            burstSeqNo=None,
            flowSeqNo=None,
            batch_max_burst_length=None,
    ):
        # transform sequences to bursts
        burst_inputs = transform_tokens2bursts(
            hidden_states, num_bursts=num_bursts, max_burst_length=batch_max_burst_length
        )
        burst_masks = transform_masks2bursts(
            attention_mask,
            num_bursts=num_bursts,
            max_burst_length=batch_max_burst_length,
        )
        burst_outputs = self.burst_encoder(
            burst_inputs, burst_masks, output_attentions=output_attentions, seqNo=burstSeqNo
        )

        # flatten bursts back to tokens
        outputs = transform_bursts2tokens(
            burst_outputs[0],
            num_bursts=num_bursts,
            max_burst_length=batch_max_burst_length,
        )

        burst_global_tokens = outputs[:, ::batch_max_burst_length].clone()
        burst_attention_mask = attention_mask[:, :, :, ::batch_max_burst_length].clone()

        burst_positions = torch.arange(1, num_bursts + 1).repeat(outputs.size(0), 1) \
                              .to(outputs.device) * (burst_attention_mask.reshape(-1, num_bursts) >= -1).int().to(
            outputs.device)
        outputs[:, ::batch_max_burst_length] += self.position_embeddings(burst_positions)

        flow_outputs = self.flow_encoder(
            burst_global_tokens,
            burst_attention_mask,
            output_attentions=output_attentions,
            seqNo=flowSeqNo
        )

        # replace burst representative tokens
        outputs[:, ::batch_max_burst_length] = flow_outputs[0]

        return outputs, burst_outputs, flow_outputs


class netFoundLayerFlat(nn.Module):
    """
    Ablation study: what if we only use burst encoder without flow encoder
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.burst_encoder = TransformerLayer(config)
        self.position_embeddings = nn.Embedding(
            config.max_bursts + 1, config.hidden_size, padding_idx=config.pad_token_id
        )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            num_bursts=None,
            output_attentions=False,
            burstSeqNo=None,
            flowSeqNo=None,
            batch_max_burst_length=None,
    ):
        burst_inputs = transform_tokens2bursts(
            hidden_states, num_bursts=num_bursts, max_burst_length=batch_max_burst_length
        )
        burst_masks = transform_masks2bursts(
            attention_mask,
            num_bursts=num_bursts,
            max_burst_length=batch_max_burst_length,
        )
        burst_outputs = self.burst_encoder(
            burst_inputs, burst_masks, output_attentions=output_attentions, seqNo=burstSeqNo
        )
        outputs = transform_bursts2tokens(
            burst_outputs[0],
            num_bursts=num_bursts,
            max_burst_length=batch_max_burst_length,
        )
        return outputs, burst_outputs


class netFoundEncoder(nn.Module):
    """
    Encoder module consisting of multiple netFound layers
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.roformer = config.roformer
        layer_class = netFoundLayerFlat if config.flat else netFoundLayer
        self.layer = nn.ModuleList(
            [layer_class(config) for _ in
             range(config.num_hidden_layers)]
        )
        self.burst_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size // config.num_attention_heads
        )
        self.flow_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_bursts + 1, config.hidden_size // config.num_attention_heads
        )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            num_bursts=None,
            output_attentions=False,
            output_hidden_states=False,
            batch_max_burst_length=None,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_burst_attentions = () if output_attentions else None

        burst_seqs = transform_tokens2bursts(
            hidden_states, num_bursts=num_bursts, max_burst_length=batch_max_burst_length
        )
        past_key_values_length = 0
        burstSeqNo = self.burst_positions(burst_seqs.shape[:-1], past_key_values_length)[None, None, :, :]
        flow_seqs = transform_bursts2tokens(
            burst_seqs,
            num_bursts=num_bursts,
            max_burst_length=batch_max_burst_length,
        )[:, ::batch_max_burst_length]
        flowSeqNo = self.flow_positions(flow_seqs.shape[:-1], past_key_values_length)[None, None, :, :]

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, num_bursts, output_attentions, burstSeqNo, flowSeqNo, batch_max_burst_length
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_burst_attentions = all_burst_attentions + (layer_outputs[2],)
            else:
                all_self_attentions = None
                all_burst_attentions = None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_burst_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithFlowAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            flow_attentions=all_burst_attentions,
        )

    def _tie_weights(self):
        # Find the source: prefer a layer whose weight has real data (not meta),
        # because safetensors deduplication may save only one copy (e.g. the last layer).
        source = None
        for module in self.layer:
            if hasattr(module, "position_embeddings"):
                assert hasattr(module.position_embeddings, "weight")
                if source is None or source.weight.is_meta:
                    source = module.position_embeddings
        # Tie all layers' position_embeddings to the source
        if source is not None:
            for module in self.layer:
                if hasattr(module, "position_embeddings"):
                    module.position_embeddings.weight = source.weight
        return


class LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu_new(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        self.decoder.bias = self.bias
