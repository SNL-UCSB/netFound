import torch
import torch.nn as nn
import torch.nn.functional as F

# reimports to keep things in the same place
from transformers.models.roformer.modeling_roformer import RoFormerAttention  # noqa: F401
from transformers.models.roberta.modeling_roberta import RobertaAttention  # noqa: F401

class SDPAMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, causal=False, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.dropout_p = dropout
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.Wqkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.causal,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class FlashSelfAttention(nn.Module):
    """Drop-in replacement — same interface as the flash_attn-based version,
    portable across CUDA and ROCm via SDPA."""
    def __init__(self, config, use_rotary=False):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("This attention module requires a CUDA/ROCm-capable device")
        self.attn = SDPAMHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            causal=config.is_decoder,
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        output = self.attn(hidden_states, attn_mask=attention_mask)
        return (output, None)
