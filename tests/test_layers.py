"""
Tests for src/modules/netFoundLayers.py

Covers:
  10. transform_tokens2bursts / transform_bursts2tokens — reshape roundtrip
"""
import torch
import pytest

from src.modules.netFoundLayers import (
    transform_tokens2bursts,
    transform_masks2bursts,
    transform_bursts2tokens,
    netFoundEncoder,
    LMHead,
)
from src.modules.netFoundConfigBase import netFoundNoPayloadConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Small config for CPU-friendly tests."""
    defaults = dict(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_bursts=4,
        max_burst_length=5,
        max_position_embeddings=5,
        model_max_length=20,
        roformer=True,
        use_flash_attn=False,
        flat=False,
        vocab_size=100,
        hidden_dropout_prob=0.0,
        mlp_dropout=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-12,
        norm_eps=1e-5,
        norm_bias=False,
        mlp_bias=False,
        type_vocab_size=2,
        initializer_range=0.02,
        pad_token_id=0,
        position_embedding_type="absolute",
        no_meta=False,
        no_mlm=False,
        no_swapped_bursts=False,
        no_metadata_loss=False,
        no_direction_loss=False,
        pretraining=True,
        p=0.0,
        limit_bursts=False,
        strip_payload=True,
        rotary_value=False,
        subflow_len=-1,
        compile=False,
    )
    defaults.update(overrides)
    return netFoundNoPayloadConfig(**defaults)


# ========================================================================
# 10. Reshape roundtrip
# ========================================================================

class TestReshapeRoundtrip:

    @pytest.mark.parametrize("batch,num_bursts,max_burst_length,hidden", [
        (2, 3, 5, 16),
        (1, 1, 4, 8),
        (4, 6, 10, 32),
    ])
    def test_tokens2bursts_then_back(self, batch, num_bursts, max_burst_length, hidden):
        """transform_bursts2tokens(transform_tokens2bursts(x)) should be identity."""
        x = torch.randn(batch, num_bursts * max_burst_length, hidden)
        burst_form = transform_tokens2bursts(x, num_bursts, max_burst_length)
        assert burst_form.shape == (batch * num_bursts, max_burst_length, hidden)

        recovered = transform_bursts2tokens(burst_form, num_bursts, max_burst_length)
        assert recovered.shape == x.shape
        assert torch.allclose(recovered, x)

    def test_tokens2bursts_shape(self):
        batch, num_bursts, max_burst_length, hidden = 2, 4, 5, 8
        x = torch.randn(batch, num_bursts * max_burst_length, hidden)
        result = transform_tokens2bursts(x, num_bursts, max_burst_length)
        assert result.shape == (batch * num_bursts, max_burst_length, hidden)

    def test_masks2bursts_shape(self):
        batch, num_bursts, max_burst_length = 2, 4, 5
        # Attention masks in HF are (batch, 1, 1, seq_len) for extended masks
        # but here input is (batch, num_bursts * max_burst_length)
        mask = torch.ones(batch, num_bursts * max_burst_length)
        result = transform_masks2bursts(mask, num_bursts, max_burst_length)
        assert result.shape == (batch * num_bursts, 1, 1, max_burst_length)

    def test_values_preserved_through_roundtrip(self):
        """Verify specific values survive the reshape."""
        batch, num_bursts, max_burst_length, hidden = 1, 2, 3, 4
        x = torch.arange(batch * num_bursts * max_burst_length * hidden, dtype=torch.float).reshape(
            batch, num_bursts * max_burst_length, hidden
        )
        burst_form = transform_tokens2bursts(x, num_bursts, max_burst_length)
        # First burst: tokens 0..2 in a batch of 1
        assert torch.equal(burst_form[0], x[0, :max_burst_length])
        # Second burst: tokens 3..5
        assert torch.equal(burst_form[1], x[0, max_burst_length:2 * max_burst_length])

        recovered = transform_bursts2tokens(burst_form, num_bursts, max_burst_length)
        assert torch.equal(recovered, x)
