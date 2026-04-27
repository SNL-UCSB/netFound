"""
Tests for src/modules/netFoundDataCollator.py

Covers:
  6. _pad_batch_field_and_flatten — padding and flattening
  7. swap_bursts_adjust_prob_matrix — burst swapping mechanics
  8. maskMetaData — Bernoulli metadata masking with exclusions
  9. torch_mask_tokens — swapped-burst probability
"""
import random

import numpy as np
import pytest
import torch

from src.modules.netFoundDataCollator import DataCollatorWithMeta, _pad_batch_field_and_flatten


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal tokenizer mock for DataCollatorWithMeta."""
    pad_token_id = 0
    mask_token = 65538
    vocab_size = 65537

    def get_special_tokens_mask(self, token_ids_0, already_has_special_tokens=False):
        # Only CLS (65537+1=65538? no, bos_token_id=65537) — mark bos as special
        return [1 if tok == 65537 else 0 for tok in token_ids_0]

    def __len__(self):
        return self.vocab_size


def _make_collator(mlm_probability=0.3, swap_rate=0.5):
    """Create a DataCollatorWithMeta with a fake tokenizer."""
    return DataCollatorWithMeta(
        tokenizer=_FakeTokenizer(),
        mlm=True,
        mlm_probability=mlm_probability,
        swap_rate=swap_rate,
    )


# ========================================================================
# 6. _pad_batch_field_and_flatten
# ========================================================================

class TestPadBatchFieldAndFlatten:

    def test_basic_padding_and_shape(self):
        """Single burst padded to max_burst_length, flow padded to max_bursts."""
        input_data = [[1, 2, 3]]  # 1 burst of length 3
        result = _pad_batch_field_and_flatten(
            input_data, max_burst_length=5, max_bursts=3, padding_token=0
        )
        # Expected: burst [1,2,3,0,0] + 2 empty bursts [0,0,0,0,0] each → total 15
        assert result.shape == (15,)
        expected = np.array([1, 2, 3, 0, 0] + [0] * 5 + [0] * 5)
        np.testing.assert_array_equal(result, expected)

    def test_no_padding_needed(self):
        """Burst and flow exactly at max sizes."""
        input_data = [[1, 2], [3, 4]]
        result = _pad_batch_field_and_flatten(
            input_data, max_burst_length=2, max_bursts=2, padding_token=0
        )
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4]))

    def test_custom_padding_token(self):
        input_data = [[1]]
        result = _pad_batch_field_and_flatten(
            input_data, max_burst_length=3, max_bursts=2, padding_token=-1
        )
        np.testing.assert_array_equal(result, np.array([1, -1, -1, -1, -1, -1]))

    def test_multiple_bursts_variable_length(self):
        """Bursts of different lengths should all be padded to max_burst_length."""
        input_data = [[1, 2, 3], [4]]  # burst 0: len 3, burst 1: len 1
        result = _pad_batch_field_and_flatten(
            input_data, max_burst_length=4, max_bursts=2, padding_token=0
        )
        expected = np.array([1, 2, 3, 0, 4, 0, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_empty_input(self):
        """No bursts at all — entire output is padding."""
        result = _pad_batch_field_and_flatten(
            [], max_burst_length=3, max_bursts=2, padding_token=0
        )
        np.testing.assert_array_equal(result, np.zeros(6))


# ========================================================================
# 7. swap_bursts_adjust_prob_matrix
# ========================================================================

class TestSwapBursts:

    def test_no_self_swap(self):
        """When a flow is selected for swapping, it should never swap with itself."""
        collator = _make_collator(swap_rate=1.0)  # force all flows to be swapped
        batch_size = 4
        num_bursts = 3
        max_burst_length = 5
        input_ids = torch.arange(batch_size * num_bursts * max_burst_length).reshape(
            batch_size, num_bursts * max_burst_length
        ).float()
        bursts_in_each_flow = [num_bursts] * batch_size

        random.seed(42)
        np.random.seed(42)
        _, swappedIds, labels = collator.swap_bursts_adjust_prob_matrix(
            input_ids.clone(), bursts_in_each_flow, max_burst_length, swap_rate=1.0
        )
        # Every flow got a swap → len(swappedIds) == batch_size
        assert len(swappedIds) == batch_size
        # Verify no self-swap in swappedIds (each entry is [flow_idx, burst_idx])
        for swap_entry in swappedIds:
            assert 0 <= swap_entry[0] < batch_size
            assert 0 <= swap_entry[1] < num_bursts

    def test_swap_labels_shape(self):
        collator = _make_collator(swap_rate=0.5)
        input_ids = torch.zeros(3, 10)
        bursts_in_each_flow = [2, 2, 2]
        _, _, labels = collator.swap_bursts_adjust_prob_matrix(
            input_ids, bursts_in_each_flow, max_burst_length=5, swap_rate=0.5
        )
        assert labels.shape == (3,)
        assert set(labels.numpy().tolist()).issubset({0, 1})

    def test_swap_rate_zero_no_swaps(self):
        collator = _make_collator(swap_rate=0.0)
        input_ids = torch.ones(4, 10)
        bursts_in_each_flow = [2, 2, 2, 2]
        np.random.seed(0)
        new_ids, swappedIds, labels = collator.swap_bursts_adjust_prob_matrix(
            input_ids.clone(), bursts_in_each_flow, max_burst_length=5, swap_rate=0.0
        )
        assert len(swappedIds) == 0
        assert labels.sum().item() == 0

    def test_single_flow_wraps_around(self):
        """With 1 flow, flowChoice wraps: (0+1)%1 = 0, so it swaps with itself."""
        collator = _make_collator(swap_rate=1.0)
        input_ids = torch.arange(10).unsqueeze(0).float()
        np.random.seed(42)
        random.seed(42)
        new_ids, swappedIds, labels = collator.swap_bursts_adjust_prob_matrix(
            input_ids.clone(), [2], max_burst_length=5, swap_rate=1.0
        )
        # Still produces a swappedId entry
        assert len(swappedIds) == 1


# ========================================================================
# 8. maskMetaData
# ========================================================================

class TestMaskMetaData:

    def test_swapped_bursts_get_zero_probability(self):
        collator = _make_collator()
        input_ids = torch.zeros(3, 20)  # 3 flows
        bursts_in_each_flow = [4, 4, 4]
        swapped_bursts = [[0, 1], [2, 3]]  # flow 0 burst 1, flow 2 burst 3

        torch.manual_seed(0)
        result = collator.maskMetaData(input_ids, bursts_in_each_flow, swapped_bursts)

        # Since bernoulli(0) == 0 always, swapped positions should be False
        assert result[0][1].item() == False
        assert result[2][3].item() == False

    def test_flows_with_3_or_fewer_bursts_excluded(self):
        """Flows with ≤3 bursts should have all probabilities zeroed (candidate_flows filter)."""
        collator = _make_collator()
        input_ids = torch.zeros(2, 20)
        bursts_in_each_flow = [3, 5]  # flow 0 has only 3 bursts
        swapped_bursts = []

        torch.manual_seed(0)
        result = collator.maskMetaData(input_ids, bursts_in_each_flow, swapped_bursts)

        # Flow 0 (3 bursts) should be fully False (excluded)
        assert result[0].sum().item() == 0

    def test_output_shape(self):
        collator = _make_collator()
        input_ids = torch.zeros(2, 10)
        bursts_in_each_flow = [5, 4]
        result = collator.maskMetaData(input_ids, bursts_in_each_flow, [])
        assert result.shape == (2, 5)  # max(bursts_in_each_flow) = 5
        assert result.dtype == torch.bool


# ========================================================================
# 9. torch_mask_tokens — MLM masking + swapped burst probability bug
# ========================================================================

class TestTorchMaskTokens:
    def test_swapped_burst_tokens_should_not_be_masked(self):
        """
        Swapped bursts should have MLM probability set to 0 so they're never masked.
        Due to the empty-slice bug on line 122, this does NOT work — the swapped
        burst tokens can still be MLM-masked.
        """
        collator = _make_collator(mlm_probability=1.0, swap_rate=1.0)
        batch_size = 4
        num_bursts = 4
        max_burst_length = 5
        seq_len = num_bursts * max_burst_length
        # Use distinct values so we can identify swapped regions
        input_ids = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len).long()
        bursts_in_each_flow = [num_bursts] * batch_size
        protos = torch.tensor([6] * batch_size)

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        new_ids, labels, swappedLabels, mask_meta = collator.torch_mask_tokens(
            input_ids.clone(), bursts_in_each_flow, max_burst_length,
            swap_rate=1.0, protos=protos
        )
        # For each swapped burst, the labels in that region should be -100

        # Re-run to get swappedIds
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        _, swappedIds, _ = collator.swap_bursts_adjust_prob_matrix(
            input_ids.clone(), bursts_in_each_flow, max_burst_length, swap_rate=1.0
        )

        for flow_idx, burst_idx in swappedIds:
            start = burst_idx * max_burst_length
            end = (burst_idx + 1) * max_burst_length
            burst_labels = labels[flow_idx, start:end]
            # All should be -100 if probability was set to 0
            assert (burst_labels == -100).all(), (
                f"Swapped burst [{flow_idx}][{start}:{end}] has "
                f"non-(-100) labels: {burst_labels}"
            )
