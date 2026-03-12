import torch

from modules.netFoundModels import netFoundLanguageModelling

class TestMaskMeta:
    @staticmethod
    def _make_model() -> netFoundLanguageModelling:
        model = netFoundLanguageModelling.__new__(netFoundLanguageModelling)
        return model

    def test_all_false_mask_no_zeroing(self):
        """All-False mask (no bursts masked) should leave data untouched."""
        model = self._make_model()
        bursts_to_mask = torch.zeros(2, 3, dtype=torch.bool)
        meta = torch.tensor([
            [1., 2., 3., 4., 5., 6.],
            [7., 8., 9., 10., 11., 12.],
        ])
        result = model.maskMeta(bursts_to_mask, meta, max_burst_length=2)
        assert torch.equal(result, meta)

    def test_all_true_mask_zeros_everything(self):
        model = self._make_model()
        bursts_to_mask = torch.ones(1, 2, dtype=torch.bool)
        meta = torch.tensor([[5., 10., 15., 20.]])
        result = model.maskMeta(bursts_to_mask, meta, max_burst_length=2)
        assert torch.equal(result, torch.zeros_like(meta))

    def test_single_burst_flow(self):
        model = self._make_model()
        bursts_to_mask = torch.tensor([[1]], dtype=torch.bool)
        meta = torch.tensor([[3., 4., 5.]])
        result = model.maskMeta(bursts_to_mask, meta, max_burst_length=3)
        assert torch.equal(result, torch.zeros_like(meta))

    def test_mixed_masking_multiple_metadata_fields(self):
        """Test with realistic-shaped metadata where burst-level mask expands to token-level."""
        model = self._make_model()
        # 2 flows, 3 bursts each, max_burst_length=4
        bursts_to_mask = torch.tensor([
            [True, False, True],
            [False, True, False],
        ])
        # Each flow has 3*4=12 tokens
        meta = torch.ones(2, 12)
        result = model.maskMeta(bursts_to_mask, meta, max_burst_length=4)

        # Flow 0: burst 0 masked (tokens 0-3 → 0), burst 1 not (tokens 4-7 → 1), burst 2 masked (tokens 8-11 → 0)
        assert torch.equal(result[0, :4], torch.zeros(4))
        assert torch.equal(result[0, 4:8], torch.ones(4))
        assert torch.equal(result[0, 8:12], torch.zeros(4))

        # Flow 1: burst 0 not (→1), burst 1 masked (→0), burst 2 not (→1)
        assert torch.equal(result[1, :4], torch.ones(4))
        assert torch.equal(result[1, 4:8], torch.zeros(4))
        assert torch.equal(result[1, 8:12], torch.ones(4))


    def test_maskmeta_zeroes_masked_bursts(self):
        model = self._make_model()
        bursts_to_mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)
        meta = torch.tensor([
            [1., 2., 3., 4., 5., 6.],
            [7., 8., 9., 10., 11., 12.],
        ])

        masked = model.maskMeta(bursts_to_mask, meta, 3)

        expected = torch.tensor([
            [0., 0., 0., 4., 5., 6.],
            [7., 8., 9., 0., 0., 0.],
        ])
        assert torch.equal(masked, expected)


    def test_maskmeta_leaves_unmasked_intact(self):
        model = self._make_model()
        bursts_to_mask = torch.zeros((1, 3), dtype=torch.bool)
        meta = torch.tensor([[1., 2., 3., 4., 5., 6.]])

        masked = model.maskMeta(bursts_to_mask, meta, 2)

        assert torch.equal(masked, meta)

