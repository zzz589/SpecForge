"""Compare packed single-image training blocks with independent cached calls."""

import unittest
from unittest.mock import patch

import torch
from transformers import DynamicCache

from specforge.data.vlm_preprocessing import build_qwen3_vl_position_ids
from tests.test_algorithms.test_dspark_vl_mrope import build_model


class TestDSparkMropeBlockParity(unittest.TestCase):
    def test_packed_blocks_match_independent_and_cached_prefixes(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.no_grad():
                model = build_model(dtype=dtype).eval()
                ids = torch.tensor([[1, 2, 13, 13, 13, 13, 3, 4, 5, 6, 8, 9]])
                positions = torch.tensor([build_qwen3_vl_position_ids(
                    ids[0].tolist(), image_pad_token_id=13, image_grid_thw=[1, 4, 4],
                    merge_size=2, source="test")])
                torch.manual_seed(38)
                features = torch.randn(1, ids.shape[1], 32).to(dtype)
                anchors = torch.tensor([[7, 9, 10]])
                keep = torch.ones_like(anchors, dtype=torch.bool)
                mask = torch.zeros_like(ids); mask[:, 7:] = 1
                with patch.object(model, "_sample_anchor_positions", return_value=(anchors, keep)):
                    packed = model._forward_draft_blocks(ids, features, mask, position_ids=positions)[2]
                cached = DynamicCache()
                cursor = 0
                for block, anchor in enumerate(anchors[0].tolist()):
                    noise = model._create_noise_embed(ids, torch.tensor([[anchor]]), torch.tensor([[True]]))
                    continuation = positions[:, anchor:anchor+1] + torch.arange(4).view(1, 4, 1)
                    complete_positions = torch.cat((positions[:, :anchor], continuation), dim=1).permute(2, 0, 1)
                    isolated = model.draft_model(target_hidden=features[:, :anchor],
                        noise_embedding=noise, position_ids=complete_positions)
                    incremental_positions = torch.cat((positions[:, cursor:anchor], continuation), dim=1).permute(2, 0, 1)
                    incremental = model.draft_model(target_hidden=features[:, cursor:anchor],
                        noise_embedding=noise, position_ids=incremental_positions,
                        past_key_values=cached, use_cache=True)
                    cached.crop(anchor)
                    self.assertEqual(cached.get_seq_length(), anchor)
                    cursor = anchor
                    expected = packed[:, block*4:(block+1)*4]
                    tolerance = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=0.02, atol=0.01)
                    torch.testing.assert_close(isolated, expected, **tolerance)
                    torch.testing.assert_close(incremental, expected, **tolerance)
                    # Compression after image expansion must be retained when
                    # new context is appended; logical cache indices stay uncompressed.
                    self.assertLess(positions[0, anchor, 0].item(), anchor)


if __name__ == "__main__":
    unittest.main()
