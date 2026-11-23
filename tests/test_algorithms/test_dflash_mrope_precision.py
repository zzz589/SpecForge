import unittest
import torch
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import Qwen3InterleavedMultiRotaryEmbedding


def check_parent_cast(dtype):
    config = Qwen3Config(hidden_size=128, num_attention_heads=1, head_dim=128)
    config.rope_parameters.update(rope_theta=5000000.0, mrope_section=[24, 20, 20])
    rotary = Qwen3InterleavedMultiRotaryEmbedding(config)
    parent = torch.nn.ModuleList([rotary, torch.nn.Linear(4, 4)])
    expected = rotary.inv_freq.clone()
    assert not torch.equal(expected, expected.to(dtype).float())
    positions = torch.tensor([[[0, 140, 1150]], [[0, 73, 1150]], [[0, 18, 1150]]])
    reference = rotary(torch.zeros(1), positions)
    parent.to(dtype=dtype)
    assert parent[1].weight.dtype == dtype
    for buf in (rotary.inv_freq, rotary.original_inv_freq):
        assert buf.dtype == torch.float32
        assert torch.equal(buf, expected)
    for result, original in zip(rotary(torch.zeros(1, dtype=dtype), positions), reference):
        assert torch.equal(result, original.to(dtype))
    assert not rotary.state_dict()


class RotaryPrecisionTest(unittest.TestCase):
    def test_bfloat16(self):
        check_parent_cast(torch.bfloat16)

    def test_float16(self):
        check_parent_cast(torch.float16)
