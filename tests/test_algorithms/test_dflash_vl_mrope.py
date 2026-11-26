"""CPU regressions for the Qwen3-VL DFlash training position contract."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.common.hidden_states_data import build_collator
from specforge.algorithms.dflash.providers import select_multimodal_capture_layout
from specforge.data.vlm_preprocessing import build_qwen3_vl_position_ids
from specforge.data.vlm_preprocessing import _render_conversation_text, ImageDataError
from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    Qwen3InterleavedMultiRotaryEmbedding,
)


def config():
    c = Qwen3Config(
        hidden_size=16, intermediate_size=32, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=1, head_dim=8,
        max_position_embeddings=256, vocab_size=32, num_target_layers=2,
        layer_types=["full_attention"], sliding_window=None,
        rope_theta=5000000,
        rope_scaling={"rope_type": "default", "mrope_interleaved": True,
                      "mrope_section": [2, 1, 1]},
        dflash_config={"mask_token_id": 31, "target_layer_ids": [0]},
        block_size=4,
    )
    c._attn_implementation = "sdpa"
    return c


class PositionsTest(unittest.TestCase):
    def test_capture_transport_preserves_triplets(self):
        from tests.test_runtime.test_server_capture import _mk, _task, _dflash_contract
        from specforge.inference.adapters.server_capture import ServerCaptureSchema

        _, server, _, adapter = _mk("dflash")
        layout = select_multimodal_capture_layout(SimpleNamespace(
            training=SimpleNamespace(dflash_teacher_metrics=True)))
        adapter.schema = ServerCaptureSchema(
            aux_feature=layout.aux_feature, last_hidden_feature=layout.last_hidden_feature,
            passthrough=layout.passthrough)
        task = _task(0, 5)
        positions = [[0, 0, 0], [1, 1, 2], [1, 2, 1], [3, 3, 3], [4, 4, 4]]
        task.payload["position_ids"] = positions
        refs = adapter.produce_refs([task], capture=_dflash_contract())
        self.assertEqual(len(refs), 1)
        tensor = server.expected[refs[0].sample_id]["position_ids"]
        torch.testing.assert_close(tensor, torch.tensor([positions]))

    def test_explicit_image_order_and_later_turn_are_preserved(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: messages)
        messages = [{"role": "user", "content": "first turn"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image"},
                        {"type": "text", "text": "after"}]}]
        actual = _render_conversation_text(tokenizer, messages, with_image=True, source="test")
        self.assertEqual(actual, messages)
        with self.assertRaises(ImageDataError):
            _render_conversation_text(tokenizer, messages, with_image=False, source="test")

    def test_legacy_image_still_attaches_to_first_user(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: messages)
        actual = _render_conversation_text(tokenizer, [{"from": "human", "value": "question"}],
                                           with_image=True, source="test")
        self.assertEqual(actual[0]["content"], [{"type": "image"}, {"type": "text", "text": "question"}])

    def test_multiple_inline_images_are_rejected(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: messages)
        with self.assertRaises(ImageDataError):
            _render_conversation_text(tokenizer, [{"role": "user", "content": [
                {"type": "image"}, {"type": "image"}]}], with_image=True, source="test")

    def test_coordinates_match_transformers_qwen3_vl(self):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        # Call the position helper without constructing a model or loading weights.
        target = object.__new__(Qwen3VLModel)
        nn.Module.__init__(target)
        target.config = SimpleNamespace(
            image_token_id=99, video_token_id=98, vision_start_token_id=97,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
        for h, w in ((4, 6), (6, 4), (2, 2)):
            ids = [10, 97] + [99] * (h * w // 4) + [96, 20, 21, 22]
            actual = build_qwen3_vl_position_ids(ids, image_pad_token_id=99,
                image_grid_thw=[1, h, w], merge_size=2, source="test")
            expected, _ = Qwen3VLModel.get_rope_index(
                target, input_ids=torch.tensor([ids]),
                mm_token_type_ids=(torch.tensor([ids]) == 99).int(),
                image_grid_thw=torch.tensor([[1, h, w]]),
            )
            torch.testing.assert_close(torch.tensor(actual).T[:, None], expected)

    def test_text_and_invalid_grids(self):
        build = lambda ids, grid: build_qwen3_vl_position_ids(
            ids, image_pad_token_id=99, image_grid_thw=grid, merge_size=2, source="test")
        self.assertEqual(build([1, 2, 3], None), [[0]*3, [1]*3, [2]*3])
        for ids, grid in (([99], None), ([99], [1, 4, 4]),
                          ([99]*2, [2, 2, 2]), ([99], [1, 3, 2])):
            with self.assertRaises(ValueError):
                build(ids, grid)

    def test_collation_padding_and_misalignment(self):
        def row(n):
            return dict(input_ids=torch.zeros(1, n, dtype=torch.long),
                loss_mask=torch.ones(1, n), hidden_states=torch.zeros(1, n, 16),
                position_ids=torch.arange(n)[None, :, None].expand(1, n, 3))
        a, b = row(5), row(3)
        result = build_collator()([a, b])
        self.assertEqual(result["position_ids"].shape, (2, 5, 3))
        torch.testing.assert_close(result["position_ids"][1, :3], b["position_ids"][0])
        b["position_ids"] = b["position_ids"][:, :-1]
        with self.assertRaises(ValueError):
            build_collator()([a, b])

    def test_teacher_capture_switch_keeps_positions(self):
        for enabled in (True, False):
            layout = select_multimodal_capture_layout(SimpleNamespace(
                training=SimpleNamespace(dflash_teacher_metrics=enabled)))
            self.assertEqual(layout.last_hidden_feature is not None, enabled)
            self.assertIn(("position_ids", "position_ids", (3,)), layout.passthrough)


class RotaryAndForwardTest(unittest.TestCase):
    def test_qwen3_vl_128_dim_rotary_matches_transformers(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

        options = dict(hidden_size=256, num_attention_heads=2, num_key_value_heads=1,
                       head_dim=128, rope_theta=5000000,
                       rope_scaling={"rope_type": "default", "mrope_interleaved": True,
                                     "mrope_section": [24, 20, 20]})
        reference = Qwen3VLTextRotaryEmbedding(Qwen3VLTextConfig(**options))
        actual = Qwen3InterleavedMultiRotaryEmbedding(Qwen3Config(**options))
        positions = torch.tensor([[[0, 1, 2, 20]], [[0, 4, 8, 20]], [[0, 9, 3, 20]]])
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.zeros(1, 4, 256, dtype=dtype)
            torch.testing.assert_close(actual(x, positions), reference(x, positions), rtol=0, atol=0)

    def test_bfloat16_conversion_preserves_full_precision_frequencies(self):
        c = config()
        ref = Qwen3InterleavedMultiRotaryEmbedding(c)
        actual = Qwen3InterleavedMultiRotaryEmbedding(c).bfloat16()
        positions = torch.tensor([[[0, 65535]], [[1, 10000]], [[3, 65536]]])
        expected = tuple(t.bfloat16() for t in ref(torch.zeros(1, 2, 16), positions))
        got = actual(torch.zeros(1, 2, 16, dtype=torch.bfloat16), positions)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        self.assertEqual(actual.inv_freq.dtype, torch.float32)

    def test_rotary_against_explicit_axis_formula_and_reload(self):
        c = config()
        rotary = Qwen3InterleavedMultiRotaryEmbedding(c)
        pos = torch.tensor([[[0, 2, 7]], [[1, 8, 10]], [[3, 6, 11]]])
        x = torch.zeros(1, 3, 16)
        cos, sin = rotary(x, pos)
        # Independent expected frequency/axis selection for sections [2,1,1].
        freq = 5000000.0 ** (-torch.arange(0, 8, 2).float() / 8)
        axes = torch.tensor([0, 1, 2, 0])
        phase = pos[axes, 0].T[None].float() * freq
        phase = torch.cat([phase, phase], -1)
        torch.testing.assert_close(cos, phase.cos())
        torch.testing.assert_close(sin, phase.sin())
        with tempfile.TemporaryDirectory() as path:
            c.save_pretrained(path)
            loaded = Qwen3Config.from_pretrained(path)
            actual = Qwen3InterleavedMultiRotaryEmbedding(loaded)(x, pos)
            torch.testing.assert_close(actual, (cos, sin))

    def test_full_draft_backward_uses_axes_and_continues_tail(self):
        torch.manual_seed(7)
        draft = DFlashDraftModel(config())
        model = OnlineDFlashModel(draft, nn.Linear(16, 32, bias=False),
            nn.Embedding(32, 16), 31, block_size=4, num_anchors=1,
            attention_backend="sdpa", teacher_metrics=False)
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        hidden = torch.randn(1, 5, 16)
        mask = torch.tensor([[0., 0., 0., 1., 1.]])
        pos = torch.tensor([[[0, 0, 0], [1, 1, 1], [1, 1, 2],
                             [3, 3, 3], [4, 4, 4]]])
        captured = []
        handle = draft.rotary_emb.register_forward_pre_hook(
            lambda module, args: captured.append(args[1].detach().clone()))
        from specforge.runtime.contracts import TrainBatch
        from specforge.training.strategies.base import DFlashTrainStrategy, StepContext

        tensors = build_collator()([dict(input_ids=ids, hidden_states=hidden,
                                         loss_mask=mask, position_ids=pos)])
        output = DFlashTrainStrategy(model).forward_loss(
            TrainBatch(sample_ids=["sample"], strategy="dflash", tensors=tensors),
            StepContext(collect_detailed_metrics=False))
        loss = output.loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(draft.fc.weight.grad).all())
        self.assertGreater(draft.fc.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(captured[0][:, :, :5], pos.permute(2, 0, 1))
        expected_tail = torch.tensor([3, 4, 5, 6])[None, None].expand(3, 1, -1)
        torch.testing.assert_close(captured[0][:, :, 5:], expected_tail)
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            model(ids, hidden, mask, collect_detailed_metrics=False)
        with self.assertRaisesRegex(ValueError, "alternate axis"):
            model(ids, hidden, mask, position_ids=pos.permute(2, 0, 1))
        handle.remove()

    def test_malformed_mrope_fails_instead_of_falling_back(self):
        c = config()
        c.rope_parameters["mrope_section"] = [1, 1, 1]
        with self.assertRaisesRegex(ValueError, "mrope_section"):
            Qwen3InterleavedMultiRotaryEmbedding(c)


if __name__ == "__main__":
    unittest.main()
