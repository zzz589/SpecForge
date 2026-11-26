"""Regressions for disabled confidence objectives and DSpark proposal slots."""

import copy
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.training.strategies.base import DSparkTrainStrategy, StepContext
from tests.test_algorithms.test_dspark_vl_mrope import as_batch, build_model, feature_row


def model_with_heads(confidence_alpha=1.0, markov_type="vanilla"):
    parts = build_model()
    config = copy.deepcopy(parts.draft_model.config)
    config.dflash_config["markov_head_type"] = markov_type
    return OnlineDSparkModel(
        DSparkDraftModel(config), parts.lm_head, parts.embed_tokens, 31,
        block_size=4, num_anchors=64, attention_backend="sdpa",
        dspark_confidence_head_alpha=confidence_alpha, objective_chunk_blocks=1,
    )


class TestDSparkHeadReview(unittest.TestCase):
    def test_disabled_confidence_head_is_frozen_but_markov_still_trains(self):
        model = model_with_heads(confidence_alpha=0)
        self.assertTrue(all(not p.requires_grad for p in model.draft_model.confidence_head.parameters()))
        with patch.object(model.draft_model, "predict_confidence", side_effect=AssertionError("disabled")):
            out = DSparkTrainStrategy(model).forward_loss(
                as_batch(feature_row([0, 0, 1, 1, 1, 1])),
                StepContext(collect_detailed_metrics=False))
            out.loss.backward()
        for name, parameter in model.draft_model.named_parameters():
            if name.startswith("confidence_head."):
                self.assertIsNone(parameter.grad)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.draft_model.markov_head.markov_w2.weight.grad.abs().sum().item(), 0)

    def test_enabled_confidence_head_still_receives_gradients(self):
        model = model_with_heads()
        out = DSparkTrainStrategy(model).forward_loss(
            as_batch(feature_row([0, 0, 1, 1, 1, 1])), StepContext(collect_detailed_metrics=False))
        out.loss.backward()
        for parameter in model.draft_model.confidence_head.parameters():
            self.assertTrue(parameter.requires_grad)
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_disabled_confidence_ddp_runs_multiple_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            created = not dist.is_initialized()
            if created:
                dist.init_process_group("gloo", init_method="file://" + str(Path(temp) / "group"),
                    rank=0, world_size=1, timeout=timedelta(seconds=30))
            try:
                model = model_with_heads(confidence_alpha=0)
                wrapped = DDP(model, broadcast_buffers=False, gradient_as_bucket_view=True)
                optimizer = torch.optim.SGD((p for p in wrapped.parameters() if p.requires_grad), lr=0.01)
                frozen = {name: p.detach().clone() for name, p in model.draft_model.confidence_head.named_parameters()}
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    out = DSparkTrainStrategy(wrapped).forward_loss(
                        as_batch(feature_row([0, 0, 1, 1, 1, 1])), StepContext(collect_detailed_metrics=False))
                    out.loss.backward()
                    optimizer.step()
                    self.assertTrue(torch.isfinite(out.loss))
                for name, p in model.draft_model.confidence_head.named_parameters():
                    torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
            finally:
                if created:
                    dist.destroy_process_group()

    def test_generation_slots_match_next_token_labels_for_all_markov_heads(self):
        for kind in (None, "vanilla", "gated", "rnn"):
            with self.subTest(kind=kind):
                model = model_with_heads(markov_type=kind or "vanilla")
                draft = model.draft_model
                if kind is None:
                    draft.markov_head = None
                else:
                    with torch.no_grad():
                        draft.markov_head.markov_w2.weight.zero_()
                # Two leading sentinel states also test selecting the last block.
                hidden = torch.zeros(1, 6, 16)
                hidden[:, :2, 12] = 100
                for slot in range(4):
                    hidden[0, slot + 2, slot + 1] = 10
                head = nn.Linear(16, 32, bias=False)
                with torch.no_grad():
                    head.weight.zero_(); head.weight[:16].copy_(torch.eye(16))
                labels, _, _ = model._build_dspark_labels_and_mask(
                    torch.tensor([[9, 1, 2, 3, 4]]), torch.ones(1, 5),
                    torch.tensor([[0]]), torch.tensor([[True]]))
                result = draft._sample_draft_tokens(SimpleNamespace(lm_head=head), hidden,
                                                    torch.tensor([[9, 31, 31, 31]]))
                torch.testing.assert_close(result, labels[:, 0, :3])

    def test_zero_confidence_checkpoint_resume_matches_continuous_training(self):
        from tests.test_algorithms.test_dspark_vl_mrope import DSparkVLMRegressionTest

        # Exercise the real Trainer/BF16Optimizer save/resume seam with the
        # reduced trainable-parameter set, including its provenance checks.
        with patch("tests.test_algorithms.test_dspark_vl_mrope.build_model",
                   side_effect=lambda: model_with_heads(confidence_alpha=0)):
            case = DSparkVLMRegressionTest("test_real_trainer_resume_and_required_dspark_contract")
            case.test_real_trainer_resume_and_required_dspark_contract()

    def test_generation_uses_predecessor_chain_on_correct_backbone_slots(self):
        for kind in ("vanilla", "gated", "rnn"):
            with self.subTest(kind=kind):
                model = model_with_heads(markov_type=kind)
                draft = model.draft_model
                hidden = torch.randn(2, 4, 16)
                blocks = torch.tensor([[3, 31, 31, 31], [5, 31, 31, 31]])
                generated = draft._sample_draft_tokens(SimpleNamespace(lm_head=model.lm_head), hidden, blocks)
                previous = torch.cat((blocks[:, :1], generated[:, :-1]), dim=1)
                trained_logits = draft.apply_logits_head(model.lm_head(hidden[:, :3]),
                    prev_token_ids=previous, hidden_states=hidden[:, :3])
                torch.testing.assert_close(generated, trained_logits.argmax(dim=-1))


if __name__ == "__main__":
    unittest.main()
