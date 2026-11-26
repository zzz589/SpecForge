"""DSpark position transport, generation boundaries, pooled objective and export."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.algorithms.common.hidden_states_data import build_dspark_collator
from specforge.algorithms.dspark.providers import algorithm_providers, resume_contract
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import DSparkTrainStrategy, StepContext


def build_model(mrope=True, dtype=torch.float32):
    torch.manual_seed(41)
    config = Qwen3Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=32, draft_vocab_size=32, max_position_embeddings=512,
        num_target_layers=3, layer_types=["full_attention"] * 2,
        sliding_window=None, eos_token_id=7, rope_theta=1000000,
        block_size=4, architectures=["DSparkDraftModel"],
        dflash_config={"target_layer_ids": [0, 1], "mask_token_id": 31,
                       "projector_type": "dspark", "markov_rank": 4,
                       "markov_head_type": "vanilla", "enable_confidence_head": True,
                       "confidence_head_with_markov": True},
        rope_scaling=({"rope_type": "default", "mrope_interleaved": True,
                       "mrope_section": [2, 1, 1]} if mrope else None),
    )
    config._attn_implementation = "sdpa"
    return OnlineDSparkModel(
        DSparkDraftModel(config), nn.Linear(16, 32, bias=False).requires_grad_(False),
        nn.Embedding(32, 16).requires_grad_(False), 31,
        block_size=4, num_anchors=64, attention_backend="sdpa",
        objective_chunk_blocks=1,
    ).to(dtype=dtype)


def feature_row(mask, dtype=torch.float32):
    n = len(mask)
    torch.manual_seed(22 + n)
    positions = torch.arange(n)[None, :, None].expand(1, n, 3).clone()
    positions[:, 2, :] = torch.tensor([1, 2, 1])
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 8, 9, 10, 11][:n]]),
        "loss_mask": torch.tensor([mask]),
        "hidden_states": torch.randn(1, n, 32).to(dtype),
        "target_last_hidden_states": torch.randn(1, n, 16).to(dtype),
        "position_ids": positions,
    }


def as_batch(row):
    return TrainBatch(sample_ids=["sample"], strategy="dspark", tensors=row)


class DSparkVLMRegressionTest(unittest.TestCase):
    def test_capture_transports_positions_and_teacher_states(self):
        from tests.test_runtime.test_server_capture import _mk, _task, _dspark_contract
        from specforge.inference.adapters.server_capture import ServerCaptureSchema
        _, server, _, adapter = _mk("dspark")
        layout = algorithm_providers().server_streaming_for("multimodal").layout
        adapter.schema = ServerCaptureSchema(aux_feature=layout.aux_feature,
            last_hidden_feature=layout.last_hidden_feature, passthrough=layout.passthrough)
        task = _task(0, 5)
        task.payload["position_ids"] = [[0, 0, 0], [1, 1, 2], [1, 2, 1], [3, 3, 3], [4, 4, 4]]
        refs = adapter.produce_refs([task], capture=_dspark_contract())
        tensors = server.expected[refs[0].sample_id]
        torch.testing.assert_close(tensors["position_ids"], torch.tensor([task.payload["position_ids"]]))
        self.assertIn("target_last_hidden_states", tensors)

    def test_collator_pads_triplets_and_rejects_misalignment(self):
        a, b = feature_row([0, 0, 1, 1, 1]), feature_row([0, 1, 1])
        collate = build_dspark_collator()
        result = collate([a, b])
        self.assertEqual(tuple(result["position_ids"].shape), (2, 5, 3))
        torch.testing.assert_close(result["position_ids"][1, :3], b["position_ids"][0])
        b["position_ids"] = b["position_ids"][:, :-1]
        with self.assertRaises(ValueError):
            collate([a, b])

    @patch.dict("os.environ", {"SPECFORGE_DSPARK_FUSED_LOSS": "0"})
    def test_mrope_backward_and_next_token_eos_boundary(self):
        self.assertEqual(os.environ.get("SPECFORGE_DSPARK_FUSED_LOSS"), "0")
        # CPU exercises the eager objective; Triton loss requires a CUDA device.
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = build_model(dtype=dtype)
                row = feature_row([0, 0, 0, 1, 1, 1], dtype)
                row["input_ids"][0, 4] = 7  # Answer token, EOS, template suffix.
                positions = []
                hook = model.draft_model.rotary_emb.register_forward_pre_hook(
                    lambda _module, args: positions.append(args[1].detach().clone()))
                out = DSparkTrainStrategy(model).forward_loss(as_batch(row), StepContext(collect_detailed_metrics=False))
                hook.remove()
                self.assertEqual(out.loss_terms[1].item(), 1)
                out.loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.draft_model.parameters() if p.requires_grad))
                self.assertIsNone(model.lm_head.weight.grad)
                torch.testing.assert_close(positions[0][:, :, :6], row["position_ids"].permute(2, 0, 1))
                torch.testing.assert_close(positions[0][:, :, 6:], torch.arange(3, 7)[None, None].expand(3, 1, 4))
                anchors, keep = model._sample_anchor_positions(6, model._generation_loss_mask(row["input_ids"], row["loss_mask"]), torch.device("cpu"))
                labels, valid, indices = model._build_dspark_labels_and_mask(row["input_ids"],
                    model._generation_loss_mask(row["input_ids"], row["loss_mask"]), anchors, keep)
                self.assertEqual(labels[valid].tolist(), [7])
                teacher = torch.arange(6)[None, :, None].expand(1, 6, 16).float()
                self.assertEqual(model._aligned_target_hidden(teacher, indices)[0, 0, 0, 0].item(), 3)

    def test_missing_positions_and_plain_multimodal_config_fail(self):
        from specforge.algorithms.dspark.providers import build_draft
        model = build_model()
        row = feature_row([0, 0, 1, 1]); row.pop("position_ids")
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            DSparkTrainStrategy(model).forward_loss(as_batch(row))
        with patch("specforge.algorithms.model_providers.build_registered_draft") as build, self.assertRaisesRegex(ValueError, "explicit mRoPE"):
            build_draft(SimpleNamespace(model=SimpleNamespace(input_modality="multimodal")), build_model(False).draft_model.config)
        build.assert_not_called()

    def test_accumulation_matches_pooled_gradient_with_unequal_weights(self):
        from specforge.training.controller import TrainerCore
        from tests.test_runtime.test_trainer import FakeBackend
        short = feature_row([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        long = feature_row([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        one, two = build_model(), build_model()
        ctx = StepContext(collect_detailed_metrics=False)
        TrainerCore(DSparkTrainStrategy(one), FakeBackend(one)).train_step(as_batch(build_dspark_collator()([short, long])), ctx)
        core = TrainerCore(DSparkTrainStrategy(two), FakeBackend(two), accumulation_steps=2)
        core.train_step(as_batch(short), ctx); core.train_step(as_batch(long), ctx)
        for name, parameter in one.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, dict(two.named_parameters())[name].grad, rtol=2e-5, atol=2e-6)

    def test_empty_local_contribution_has_finite_zero_gradients(self):
        model = build_model()
        row = feature_row([0, 0, 0, 1])
        out = DSparkTrainStrategy(model).forward_loss(as_batch(row), StepContext(collect_detailed_metrics=False))
        self.assertEqual(out.loss_terms[1].item(), 0)
        out.loss_terms[0].backward()
        for parameter in model.draft_model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertEqual(parameter.grad.abs().sum().item(), 0)

    def test_export_reload_and_compute_or_position_drift(self):
        from specforge.export.checkpoint_io import materialize_draft
        model = build_model(dtype=torch.bfloat16)
        state = {"draft_state_dict": DSparkTrainStrategy(model).checkpoint_state_filter(model.state_dict()),
                 **resume_contract(None, model.draft_model, model)}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(model.draft_model.config.to_dict()))
            loaded = materialize_draft(state, str(path))
            self.assertTrue(loaded.use_interleaved_mrope)
            for name, tensor in model.draft_model.state_dict().items():
                torch.testing.assert_close(tensor, loaded.state_dict()[name], rtol=0, atol=0)
            for key in ("dspark_position_convention", "dspark_mrope_theta", "dspark_mrope_section"):
                bad = copy.deepcopy(state); bad.pop(key)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    materialize_draft(bad, str(path))
            changed = copy.deepcopy(model.draft_model.config)
            changed.rope_parameters["rope_theta"] *= 2
            path.write_text(json.dumps(changed.to_dict()))
            with self.assertRaisesRegex(ValueError, "computation mismatch"):
                materialize_draft(state, str(path))
            self.assertNotEqual(resume_contract(None, model.draft_model, model),
                                resume_contract(None, build_model(False).draft_model, build_model(False)))

    def test_legacy_plain_export_still_requires_matching_original_config(self):
        from specforge.export.checkpoint_io import materialize_draft
        from specforge.training.provenance import model_source_identity
        model = build_model(False)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"; path.write_text(json.dumps(model.draft_model.config.to_dict()))
            state = {"draft_state_dict": model.draft_model.state_dict(),
                     "specforge_model_provenance": (("draft_config", model_source_identity(str(path))),)}
            self.assertFalse(materialize_draft(state, str(path)).use_interleaved_mrope)

    def test_real_trainer_resume_and_required_dspark_contract(self):
        import torch.distributed as dist
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.algorithms.common.providers import StepRuntimeConfig, MODEL_PROVENANCE_CONTRACT_KEY
        from specforge.runtime.control_plane import DataFlowController
        from specforge.runtime.data_plane.feature_store import LocalFeatureStore
        from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
        from specforge.training.trainer import Trainer
        from specforge.optimizer import BF16Optimizer
        from tests.test_runtime.test_checkpoint_resume import _no_fsdp_wrap

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            created_group = not dist.is_initialized()
            if created_group:
                dist.init_process_group("gloo", init_method="file://" + str(root / "group"), rank=0, world_size=1)
            try:
                features = root / "features"; features.mkdir()
                for i in range(4):
                    row = feature_row([0, 0, 1, 1, 1, 1])
                    torch.save({key: value.squeeze(0) for key, value in row.items()}, features / f"{i}.ckpt")

                def trainer(name, steps, checkpoint=None):
                    model = build_model()
                    contract = resume_contract(None, model.draft_model, model)
                    contract[MODEL_PROVENANCE_CONTRACT_KEY] = (("test_model", "tiny-seed-41"),)
                    # Stored fixture represents captured streaming tensors, including
                    # triplets. The public offline-v1 provider remains text-only.
                    refs = OfflineManifestReader(str(features), run_id="dspark", strategy="dspark",
                        feature_keys=tuple(row)).read()
                    with _no_fsdp_wrap():
                        built = Trainer(
                            algorithm_name="dspark",
                            make_step_strategy=algorithm_providers().step.build,
                            controller=DataFlowController("dspark"), store=LocalFeatureStore("fixture"),
                            ref_source={"refs": refs}, model=model, target_head=None,
                            optimizer_factory=lambda module: BF16Optimizer(module, lr=1e-3,
                                max_grad_norm=0.5, warmup_ratio=0.0, total_steps=4),
                            run_id="dspark", output_dir=str(root / name),
                            batch_size=1, max_steps=steps, total_steps=4, resume_from=checkpoint,
                            accumulation_steps=1, num_epochs=1, save_interval=0, logger=None, log_interval=50,
                            collate_fn=build_dspark_collator(),
                            per_sample_transform=lambda raw: {key: value.unsqueeze(0) for key, value in raw.items()},
                            strategy_kwargs=StepRuntimeConfig(options={}, resume_contract=contract,
                                allowed_missing_checkpoint_keys=frozenset()))
                    return built, model

                full, expected = trainer("full", 2)
                self.assertEqual(full.fit(), 2)
                first, _ = trainer("first", 1)
                self.assertEqual(first.fit(), 1)
                checkpoint = str((root / "first/dspark-latest").resolve())
                state_path = Path(checkpoint) / "training_state.pt"
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                resumed, actual = trainer("resumed", 2, checkpoint)
                self.assertEqual(resumed.global_step, 1)
                self.assertEqual(resumed.fit(), 2)
                for key, tensor in expected.draft_model.state_dict().items():
                    torch.testing.assert_close(actual.draft_model.state_dict()[key], tensor, rtol=0, atol=0)
                for key in ("dspark_compute_config", "dspark_position_convention",
                            "dspark_mrope_section", "dspark_mrope_theta",
                            "dspark_generation_mask", "dspark_objective_normalization"):
                    bad = copy.deepcopy(state); bad.pop(key)
                    torch.save(bad, state_path)
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                        trainer("missing-" + key, 2, checkpoint)
                bad = copy.deepcopy(state); bad["dspark_mrope_theta"] *= 2
                torch.save(bad, state_path)
                with self.assertRaisesRegex(ValueError, "dspark_mrope_theta"):
                    trainer("theta-drift", 2, checkpoint)
            finally:
                if created_group:
                    dist.destroy_process_group()

    def test_offline_filter_preserves_teacher_features(self):
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.launch import build_offline_runtime
        with tempfile.TemporaryDirectory() as temp:
            for i, ids in enumerate(([1, 7, 8], [1, 2, 7, 8])):
                torch.save({"input_ids": torch.tensor(ids), "loss_mask": torch.tensor([0] + [1] * (len(ids) - 1)),
                    "hidden_states": torch.ones(len(ids), 32), "target_last_hidden_states": torch.ones(len(ids), 16)}, Path(temp) / f"{i}.ckpt")
            with patch("specforge.launch._assemble_trainer", side_effect=lambda **kw: kw):
                built = build_offline_runtime(algorithm=builtin_algorithm_registry().resolve("dspark"),
                    hidden_states_path=temp, draft_model=build_model(False), target_head=None, optimizer_factory=None,
                    run_id="test", output_dir=str(Path(temp)/"out"), max_len=4)
            self.assertEqual(built["checkpoint_extra"]["source_dataset_size"], 1)
            tensors, handle = built["store"].get(built["ref_source"]["refs"][0])
            try:
                normalized = built["per_sample_transform"](tensors)
            finally:
                built["store"].release(handle)
            self.assertEqual(normalized["loss_mask"].tolist(), [[0, 1, 1, 0]])
            self.assertEqual(tuple(normalized["target_last_hidden_states"].shape), (1, 4, 16))
