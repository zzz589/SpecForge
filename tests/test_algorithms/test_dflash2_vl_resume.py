import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from specforge.algorithms.dflash.providers import algorithm_providers, resume_contract
from specforge.algorithms.common.hidden_states_data import build_collator
from tests.test_algorithms.test_dflash2_vl_unified import build_model
from tests.test_algorithms.test_dspark_vl_mrope import feature_row

class DFlash2ResumeTest(unittest.TestCase):
    def test_real_trainer_resume_and_required_dflash_contract(self):
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
                    refs = OfflineManifestReader(str(features), run_id="dflash", strategy="dflash",
                        feature_keys=tuple(row)).read()
                    with _no_fsdp_wrap():
                        built = Trainer(
                            algorithm_name="dflash",
                            make_step_strategy=algorithm_providers().step.build,
                            controller=DataFlowController("dflash"), store=LocalFeatureStore("fixture"),
                            ref_source={"refs": refs}, model=model, target_head=None,
                            optimizer_factory=lambda module: BF16Optimizer(module, lr=1e-3,
                                max_grad_norm=0.5, warmup_ratio=0.0, total_steps=4),
                            run_id="dflash", output_dir=str(root / name),
                            batch_size=1, max_steps=steps, total_steps=4, resume_from=checkpoint,
                            accumulation_steps=1, num_epochs=1, save_interval=0, logger=None, log_interval=50,
                            collate_fn=build_collator(),
                            per_sample_transform=lambda raw: {key: value.unsqueeze(0) for key, value in raw.items()},
                            strategy_kwargs=StepRuntimeConfig(options={}, resume_contract=contract,
                                allowed_missing_checkpoint_keys=frozenset()))
                    return built, model

                full, expected = trainer("full", 2)
                self.assertEqual(full.fit(), 2)
                first, _ = trainer("first", 1)
                self.assertEqual(first.fit(), 1)
                checkpoint = str((root / "first/dflash-latest").resolve())
                state_path = Path(checkpoint) / "training_state.pt"
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                resumed, actual = trainer("resumed", 2, checkpoint)
                self.assertEqual(resumed.global_step, 1)
                self.assertEqual(resumed.fit(), 2)
                for key, tensor in expected.draft_model.state_dict().items():
                    torch.testing.assert_close(actual.draft_model.state_dict()[key], tensor, rtol=0, atol=0)
                for key in ("dflash_position_convention", "dflash2_selector_top_k",
                            "dflash_mrope_section", "dflash_mrope_theta",
                            "dflash_generation_mask", "dflash_objective_normalization"):
                    bad = copy.deepcopy(state); bad.pop(key)
                    torch.save(bad, state_path)
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                        trainer("missing-" + key, 2, checkpoint)
                bad = copy.deepcopy(state); bad["dflash_mrope_theta"] *= 2
                torch.save(bad, state_path)
                with self.assertRaisesRegex(ValueError, "dflash_mrope_theta"):
                    trainer("theta-drift", 2, checkpoint)
            finally:
                if created_group:
                    dist.destroy_process_group()

