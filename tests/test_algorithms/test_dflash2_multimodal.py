from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml
from transformers import AutoConfig

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.contracts import FeatureMode
from specforge.data.vlm_preprocessing import build_qwen3_vl_position_ids
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel


REPO_ROOT = Path(__file__).resolve().parents[2]
DRAFT_CONFIG = REPO_ROOT / "configs" / "qwen3-vl-4b-dflash2-b16.json"
RECIPE = (
    REPO_ROOT
    / "examples"
    / "configs"
    / "online"
    / "disaggregated"
    / "external"
    / "qwen3-vl-4b-dflash2-b16.yaml"
)


class DFlash2MultimodalCompositionTest(unittest.TestCase):
    """Lock the DFlash2 architecture to PR785's multimodal data contract."""

    def test_recipe_selects_dflash2_through_the_dflash_strategy(self):
        recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
        draft = json.loads(DRAFT_CONFIG.read_text(encoding="utf-8"))

        self.assertEqual(recipe["training"]["strategy"], "dflash")
        self.assertEqual(recipe["model"]["input_modality"], "multimodal")
        self.assertFalse(recipe["training"]["dflash_teacher_metrics"])
        self.assertEqual(draft["architectures"], ["DFlash2DraftModel"])
        self.assertEqual(draft["dflash_config"]["block_size"], 16)
        self.assertEqual(draft["dflash_config"]["selector_top_k"], 16)

    def test_config_builds_the_registered_dflash2_model(self):
        config = AutoConfig.from_pretrained(DRAFT_CONFIG)
        # Keep this an architecture-composition test rather than allocating the
        # full five-layer, 2,560-wide training draft on a CPU test runner.
        config.hidden_size = 16
        config.intermediate_size = 32
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 4
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        config.vocab_size = 32
        config.block_size = 4
        config.dflash_config = {
            "block_size": 4,
            "conv_group_size": 4,
            "conv_kernel_size": 2,
            "mask_token_id": 31,
            "selector_rank": 4,
            "selector_top_k": 3,
            "target_layer_ids": [1],
        }
        config._attn_implementation = "eager"

        model = AutoDraftModel.from_config(config)

        self.assertIsInstance(model, DFlash2DraftModel)
        self.assertEqual(model.block_size, 4)
        self.assertIsNotNone(model.candidate_selector)
        self.assertEqual(model.target_layer_ids, [1])

    def test_multimodal_provider_matches_dflash2_training_inputs(self):
        registration = builtin_algorithm_registry().resolve("dflash")
        contract = registration.spec.feature_contract(
            FeatureMode.STREAMING, "multimodal"
        )
        provider = registration.providers.server_streaming_for("multimodal")

        self.assertIn(
            "DFlash2DraftModel", registration.spec.draft.compatible_architectures
        )
        self.assertEqual(
            set(contract.required_tensors),
            {"input_ids", "loss_mask", "hidden_states", "position_ids"},
        )
        self.assertEqual(provider.capture_method, "dflash")
        self.assertIsNone(provider.layout.last_hidden_feature)
        self.assertIn(
            ("position_ids", "position_ids", (3,)), provider.layout.passthrough
        )

    def test_qwen3_vl_position_ids_use_the_merged_image_grid(self):
        positions = build_qwen3_vl_position_ids(
            [10, 11, 99, 99, 99, 99, 12],
            image_pad_token_id=99,
            image_grid_thw=[1, 4, 4],
            merge_size=2,
            source="test",
        )

        self.assertEqual(
            positions,
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 3],
                [2, 3, 2],
                [2, 3, 3],
                [4, 4, 4],
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
