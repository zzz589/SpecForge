"""Built-in DSpark registration and executable providers."""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
)
from specforge.algorithms.common.hidden_states_data import (
    DSPARK_NORMALIZER_ID,
    build_dspark_collator,
    build_dspark_offline_normalizer,
    build_dspark_offline_reader,
)
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineCaptureLayout,
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepProvider,
    make_registration,
)
from specforge.algorithms.common.vlm_input import build_vlm_input_adapter
from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)
from specforge.data.loss_mask import has_consecutive_supervised_tokens
from specforge.algorithms.dflash.offline_data import prepare_offline_data

ALGORITHM_NAME = "dspark"
DRAFT_ARCHITECTURE = "DSparkDraftModel"
COMPATIBLE_DRAFT_ARCHITECTURES = frozenset({DRAFT_ARCHITECTURE})


def build_step(wrapped_model, *, target_head=None, **_options):
    del target_head
    from specforge.training.strategies.base import DSparkTrainStrategy

    return DSparkTrainStrategy(wrapped_model)


def resume_contract(_config, draft_model, training_model):
    """Persist resolved DSpark model, sampling, and objective semantics."""

    from specforge.modeling.draft.dflash_config import dflash_compute_config

    contract = {
        "dspark_compute_config": dflash_compute_config(draft_model.config),
        "dspark_objective_normalization": "global_weighted_tokens_per_optimizer_step_v1",
        "dspark_generation_mask": "first_eos_in_supervised_run_v1",
        "dspark_draft_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "dspark_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "dspark_block_size": int(training_model.block_size),
        "dspark_mask_token_id": int(training_model.mask_token_id),
        "dspark_attention_backend": str(training_model.attention_backend),
        "dspark_num_anchors": int(training_model.num_anchors),
        "dspark_loss_decay_gamma": training_model.loss_decay_gamma,
        "dspark_ce_loss_alpha": float(training_model.dspark_ce_loss_alpha),
        "dspark_l1_loss_alpha": float(training_model.dspark_l1_loss_alpha),
        "dspark_confidence_head_alpha": float(
            training_model.dspark_confidence_head_alpha
        ),
    }
    if getattr(draft_model, "use_interleaved_mrope", False):
        from specforge.modeling.draft.dflash import get_rope_scaling_value

        contract["dspark_position_convention"] = "qwen3_vl_triplets_next_token_anchor_continuation_v1"
        contract["dspark_mrope_section"] = tuple(get_rope_scaling_value(draft_model.config, "mrope_section"))
        contract["dspark_mrope_theta"] = get_rope_scaling_value(
            draft_model.config, "rope_theta", getattr(draft_model.config, "rope_theta", None)
        )
    return contract


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    if config.model.input_modality == "multimodal":
        from specforge.modeling.draft.dflash import get_rope_scaling_value

        if get_rope_scaling_value(draft_config, "mrope_interleaved", False) is not True:
            raise ValueError("DSpark-VL requires an explicit mRoPE draft config; use configs/qwen3-vl-4b-dspark-mrope.json")
    return build_registered_draft(config, draft_config)


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.model_providers import build_dspark_model

    from specforge.modeling.target.target_utils import target_text_config

    if getattr(draft_model.config, "eos_token_id", None) is None:
        draft_model.config.eos_token_id = getattr(target_text_config(target_config), "eos_token_id", None)
        if draft_model.config.eos_token_id is None:
            draft_model.config.eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if config.model.input_modality == "multimodal":
        if getattr(target_config, "model_type", None) != "qwen3_vl":
            raise ValueError("DSpark-VL mRoPE supports Qwen3-VL still images only")
        if not getattr(draft_model, "use_interleaved_mrope", False):
            raise ValueError("DSpark-VL requires an mRoPE-enabled draft")
    return build_dspark_model(
        config,
        draft_model,
        draft_config,
        target_config,
        tokenizer,
    )


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.algorithms.model_providers import resolve_dflash_capture_layers

    return resolve_dflash_capture_layers(config, draft_config, target_config)


def minimum_loss_tokens(config, draft_config):
    from specforge.algorithms.model_providers import dflash_min_loss_tokens

    return dflash_min_loss_tokens(config, draft_config)


def needs_input_tools(config, draft_model):
    from specforge.algorithms.model_providers import dflash_needs_input_tools

    return dflash_needs_input_tools(config, draft_model)


def algorithm_spec() -> AlgorithmSpec:
    ready = {
        "input_ids",
        "loss_mask",
        "hidden_states",
        "target_last_hidden_states",
    }
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures=COMPATIBLE_DRAFT_ARCHITECTURES,
            default_architecture=DRAFT_ARCHITECTURE,
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=ready,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
                storage=OfflineStorageContract(
                    format="specforge_hidden_states_v1",
                    required_tensors=ready,
                    normalizer=DSPARK_NORMALIZER_ID,
                ),
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=ready,
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="multimodal",
                required_tensors=ready | {"position_ids"},
                allowed_target_representations={"hidden_state"},
                default_target_representation="hidden_state",
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"eager", "sdpa", "flex_attention"},
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=empty_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                compatible_architectures=COMPATIBLE_DRAFT_ARCHITECTURES,
                expected_auto_map_model="dspark.DSparkDraftModel",
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=needs_input_tools,
            default_dataloader_num_workers=8,
            loss_mask_filter=has_consecutive_supervised_tokens,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=DSPARK_NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="dspark",
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                    ),
                ),
                build_reader=partial(build_dspark_offline_reader, ALGORITHM_NAME),
                prepare_data=prepare_offline_data,
                build_normalizer=build_dspark_offline_normalizer,
                build_collator=build_dspark_collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dspark",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=build_dspark_collator,
            ),
            ServerStreamingProvider(
                modality="multimodal",
                capture_method="dspark",
                target_representation="hidden_state",
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature="target_last_hidden_states",
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                        ("position_ids", "position_ids", (3,)),
                    ),
                ),
                build_collator=build_dspark_collator,
                build_input_adapter=partial(
                    build_vlm_input_adapter,
                    minimum_loss_tokens=minimum_loss_tokens,
                ),
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
