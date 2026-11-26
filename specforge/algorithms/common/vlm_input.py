# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Multimodal (image+text) ServerInputAdapter for server-side capture.

Owns the three modality seams for ``modality="multimodal"``:

- ``load_input_tools``: the target tokenizer (the processor is loaded lazily
  in ``prepare_prompts`` so training-model construction receives a plain
  tokenizer, exactly like the text path).
- ``prepare_prompts``: ShareGPT-style JSONL (+ optional ``image`` field) ->
  payload dicts with expanded ``input_ids``/``loss_mask`` (what the trainer
  and the passthrough capture use) plus ``request_input_ids`` (single
  placeholder) and ``data:``-URI ``image_data`` (what the capture request sends).
- ``build_request_inputs``: batch payloads -> the ``/generate`` model-input
  fields ``{"input_ids", "image_data"}``.

The capture server must run with ``SGLANG_MM_AVOID_RETOKENIZE=1`` so its
multimodal processor re-expands placeholders in id space (no retokenization
drift); the managed launcher sets this for ``input_modality="multimodal"``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence


class VlmServerInputAdapter:
    """Image+text input adapter for the SGLang server-capture transport.

    ``minimum_loss_tokens`` is the algorithm-owned eligibility floor injected
    by the registering provider (e.g. DSpark passes its own); ``None`` keeps
    the DFlash default.
    """

    def __init__(
        self,
        config: Any,
        *,
        minimum_loss_tokens: Callable[[Any, Any], int] | None = None,
    ) -> None:
        self._config = config
        self._minimum_loss_tokens = minimum_loss_tokens

    def load_input_tools(self, config: Any) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.target_model_path,
            cache_dir=config.model.cache_dir,
            trust_remote_code=config.model.trust_remote_code,
        )
        if config.model.tokenizer_pad_token_id is not None:
            tokenizer.pad_token_id = config.model.tokenizer_pad_token_id
        elif tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def prepare_prompts(
        self,
        config: Any,
        input_tools: Any,
        *,
        draft_config: Any,
    ) -> Sequence[dict[str, Any]]:
        from transformers import AutoProcessor

        from specforge.data.vlm_preprocessing import (
            build_vlm_prompt_payloads, build_mapped_vlm_prompt_payloads,
        )

        minimum_loss_tokens = self._minimum_loss_tokens
        if minimum_loss_tokens is None:
            from specforge.algorithms.model_providers import dflash_min_loss_tokens

            minimum_loss_tokens = dflash_min_loss_tokens

        tokenizer = input_tools
        # The processor must inherit the target's own preprocessor config
        # (min/max pixels, merge size) so the client expansion matches the
        # capture server's expansion one-for-one; no overrides are accepted.
        processor = AutoProcessor.from_pretrained(
            config.model.target_model_path,
            cache_dir=config.model.cache_dir,
            trust_remote_code=config.model.trust_remote_code,
        )
        generation_eos_token_id = None
        if getattr(getattr(config, "training", None), "strategy", None) in {"dflash", "dspark"}:
            from transformers import AutoConfig

            target_config = AutoConfig.from_pretrained(
                config.model.target_model_path,
                cache_dir=config.model.cache_dir,
                trust_remote_code=config.model.trust_remote_code,
            )
            if target_config.model_type != "qwen3_vl":
                raise ValueError("DFlash/DSpark-VL mRoPE positions currently support Qwen3-VL still images only")
            generation_eos_token_id = getattr(draft_config, "eos_token_id", None)
            if generation_eos_token_id is None:
                generation_eos_token_id = tokenizer.eos_token_id
        path = config.data.prompts_path or config.data.train_data_path
        if not path:
            raise ValueError("multimodal prompt preparation requires a data path")
        if getattr(config.data, "mapped_manifest_path", None):
            return build_mapped_vlm_prompt_payloads(
                path, config.data.mapped_manifest_path, tokenizer,
                target_model_path=config.model.target_model_path,
                chat_template=config.data.chat_template,
                max_length=config.data.max_length,
                min_loss_tokens=minimum_loss_tokens(config, draft_config),
                max_prompts=config.data.max_prompts,
                generation_eos_token_id=generation_eos_token_id,
                processor=processor,
            )
        return build_vlm_prompt_payloads(
            path,
            tokenizer,
            processor,
            chat_template=config.data.chat_template,
            max_length=config.data.max_length,
            min_loss_tokens=minimum_loss_tokens(config, draft_config),
            max_prompts=config.data.max_prompts,
            generation_eos_token_id=generation_eos_token_id,
            exact_journal_path=getattr(config.data, "exact_journal_path", None),
        )

    def build_request_inputs(
        self,
        tasks: Sequence[Any],
    ) -> Mapping[str, Any]:
        return {
            "input_ids": [list(task.payload["request_input_ids"]) for task in tasks],
            "image_data": [task.payload["image_data"] for task in tasks],
        }


def build_vlm_input_adapter(
    config: Any,
    *,
    minimum_loss_tokens: Callable[[Any, Any], int] | None = None,
) -> VlmServerInputAdapter:
    return VlmServerInputAdapter(config, minimum_loss_tokens=minimum_loss_tokens)


__all__ = ["VlmServerInputAdapter", "build_vlm_input_adapter"]
