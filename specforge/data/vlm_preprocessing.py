# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""VLM (image+text) prompt preparation for server-side multimodal capture.

The multimodal online path needs, per sample:

- ``input_ids`` / ``loss_mask``: the FULLY image-expanded token sequence.
  These are what the trainer consumes and what the passthrough capture stores,
  so they must match the capture server's own expansion one-for-one.
- ``request_input_ids``: the same sequence with each image region collapsed
  back to a single ``<|image_pad|>`` placeholder. This is what the capture
  request sends; the patched server re-expands it in id space
  (``SGLANG_MM_AVOID_RETOKENIZE=1``), which is guaranteed to reproduce the
  client expansion because both sides run the target model's own HF processor
  on the same image.
- ``image_data``: a ``data:``-URI image string for the capture request (the image
  itself never enters the feature store; only text tokens and captured hidden
  states do).

Limitations (v1): at most one image per sample; structured user messages
preserve the image position. Legacy text-only messages with a separate image
field attach it to the first user turn. Text-only samples are
supported in the same run (``image_data=None``). Image references are read
from ``image`` / ``image_path`` (string) or ``images`` (one-element list);
unreadable images and multi-image samples are fatal ``ImageDataError``s,
never a silent text-only downgrade.
"""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Iterator, Sequence
from typing import Any, Dict, List, Optional, Tuple

from .prompt_builder import _iter_records
from .template import TEMPLATE_REGISTRY

_IMAGE_PLACEHOLDER = "<|image_pad|>"
_VISION_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>"


def _image_token_count(image_grid_thw, merge_size: int) -> int:
    """Merged token count of one image region from its processor grid."""

    grid = list(image_grid_thw[0])
    count = 1
    for dim in grid:
        count *= int(dim)
    return count // (merge_size * merge_size)


def _normalize_image_grid_thw(value: Any, *, source: str) -> Optional[List[int]]:
    """Normalize one cached/processor image grid to ``[temporal, height, width]``."""

    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{source}: image_grid_thw must contain exactly 3 integers")
    grid = [int(item) for item in value]
    if any(item <= 0 for item in grid):
        raise ValueError(f"{source}: image_grid_thw values must be positive")
    return grid


def build_qwen3_vl_position_ids(
    input_ids: Sequence[int],
    *,
    image_pad_token_id: int,
    image_grid_thw: Optional[Sequence[int]],
    merge_size: int,
    source: str,
) -> List[List[int]]:
    """Build Qwen3-VL's interleaved temporal/height/width position triplets.

    This mirrors ``Qwen3VLModel.get_rope_index`` for the framework's v1
    single-image contract. Text tokens advance all three axes together. The
    expanded image-pad run uses the merged vision grid and the following text
    resumes at one past the image grid's maximum coordinate.
    """

    ids = [int(token) for token in input_ids]
    positions: List[List[int]] = []
    current_pos = 0
    cursor = 0
    image_runs = 0
    while cursor < len(ids):
        is_image = ids[cursor] == image_pad_token_id
        end = cursor + 1
        while end < len(ids) and (ids[end] == image_pad_token_id) == is_image:
            end += 1
        run_length = end - cursor
        if not is_image:
            positions.extend(
                [[position, position, position]
                 for position in range(current_pos, current_pos + run_length)]
            )
            current_pos += run_length
        else:
            image_runs += 1
            if image_runs > 1:
                raise ValueError(f"{source}: multiple image token runs are unsupported")
            grid = _normalize_image_grid_thw(image_grid_thw, source=source)
            if grid is None:
                raise ValueError(f"{source}: image tokens require image_grid_thw")
            temporal, raw_height, raw_width = grid
            if raw_height % merge_size or raw_width % merge_size:
                raise ValueError(
                    f"{source}: image grid {grid} is not divisible by merge_size={merge_size}"
                )
            height = raw_height // merge_size
            width = raw_width // merge_size
            expected = temporal * height * width
            if run_length != expected:
                raise ValueError(
                    f"{source}: expanded image run has {run_length} tokens, "
                    f"but image_grid_thw={grid} requires {expected}"
                )
            for temporal_index in range(temporal):
                for height_index in range(height):
                    for width_index in range(width):
                        positions.append(
                            [
                                current_pos + temporal_index,
                                current_pos + height_index,
                                current_pos + width_index,
                            ]
                        )
            # Qwen3-VL advances by the largest merged spatial axis.
            current_pos += max(raw_height, raw_width) // merge_size
        cursor = end
    if image_runs != (1 if image_grid_thw is not None else 0):
        raise ValueError(
            f"{source}: image grid/token-run mismatch (runs={image_runs}, "
            f"grid_present={image_grid_thw is not None})"
        )
    return positions


class ImageDataError(ValueError):
    """Fatal image-contract violation (unreadable image, multi-image sample).

    Raised instead of silently degrading an image-bearing sample to text-only
    training; the prompt-preparation loop re-raises it to abort loudly.
    """


def _extract_image_field(record: Dict[str, Any], *, source: str) -> Optional[str]:
    """Resolve the single image reference of one record, or None.

    Accepts ``image`` / ``image_path`` (string) and ``images`` (a one-element
    list, the convention used by common VLM corpora). A record carrying more
    than one image is a fatal error in v1 (single-image contract), and so is
    an ``images`` list whose element is not a string.
    """

    image_field = record.get("image") or record.get("image_path")
    images = record.get("images")
    if image_field and images:
        raise ImageDataError(
            f"{source}: record has both 'image' and 'images' fields; use one"
        )
    if images is None:
        return image_field
    if not isinstance(images, list):
        raise ImageDataError(
            f"{source}: 'images' must be a list, got {type(images).__name__}"
        )
    if len(images) == 0:
        return image_field
    if len(images) > 1:
        raise ImageDataError(
            f"{source}: multi-image samples are not supported (got "
            f"{len(images)} images); the v1 contract is one image per sample"
        )
    first = images[0]
    if not isinstance(first, str):
        raise ImageDataError(
            f"{source}: 'images[0]' must be a path or base64 string, got "
            f"{type(first).__name__}"
        )
    return first


def _as_data_uri(raw: bytes, *, media_type: str = "image/jpeg") -> str:
    """Wrap raw image bytes as a data URI.

    Some SGLang builds (e.g. the Ascend 0.5.14 line) only route ``data:``
    URIs through base64 decoding and treat every other string as a file
    path; the data-URI form is accepted by all supported server versions.
    """

    return f"data:{media_type};base64," + base64.b64encode(raw).decode("ascii")


def _load_image(image_field: Any, *, source: str):
    """Return (pil_image, data_uri) from a path / base64 / data-URI field."""

    from PIL import Image

    if not isinstance(image_field, str) or not image_field:
        raise ImageDataError(
            f"{source}: image field must be a file path or base64 string, got "
            f"{type(image_field).__name__}"
        )
    if os.path.isfile(image_field):
        with open(image_field, "rb") as image_file:
            raw = image_file.read()
        return Image.open(io.BytesIO(raw)).convert("RGB"), _as_data_uri(raw)
    encoded = image_field
    if encoded.startswith("data:"):
        media_type = encoded[5 : encoded.index(";")] if ";" in encoded else "image/jpeg"
        encoded = encoded.split(",", 1)[-1]
    else:
        media_type = "image/jpeg"
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ImageDataError(
            f"{source}: image field is neither an existing file nor valid base64"
        ) from exc
    return (
        Image.open(io.BytesIO(raw)).convert("RGB"),
        _as_data_uri(raw, media_type=media_type),
    )


def _conversation_messages(
    conversations: List[Dict[str, Any]],
    *,
    with_image: bool,
    source: str,
) -> list:
    """Preserve ordered content parts; synthesize only a missing legacy marker."""

    messages: List[Dict[str, Any]] = []
    image_count = 0
    for message in conversations:
        role = message.get("role", message.get("from", ""))
        content = message.get("content")
        if content is None:
            content = message.get("value", "")
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") in ("image", "image_url") and role == "user":
                    parts.append({"type": "image"})
                    image_count += 1
                elif part.get("type") == "text":
                    parts.append({"type": "text", "text": part["text"]})
                else:
                    raise ImageDataError(f"{source}: unsupported content part")
            content = parts
        elif not isinstance(content, str):
            raise ValueError(f"{source}: unsupported message content type")
        messages.append({"role": role, "content": content})
    if image_count > 1 or (image_count and not with_image):
        raise ImageDataError(f"{source}: structured image marker/asset mismatch")
    if with_image and not image_count:
        first = next((m for m in messages if m["role"] == "user"), None)
        if first is None:
            raise ImageDataError(f"{source}: image sample has no user turn")
        parts = first["content"]
        if isinstance(parts, str):
            parts = [{"type": "text", "text": parts}]
        first["content"] = [{"type": "image"}, *parts]
    return messages


def _render_conversation_text(tokenizer, conversations, *, with_image, source):
    messages = _conversation_messages(conversations, with_image=with_image, source=source)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _expand_image_region(
    ids: List[int],
    mask: List[int],
    *,
    pad_token_id: int,
    count: int,
    source: str,
) -> Tuple[List[int], List[int]]:
    """Splice the single image placeholder into ``count`` copies in id space."""

    positions = [index for index, token in enumerate(ids) if token == pad_token_id]
    if len(positions) != 1:
        raise ValueError(
            f"{source}: expected exactly 1 image placeholder token, found "
            f"{len(positions)}"
        )
    index = positions[0]
    expanded_ids = ids[:index] + [pad_token_id] * count + ids[index + 1 :]
    expanded_mask = mask[:index] + [0] * count + mask[index + 1 :]
    return expanded_ids, expanded_mask


def _unwrap_token_row(value: Any, *, source: str) -> List[int]:
    """Normalize one Arrow row written as ``[1, seq]`` into ``[seq]``."""

    while (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    if not isinstance(value, list):
        raise ValueError(f"{source}: expected a list token row")
    return [int(item) for item in value]


def _collapse_mapped_image_region(
    ids: List[int],
    mask: List[int],
    *,
    pad_token_id: int,
    has_image: bool,
    source: str,
) -> Tuple[List[int], List[int]]:
    """Collapse each contiguous mapped image-pad run to one request token."""

    if len(ids) != len(mask):
        raise ValueError(
            f"{source}: input_ids/loss_mask length mismatch "
            f"({len(ids)} != {len(mask)})"
        )
    collapsed_ids: List[int] = []
    collapsed_mask: List[int] = []
    in_image_run = False
    image_runs = 0
    for token, loss in zip(ids, mask):
        if token == pad_token_id:
            if in_image_run:
                continue
            in_image_run = True
            image_runs += 1
        else:
            in_image_run = False
        collapsed_ids.append(token)
        collapsed_mask.append(loss)
    expected_runs = 1 if has_image else 0
    if image_runs != expected_runs:
        raise ValueError(
            f"{source}: expected {expected_runs} mapped image region(s), "
            f"found {image_runs}"
        )
    return collapsed_ids, collapsed_mask


class _MappedVlmPromptSequence(Sequence[Dict[str, Any]]):
    """Expose memory-mapped VLM Arrow rows without remapping or pixel copies."""

    def __init__(
        self,
        dataset,
        image_fields: Sequence[Optional[str]],
        *,
        pad_token_id: int,
        max_length: int,
        min_loss_tokens: int,
        merge_size: int,
        processor=None,
        limit: Optional[int] = None,
    ) -> None:
        if len(dataset) != len(image_fields):
            raise ValueError(
                "mapped VLM dataset/image index length mismatch: "
                f"{len(dataset)} != {len(image_fields)}"
            )
        self._dataset = dataset
        self._image_fields = image_fields
        self._pad_token_id = int(pad_token_id)
        self._max_length = int(max_length)
        self._min_loss_tokens = int(min_loss_tokens)
        self._merge_size = int(merge_size)
        self._processor = processor
        self._length = len(dataset) if limit is None else min(len(dataset), int(limit))

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source = f"mapped VLM row {index}"
        record = self._dataset[index]
        ids = _unwrap_token_row(record["input_ids"], source=source)
        mask = _unwrap_token_row(record["loss_mask"], source=source)
        if len(ids) > self._max_length:
            raise ValueError(
                f"{source}: cached sequence length {len(ids)} exceeds configured "
                f"max_length {self._max_length}"
            )
        if sum(mask) < self._min_loss_tokens:
            raise ValueError(
                f"{source}: cached loss mask has fewer than "
                f"{self._min_loss_tokens} trainable tokens"
            )
        image_field = self._image_fields[index]
        image_grid_thw = _normalize_image_grid_thw(
            record.get("image_grid_thw"), source=source
        )
        if image_field and image_grid_thw is None:
            if self._processor is None:
                raise ValueError(
                    f"{source}: mapped cache has no image_grid_thw and no "
                    "processor fallback was configured"
                )
            pil_image, _ = _load_image(image_field, source=source)
            processor_output = self._processor.image_processor(
                images=[pil_image], return_tensors="pt"
            )
            image_grid_thw = _normalize_image_grid_thw(
                processor_output["image_grid_thw"], source=source
            )
        request_ids, _ = _collapse_mapped_image_region(
            ids,
            mask,
            pad_token_id=self._pad_token_id,
            has_image=bool(image_field),
            source=source,
        )
        position_ids = build_qwen3_vl_position_ids(
            ids,
            image_pad_token_id=self._pad_token_id,
            image_grid_thw=image_grid_thw,
            merge_size=self._merge_size,
            source=source,
        )
        return {
            "payload": {
                "input_ids": ids,
                "loss_mask": mask,
                "position_ids": position_ids,
                "request_input_ids": request_ids,
                # Keep only the path/base64 source in the resident prompt list.
                # The request adapter converts one leased sample at a time.
                "image_path": image_field,
            }
        }


def build_mapped_vlm_prompt_payloads(
    path: str,
    manifest_path: str,
    tokenizer,
    *,
    target_model_path: str,
    chat_template: Optional[str],
    max_length: int,
    min_loss_tokens: int = 1,
    max_prompts: Optional[int] = None,
) -> Sequence[Dict[str, Any]]:
    """Reuse a sharded VLM Arrow manifest without calling ``Dataset.map``.

    The mapper shuffles every source shard independently with the manifest's
    seed.  The companion JSONL retains pre-shuffle order, so this loader
    reconstructs that deterministic permutation before attaching image paths.
    Only token/loss-mask columns stay in the Arrow view; cached pixel tensors
    are neither read nor copied because the capture server owns vision encode.
    """

    from datasets import Dataset, concatenate_datasets
    import numpy as np

    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not manifest.get("complete"):
        raise ValueError(f"mapped VLM manifest is incomplete: {manifest_path}")
    if manifest.get("storage_format") != "sharded_manifest":
        raise ValueError(
            "mapped VLM fast path requires storage_format='sharded_manifest', "
            f"got {manifest.get('storage_format')!r}"
        )
    if manifest.get("chat_template") != chat_template:
        raise ValueError(
            "mapped VLM chat_template mismatch: "
            f"manifest={manifest.get('chat_template')!r}, config={chat_template!r}"
        )
    if int(manifest.get("max_length", -1)) != int(max_length):
        raise ValueError(
            "mapped VLM max_length mismatch: "
            f"manifest={manifest.get('max_length')}, config={max_length}"
        )
    manifest_target = manifest.get("target_model_path")
    if manifest_target and os.path.realpath(manifest_target) != os.path.realpath(
        target_model_path
    ):
        raise ValueError(
            "mapped VLM target model mismatch: "
            f"manifest={manifest_target!r}, config={target_model_path!r}"
        )

    arrow_files = [
        file_name
        for shard in manifest.get("shards", [])
        for file_name in sorted(shard.get("files", []))
    ]
    if not arrow_files:
        raise ValueError(f"mapped VLM manifest has no Arrow files: {manifest_path}")
    missing = [file_name for file_name in arrow_files if not os.path.isfile(file_name)]
    if missing:
        raise FileNotFoundError(
            "mapped VLM manifest references missing Arrow files: "
            + ", ".join(missing[:5])
        )
    mapped = concatenate_datasets(
        [Dataset.from_file(file_name) for file_name in arrow_files]
    )
    required_columns = {"input_ids", "loss_mask"}
    missing_columns = required_columns - set(mapped.column_names)
    if missing_columns:
        raise ValueError(
            f"mapped VLM dataset is missing columns: {sorted(missing_columns)}"
        )
    kept_columns = required_columns | ({"image_grid_thw"} & set(mapped.column_names))
    mapped = mapped.remove_columns(
        [column for column in mapped.column_names if column not in kept_columns]
    )

    raw_image_fields: List[Optional[str]] = []
    for line_number, record in _iter_records(path):
        raw_image_fields.append(
            _extract_image_field(record, source=f"{path}:{line_number}")
        )
    expected_rows = int(manifest.get("dataset_size", -1))
    if len(mapped) != expected_rows or len(raw_image_fields) != expected_rows:
        raise ValueError(
            "mapped VLM row-count mismatch: "
            f"manifest={expected_rows}, arrow={len(mapped)}, "
            f"jsonl={len(raw_image_fields)}"
        )

    mapped_to_raw: List[int] = []
    raw_offset = 0
    shuffle_seed = int(manifest.get("shuffle_seed", 42))
    for shard in manifest["shards"]:
        row_count = int(shard["rows"])
        order = np.random.default_rng(shuffle_seed).permutation(row_count)
        mapped_to_raw.extend(raw_offset + int(index) for index in order)
        raw_offset += row_count
    if raw_offset != expected_rows or len(mapped_to_raw) != expected_rows:
        raise ValueError(
            "mapped VLM shard row-count mismatch: "
            f"manifest={expected_rows}, reconstructed={len(mapped_to_raw)}"
        )
    ordered_image_fields = [raw_image_fields[index] for index in mapped_to_raw]

    pad_token_id = tokenizer.convert_tokens_to_ids(_IMAGE_PLACEHOLDER)
    if not isinstance(pad_token_id, int):
        raise ValueError(
            f"tokenizer has no {_IMAGE_PLACEHOLDER!r} special token"
        )
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(target_model_path)
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    uses_cached_grids = "image_grid_thw" in mapped.column_names
    processor_fallback = None if uses_cached_grids else processor
    limit = None if max_prompts in (None, 0) else int(max_prompts)
    print(
        "MAPPED VLM CACHE HIT: "
        f"rows={expected_rows}, source_shards={len(manifest['shards'])}, "
        f"arrow_files={len(arrow_files)}, manifest={manifest_path}; "
        "Dataset.map skipped; "
        + (
            "cached image_grid_thw reused"
            if uses_cached_grids
            else "image_grid_thw will be derived lazily (no remap)"
        ),
        flush=True,
    )
    print(
        "MAPPED VLM PIXEL CONTRACT: "
        f"min_pixels={manifest.get('min_pixels')}, "
        f"max_pixels={manifest.get('max_pixels')}",
        flush=True,
    )
    return _MappedVlmPromptSequence(
        mapped,
        ordered_image_fields,
        pad_token_id=pad_token_id,
        max_length=max_length,
        min_loss_tokens=min_loss_tokens,
        merge_size=merge_size,
        processor=processor_fallback,
        limit=limit,
    )


def build_vlm_prompt_payloads(
    path: str,
    tokenizer,
    processor,
    *,
    chat_template: Optional[str],
    max_length: int,
    min_loss_tokens: int = 1,
    max_prompts: Optional[int] = None,
    exact_journal_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build prompt payloads for multimodal server capture from a JSONL file.

    Records follow the ShareGPT shape (``conversations`` column); an optional
    top-level ``image`` field carries one image (file path or base64). Returns
    control-plane dicts ``{"payload": {...}}`` mirroring
    :func:`specforge.data.prompt_builder.prepare_prompt_tasks`.
    """

    template = TEMPLATE_REGISTRY.get(chat_template) if chat_template else None
    pad_token_id = tokenizer.convert_tokens_to_ids(_IMAGE_PLACEHOLDER)
    if not isinstance(pad_token_id, int):
        raise ValueError(
            f"tokenizer has no {_IMAGE_PLACEHOLDER!r} special token; is this a "
            "Qwen-VL-family tokenizer?"
        )
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))

    payloads: List[Dict[str, Any]] = []
    from .exact_vlm import ExactJournal
    journal = ExactJournal(exact_journal_path) if exact_journal_path is not None else None
    if journal and max_prompts not in (None, 0):
        raise ValueError("exact mode requires a complete matching generation journal")
    skipped = 0
    for line_number, record in _iter_records(path):
        source = f"{path}:{line_number}"
        if "visual_scores" in record and journal is None:
            raise ValueError(f"{source}: visual scores require a trusted exact journal")
        if journal and not all(record.get(k) for k in ("id", "generation_prompt_ids", "response_token_ids")):
            raise ValueError(f"{source}: missing exact id or token IDs")
        try:
            prepared = _prepare_one_record(
                record,
                source=source,
                tokenizer=tokenizer,
                processor=processor,
                template=template,
                pad_token_id=pad_token_id,
                merge_size=merge_size,
                max_length=max_length,
                min_loss_tokens=min_loss_tokens,
            )
        except ImageDataError:
            # Image-contract violations are fatal: never degrade an
            # image-bearing sample to text-only training silently.
            raise
        except ValueError as exc:
            if journal or "response_token_ids" in record or "generation_prompt_ids" in record:
                raise
            print(f"WARNING: skipping {source}: {exc}")
            skipped += 1
            continue
        if prepared is None:
            if journal:
                raise ValueError(f"{source}: exact row failed eligibility")
            skipped += 1
            continue
        if journal:
            journal.validate(record, prepared, tokenizer)
        payloads.append(prepared)
        if max_prompts not in (None, 0) and len(payloads) >= max_prompts:
            break
    if journal:
        journal.finish()
    print(
        f"VLM prompt preparation done: {len(payloads)} prepared, "
        f"{skipped} skipped ({path})"
    )
    return payloads


def _exact_qwen_turn(record, tokenizer, with_image, source):
    """Restore a single regenerated Qwen assistant turn without re-tokenizing it."""
    messages = _conversation_messages(record["conversations"], with_image=with_image, source=source)
    if [m["role"] for m in messages] != ["user", "assistant"]:
        raise ValueError(f"{source}: exact token records require one user and one assistant")
    kwargs = record.get("chat_template_kwargs", {})
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, **kwargs
    )
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    if prefix != record.get("generation_prompt_ids"):
        raise ValueError(f"{source}: regenerated prompt IDs do not match template")
    ids = record.get("response_token_ids")
    vocab_size = len(tokenizer)
    if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
        raise ValueError(f"{source}: invalid saved response IDs")
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if text != messages[-1]["content"] or tokenizer.eos_token_id in ids:
        raise ValueError(f"{source}: saved response IDs disagree with assistant content")
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kwargs)
    if full != prefix_text + text + "<|im_end|>\n":
        raise ValueError(f"{source}: unsupported Qwen turn boundary")
    suffix = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    if suffix[0] != tokenizer.eos_token_id:
        raise ValueError(f"{source}: unexpected Qwen end token")
    # The registered qwen template supervises its complete end-of-turn
    # delimiter, including the newline after the EOS token.
    return prefix + ids + suffix, [0] * len(prefix) + [1] * (len(ids) + len(suffix))


def _prepare_one_record(
    record: Dict[str, Any],
    *,
    source: str,
    tokenizer,
    processor,
    template,
    pad_token_id: int,
    merge_size: int,
    max_length: int,
    min_loss_tokens: int,
) -> Optional[Dict[str, Any]]:
    from .preprocessing import preprocess_conversations

    conversations = record.get("conversations")
    if not conversations:
        raise ValueError("record has no 'conversations' field")
    image_field = _extract_image_field(record, source=source)

    if "response_token_ids" in record or "generation_prompt_ids" in record:
        if getattr(template, "end_of_turn_token", None) != "<|im_end|>\n" or getattr(template, "ignore_token", None):
            raise ValueError(f"{source}: exact Qwen labels require the qwen loss-mask convention")
        collapsed, mask = _exact_qwen_turn(record, tokenizer, bool(image_field), source)
        if len(collapsed) > max_length:
            raise ValueError(f"{source}: exact token sequence exceeds max_length")
    else:
        text = _render_conversation_text(
            tokenizer, conversations, with_image=bool(image_field), source=source
        )
        parsed = preprocess_conversations(
            tokenizer, [text], template, max_length=max_length, is_preformatted=True
        )
        if not parsed["input_ids"]:
            return None
        collapsed = parsed["input_ids"][0][0].tolist()
        mask = parsed["loss_mask"][0][0].tolist()

    image_data = None
    if image_field:
        pil_image, image_data = _load_image(image_field, source=source)
        processor_output = processor.image_processor(
            images=[pil_image], return_tensors="pt"
        )
        image_grid_thw = _normalize_image_grid_thw(
            processor_output["image_grid_thw"], source=source
        )
        count = _image_token_count(
            [image_grid_thw], merge_size
        )
        ids, mask = _expand_image_region(
            collapsed,
            mask,
            pad_token_id=pad_token_id,
            count=count,
            source=source,
        )
        if len(ids) > max_length:
            # Truncating into an image region would desynchronize the client's
            # expanded ids from the server's own expansion; drop the sample.
            raise ValueError(
                f"expanded sequence length {len(ids)} exceeds max_length "
                f"{max_length} (image region cannot be truncated)"
            )
    else:
        ids = collapsed
        image_grid_thw = None

    ids = ids[:max_length]
    mask = mask[:max_length]
    if sum(mask) < min_loss_tokens:
        return None
    position_ids = build_qwen3_vl_position_ids(
        ids,
        image_pad_token_id=pad_token_id,
        image_grid_thw=image_grid_thw,
        merge_size=merge_size,
        source=source,
    )
    return {
        "payload": {
            "input_ids": ids,
            "loss_mask": mask,
            "position_ids": position_ids,
            "request_input_ids": collapsed,
            "image_data": image_data,
        }
    }


__all__ = [
    "ImageDataError",
    "build_mapped_vlm_prompt_payloads",
    "build_qwen3_vl_position_ids",
    "build_vlm_prompt_payloads",
]
