"""Rotary coordinates for DSpark; logical attention/cache indices stay unchanged."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def build_dspark_rope_positions(
    *,
    batch: "ScheduleBatch",
    logical_positions: torch.Tensor,
    token_counts: Sequence[int],
    use_mrope: bool,
) -> torch.Tensor:
    if not use_mrope:
        return logical_positions
    if logical_positions.ndim != 1:
        raise ValueError("logical_positions must be one-dimensional")
    counts = tuple(int(count) for count in token_counts)
    if any(count < 0 for count in counts) or sum(counts) != logical_positions.numel():
        raise ValueError("nonnegative token_counts must cover logical_positions")
    mm_inputs = batch.multimodal_inputs
    if mm_inputs is None:
        mm_inputs = [None] * len(counts)
    if len(mm_inputs) != len(counts):
        raise ValueError("multimodal_inputs and token_counts have different lengths")
    logical_positions = logical_positions.to(dtype=torch.int64)
    chunks = []
    offset = 0
    for count, mm_input in zip(counts, mm_inputs):
        pos = logical_positions[offset : offset + count]
        offset += count
        delta = getattr(mm_input, "mrope_position_delta", None)
        delta = torch.as_tensor(
            0 if delta is None else delta, device=pos.device, dtype=torch.int64
        )
        if delta.numel() != 1:
            raise ValueError("each request must have one mrope_position_delta")
        # Padding/sentinel positions are not real continuation tokens.
        continuation = torch.where(pos >= 0, pos + delta.reshape(()), pos)
        rope = continuation.unsqueeze(0).expand(3, -1)
        stored = getattr(mm_input, "mrope_positions", None)
        if stored is not None:
            stored = torch.as_tensor(stored, device=pos.device, dtype=torch.int64)
            if stored.ndim == 3 and stored.shape[0] == 1:
                stored = stored.squeeze(0)
            if stored.ndim != 2 or stored.shape[0] != 3:
                raise ValueError("stored mrope_positions must have shape [3, seq]")
            if stored.shape[1]:
                in_prompt = (pos >= 0) & (pos < stored.shape[1])
                prompt = stored[:, pos.clamp(0, stored.shape[1] - 1)]
                rope = torch.where(in_prompt.unsqueeze(0), prompt, rope)
        chunks.append(rope)
    if not chunks:
        return logical_positions.new_empty((3, 0))
    return torch.cat(chunks, dim=1).contiguous()
