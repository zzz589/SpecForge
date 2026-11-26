"""Loss-mask predicates shared by online and offline data paths."""

from collections.abc import Sequence
from typing import Any


def has_consecutive_supervised_tokens(loss_mask: Any) -> bool:
    """Return whether one sample contains adjacent supervised tokens."""

    values = loss_mask.tolist() if hasattr(loss_mask, "tolist") else list(loss_mask)
    if values and isinstance(values[0], Sequence):
        if len(values) != 1:
            raise ValueError("expected one loss-mask row")
        values = list(values[0])
    return any(
        bool(current) and bool(following)
        for current, following in zip(values, values[1:])
    )


def trim_loss_mask_after_eos(input_ids, loss_mask, eos_token_id):
    """Keep the first EOS in each supervised run and remove its suffix.

    Accept a list row or tensor rows. The output preserves the mask's type,
    shape, dtype and device, allowing preprocessing and training to agree.
    """
    if eos_token_id is None:
        return loss_mask
    import torch

    is_tensor = isinstance(loss_mask, torch.Tensor)
    mask = torch.as_tensor(loss_mask)
    ids = torch.as_tensor(input_ids, device=mask.device)
    if ids.shape != mask.shape or mask.ndim not in (1, 2):
        raise ValueError("input_ids and loss_mask must have matching [S] or [B,S] shapes")
    if mask.shape[-1] == 0:
        return loss_mask
    eos = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
    terminal = torch.zeros_like(ids, dtype=torch.bool)
    for token in eos:
        terminal |= ids == token
    valid = mask > 0.5
    index = torch.arange(mask.shape[-1], device=mask.device)
    last_gap = torch.where(~valid, index, -1).cummax(dim=-1).values
    last_eos = torch.where(terminal & valid, index, -1).cummax(dim=-1).values
    previous_eos = torch.cat([torch.full_like(last_eos[..., :1], -1), last_eos[..., :-1]], dim=-1)
    result = mask * (previous_eos <= last_gap).to(mask.dtype)
    return result if is_tensor else result.tolist()


__all__ = ["has_consecutive_supervised_tokens", "trim_loss_mask_after_eos"]
