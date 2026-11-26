"""Resolve DFlash offline eligibility before sharding or resume positioning."""

import logging
from functools import partial

from specforge.data.loss_mask import has_consecutive_supervised_tokens, trim_loss_mask_after_eos

logger = logging.getLogger(__name__)


def prepare_offline_data(refs, *, store, model, max_len, normalizer):
    # Production supplies the composite training model, before distributed wrapping.
    draft = getattr(model, "draft_model", model)
    eos_token_id = getattr(getattr(draft, "config", None), "eos_token_id", None)
    refs = list(refs)
    if any(not ref.feature_store_uri.startswith("file://") for ref in refs):
        if not getattr(store, "retain_on_release", False):
            raise ValueError("DFlash offline inspection requires a re-iterable feature store with retain_on_release=True")

    def eligible_indices():
        selected = []
        logger.info("Checking DFlash generation anchors in %d offline samples", len(refs))
        for index, ref in enumerate(refs):
            tensors, handle = store.get(ref, device="cpu", names=["input_ids", "loss_mask"])
            try:
                ids = tensors["input_ids"]
                mask = tensors["loss_mask"]
                if ids.ndim != 1 or mask.ndim != 1 or ids.shape != mask.shape:
                    raise ValueError(f"{ref.sample_id}: offline input_ids/loss_mask must be matching [S] tensors")
                mask = trim_loss_mask_after_eos(ids[:max_len], mask[:max_len], eos_token_id)
                if has_consecutive_supervised_tokens(mask):
                    selected.append(index)
            finally:
                store.release(handle, reason="offline-eligibility-inspection")
            if (index + 1) % 1000 == 0:
                logger.info("Checked %d/%d offline samples; %d eligible", index + 1, len(refs), len(selected))
        logger.info("DFlash offline eligibility: %d -> %d samples", len(refs), len(selected))
        if not selected:
            raise ValueError("DFlash offline dataset has no eligible generation anchors after truncation/EOS filtering")
        return selected

    # Reuse the rank-0 value/error broadcast used by provenance resolution.
    from specforge.training.provenance import _rank_zero_value

    indices = _rank_zero_value(eligible_indices, operation="DFlash offline eligibility")
    # Preserve algorithm-owned fields such as DSpark teacher hidden states.
    transform = partial(normalizer, generation_eos_token_id=eos_token_id)
    return [refs[index] for index in indices], transform
