"""Checks for single-image Qwen training from a trusted generation journal."""
import base64
import hashlib
import json
import math

from .prompt_builder import _iter_records


class ExactJournal:
    def __init__(self, path):
        self.rows = {}
        self.seen = set()
        for line, row in _iter_records(path):
            key = row.get("id")
            if not isinstance(key, str) or not key or key in self.rows:
                raise ValueError(f"{path}:{line}: missing or duplicate generation id")
            self.rows[key] = row
        if not self.rows:
            raise ValueError(f"{path}: empty generation journal")

    def validate(self, row, prepared, tokenizer):
        key = row.get("id")
        if not isinstance(key, str) or key not in self.rows or key in self.seen:
            raise ValueError(f"unknown, missing or duplicate training id {key!r}")
        saved = self.rows[key]
        expected_message = {"role": "user", "content": [
            {"type": "text", "text": saved["prompt"]}, {"type": "image"},
        ]}
        prompt = tokenizer.apply_chat_template([expected_message], tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        if row.get("generation_prompt_ids") != prefix or row.get("response_token_ids") != saved["response_token_ids"]:
            raise ValueError(f"{key}: token IDs missing or different from generation")
        uri = prepared["payload"]["image_data"]
        if not uri or hashlib.sha256(base64.b64decode(uri.split(",", 1)[1], validate=True)).hexdigest() != saved["image_sha256"]:
            raise ValueError(f"{key}: captured image differs from generation")
        if "visual_score_sha256" in saved or "visual_scores" in row:
            scores = row.get("visual_scores")
            response = saved["response_token_ids"]
            if not isinstance(scores, list) or len(scores) != len(response) or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in scores
            ):
                raise ValueError(f"{key}: expected one finite visual score per generated token")
            start = prepared["payload"]["loss_mask"].index(1)
            if score_digest(key, saved["image_sha256"], start, response, scores) != saved.get("visual_score_sha256"):
                raise ValueError(f"{key}: visual scores differ from trusted causal reference")
        self.seen.add(key)

    def finish(self):
        if len(self.seen) != len(self.rows):
            raise ValueError(f"training omitted {len(self.rows) - len(self.seen)} generation records")


def score_digest(key, image_sha256, response_start, response, scores):
    """Bind score values, including their order, to the generated label sequence."""
    content = ["causal-response-v1", key, image_sha256, response_start - 1, response, scores]
    return hashlib.sha256(json.dumps(content, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def align_response_scores(with_ids, without_ids, response, with_start, without_start,
                          scores_with, scores_without, *, turn_suffix=(151645, 198)):
    """Gather two causal predictor traces by label. Traces have one scalar per logit row.

    Both traces must use the same score definition. The returned pairs exclude
    template-appended EOS and newline. No weighting is applied to training.
    """
    result = []
    for ids, start, scores in ((with_ids, with_start, scores_with),
                               (without_ids, without_start, scores_without)):
        if not response or start < 1 or list(ids[start:]) != list(response) + list(turn_suffix):
            raise ValueError("response token alignment mismatch")
        if len(scores) != len(ids):
            raise ValueError("expected one score per causal logit position")
        result.append(scores[start-1:start-1+len(response)])
    return tuple(result)


def check_response_scores(observed, expected):
    """Compare downstream values to an independently retained causal reference."""
    import torch
    observed, expected = torch.as_tensor(observed), torch.as_tensor(expected)
    if observed.shape != expected.shape or not torch.isfinite(expected).all() or not torch.equal(observed, expected):
        raise ValueError("visual score values disagree with causal reference")
