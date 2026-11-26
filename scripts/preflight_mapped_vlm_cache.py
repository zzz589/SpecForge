#!/usr/bin/env python3
"""Read-only preflight for the mapped VLM online-capture fast path."""

from __future__ import annotations

import argparse
import json
import os
import time

from transformers import AutoTokenizer

from specforge.data.vlm_preprocessing import build_mapped_vlm_prompt_payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--chat-template", default="qwen")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-loss-tokens", type=int, default=2)
    args = parser.parse_args()

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompts = build_mapped_vlm_prompt_payloads(
        args.jsonl,
        args.manifest,
        tokenizer,
        target_model_path=args.target_model,
        chat_template=args.chat_template,
        max_length=args.max_length,
        min_loss_tokens=args.min_loss_tokens,
    )
    missing_images = []
    min_sequence = None
    max_sequence = 0
    min_loss = None
    max_loss = 0
    for index in range(len(prompts)):
        payload = prompts[index]["payload"]
        sequence_tokens = len(payload["input_ids"])
        loss_tokens = sum(int(item) for item in payload["loss_mask"])
        min_sequence = sequence_tokens if min_sequence is None else min(min_sequence, sequence_tokens)
        max_sequence = max(max_sequence, sequence_tokens)
        min_loss = loss_tokens if min_loss is None else min(min_loss, loss_tokens)
        max_loss = max(max_loss, loss_tokens)
        image_path = payload.get("image_path")
        if image_path and not image_path.startswith("data:") and not os.path.isfile(image_path):
            missing_images.append({"index": index, "path": image_path})
            if len(missing_images) >= 20:
                break
    report = {
        "status": "PASS" if not missing_images else "FAIL",
        "rows": len(prompts),
        "sequence_tokens_min": min_sequence,
        "sequence_tokens_max": max_sequence,
        "loss_tokens_min": min_loss,
        "loss_tokens_max": max_loss,
        "missing_image_examples": missing_images,
        "dataset_map_called": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing_images:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
