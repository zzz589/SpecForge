#!/usr/bin/env python3
"""Pre-tokenize regenerated JSONL shards into reusable prompt-cache shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from transformers import AutoTokenizer

from specforge.data.prompt_builder import prepare_prompt_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-glob", default="*.jsonl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--chat-template", default="qwen")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--num-proc", type=int, default=32)
    parser.add_argument("--train-only-last-turn", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(item for item in path.glob(pattern) if item.is_file())
        if files:
            return files
    raise FileNotFoundError(f"no input files at {path} matching {pattern!r}")


def main() -> None:
    args = parse_args()
    if args.max_length <= 0 or args.num_proc <= 0:
        raise ValueError("--max-length and --num-proc must be positive")

    inputs = source_files(args.input_path, args.input_glob)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    manifest_entries = []

    for source in inputs:
        output = args.output_dir / f"{source.stem}.prompts.jsonl"
        metadata_path = output.with_suffix(".meta.json")
        metadata = {
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source),
            "model_path": str(Path(args.model_path).resolve()),
            "chat_template": args.chat_template,
            "max_length": args.max_length,
            "train_only_last_turn": args.train_only_last_turn,
            "format": "specforge-pretokenized-prompts-v1",
        }

        if output.exists() and not args.overwrite:
            if metadata_path.is_file():
                saved = json.loads(metadata_path.read_text(encoding="utf-8"))
                if all(saved.get(key) == value for key, value in metadata.items()):
                    print(f"reusing {output}")
                    manifest_entries.append(str(output.resolve()))
                    continue
            raise FileExistsError(
                f"{output} exists with different metadata; use --overwrite "
                "or choose another output directory"
            )

        rows = prepare_prompt_tasks(
            str(source),
            tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=False,
            train_only_last_turn=args.train_only_last_turn,
            cache_dir=None,
            cache_key=None,
            num_proc=args.num_proc,
        )

        count = 0
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output_dir, delete=False
        ) as handle:
            temporary = Path(handle.name)
            try:
                for row in rows:
                    json.dump(
                        row["payload"],
                        handle,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    handle.write("\n")
                    count += 1
                handle.flush()
                os.fsync(handle.fileno())
                os.replace(temporary, output)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise

        metadata["mapped_rows"] = count
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_entries.append(str(output.resolve()))
        print(f"mapped {source.name}: {count} rows -> {output}")

    manifest = args.output_dir / "prompts.manifest"
    manifest.write_text("\n".join(manifest_entries) + "\n", encoding="utf-8")
    print(f"wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
