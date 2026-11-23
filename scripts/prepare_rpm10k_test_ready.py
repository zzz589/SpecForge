#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-pixels", type=int, default=802816)
    args = parser.parse_args()

    output = Path(args.output)
    image_dir = Path(args.image_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    resized_count = 0
    with open(args.input, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            image_path = Path(row["image"])
            with Image.open(image_path) as opened:
                opened.load()
                width, height = opened.size
                if width * height > args.max_pixels:
                    scale = math.sqrt(args.max_pixels / float(width * height))
                    new_width = max(1, math.floor(width * scale))
                    new_height = max(1, math.floor(height * scale))
                    while new_width * new_height > args.max_pixels:
                        if new_width >= new_height:
                            new_width -= 1
                        else:
                            new_height -= 1
                    resized = opened.convert("RGB").resize(
                        (new_width, new_height), Image.Resampling.LANCZOS
                    )
                    destination = image_dir / image_path.name
                    temporary = destination.with_suffix(destination.suffix + ".tmp")
                    resized.save(temporary, format="JPEG", quality=95, subsampling=0)
                    os.replace(temporary, destination)
                    row["image"] = str(destination)
                    resized_count += 1
            if not row.get("reference_answer"):
                raise ValueError(f"line {line_number}: missing reference_answer")
            rows.append(row)

    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary_output, output)
    report = {
        "status": "pass",
        "input": str(Path(args.input).resolve()),
        "output": str(output.resolve()),
        "rows": len(rows),
        "resized_images": resized_count,
        "max_pixels": args.max_pixels,
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
