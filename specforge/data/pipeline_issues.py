"""Structured issue reporting for the VLM data pipeline."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional


class IssueCode(StrEnum):
    # Raw preparation
    RAW_JSON_PARSE_ERROR = "RAW_JSON_PARSE_ERROR"
    RAW_SCHEMA_MISMATCH = "RAW_SCHEMA_MISMATCH"
    RAW_MISSING_ID = "RAW_MISSING_ID"
    RAW_DUPLICATE_ID = "RAW_DUPLICATE_ID"
    RAW_MISSING_IMAGE_FIELD = "RAW_MISSING_IMAGE_FIELD"
    RAW_UNSAFE_IMAGE_PATH = "RAW_UNSAFE_IMAGE_PATH"
    RAW_MISSING_IMAGE = "RAW_MISSING_IMAGE"
    RAW_IMAGE_DECODE_ERROR = "RAW_IMAGE_DECODE_ERROR"
    RAW_INVALID_CONVERSATION = "RAW_INVALID_CONVERSATION"
    RAW_INVALID_ROLE = "RAW_INVALID_ROLE"
    RAW_ROLE_ORDER_ERROR = "RAW_ROLE_ORDER_ERROR"
    RAW_EMPTY_CONTENT = "RAW_EMPTY_CONTENT"

    # Regeneration
    REGEN_REQUEST_FAILED = "REGEN_REQUEST_FAILED"
    REGEN_TIMEOUT = "REGEN_TIMEOUT"
    REGEN_EMPTY_RESPONSE = "REGEN_EMPTY_RESPONSE"
    REGEN_LENGTH_TRUNCATED = "REGEN_LENGTH_TRUNCATED"
    REGEN_TURN_COUNT_MISMATCH = "REGEN_TURN_COUNT_MISMATCH"

    # Map and DFlash
    MAP_PROCESSOR_ERROR = "MAP_PROCESSOR_ERROR"
    MAP_ASSISTANT_COUNT_MISMATCH = "MAP_ASSISTANT_COUNT_MISMATCH"
    MAP_EMPTY_LOSS_MASK = "MAP_EMPTY_LOSS_MASK"
    ANCHOR_NO_VALID_POSITION = "ANCHOR_NO_VALID_POSITION"

    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"


@dataclass(frozen=True)
class PipelineIssue:
    stage: str
    error_code: IssueCode
    message: str
    retryable: bool = False
    action: str = "rejected"
    sample_id: Optional[str] = None
    source_file: Optional[str] = None
    source_row: Optional[int] = None
    context: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["error_code"] = self.error_code.value
        return value


class IssueReporter:
    """Write issues incrementally and maintain aggregate counts."""

    def __init__(self, issue_path: str | Path) -> None:
        self.issue_path = Path(issue_path)
        self.issue_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.issue_path.open("w", encoding="utf-8")
        self._counts: Counter[str] = Counter()
        self.accepted = 0
        self.rejected = 0
        self.retryable = 0

    def record(self, issue: PipelineIssue) -> None:
        self._handle.write(
            json.dumps(issue.to_dict(), ensure_ascii=False) + "\n"
        )
        self._counts[issue.error_code.value] += 1

        if issue.retryable:
            self.retryable += 1
        else:
            self.rejected += 1

    def record_accepted(self) -> None:
        self.accepted += 1

    def summary(self, *, input_rows: int) -> dict[str, Any]:
        return {
            "input_rows": input_rows,
            "accepted_rows": self.accepted,
            "rejected_rows": self.rejected,
            "retryable_rows": self.retryable,
            "issues_by_code": dict(sorted(self._counts.items())),
        }

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "IssueReporter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")

    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")

    temporary_path.replace(path)
