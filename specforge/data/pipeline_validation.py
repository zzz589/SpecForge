"""Reusable validation helpers for the VLM data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PIL import Image, UnidentifiedImageError


try:
    from .pipeline_issues import IssueCode
except ImportError:
    # CPU-only standalone import used by preparation scripts.
    from pipeline_issues import IssueCode


ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
}


class SampleRejected(ValueError):
    def __init__(
        self,
        error_code: IssueCode,
        message: str,
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}


def validate_sample_id(value: Any) -> str:
    if value is None:
        raise SampleRejected(
            IssueCode.RAW_MISSING_ID,
            "Sample is missing an id",
        )

    sample_id = str(value).strip()
    if not sample_id:
        raise SampleRejected(
            IssueCode.RAW_MISSING_ID,
            "Sample id is empty",
        )

    return sample_id


def convert_llava_conversations(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise SampleRejected(
            IssueCode.RAW_INVALID_CONVERSATION,
            "conversations must be a list",
            context={
                "actual_type": type(messages).__name__,
            },
        )

    if len(messages) < 2:
        raise SampleRejected(
            IssueCode.RAW_INVALID_CONVERSATION,
            "conversation must contain at least one user/assistant pair",
            context={
                "message_count": len(messages),
            },
        )

    if len(messages) % 2 != 0:
        raise SampleRejected(
            IssueCode.RAW_ROLE_ORDER_ERROR,
            "conversation must end with an assistant message",
            context={
                "message_count": len(messages),
            },
        )

    converted = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise SampleRejected(
                IssueCode.RAW_SCHEMA_MISMATCH,
                "conversation message must be an object",
                context={
                    "message_index": index,
                    "actual_type": type(message).__name__,
                },
            )

        source_role = message.get("from")
        content = message.get("value")

        if source_role not in ROLE_MAPPING:
            raise SampleRejected(
                IssueCode.RAW_INVALID_ROLE,
                f"Unsupported conversation role: {source_role!r}",
                context={
                    "message_index": index,
                    "role": source_role,
                },
            )

        if not isinstance(content, str):
            raise SampleRejected(
                IssueCode.RAW_SCHEMA_MISMATCH,
                "conversation content must be a string",
                context={
                    "message_index": index,
                    "actual_type": type(content).__name__,
                },
            )

        # Images are represented by the row-level image field. Remove all
        # textual markers so the processor attaches the image exactly once.
        content = content.replace("<image>", "").strip()

        if not content:
            raise SampleRejected(
                IssueCode.RAW_EMPTY_CONTENT,
                "conversation content is empty after removing image markers",
                context={
                    "message_index": index,
                },
            )

        role = ROLE_MAPPING[source_role]
        expected_role = "user" if index % 2 == 0 else "assistant"

        if role != expected_role:
            raise SampleRejected(
                IssueCode.RAW_ROLE_ORDER_ERROR,
                f"Expected role {expected_role!r}, got {role!r}",
                context={
                    "message_index": index,
                    "expected_role": expected_role,
                    "actual_role": role,
                },
            )

        converted.append({
            "role": role,
            "content": content,
        })

    return converted


def resolve_and_validate_image(
    *,
    image_root: Path,
    image_value: Any,
    decode_image: bool,
) -> Path:
    if not isinstance(image_value, str) or not image_value.strip():
        raise SampleRejected(
            IssueCode.RAW_MISSING_IMAGE_FIELD,
            "Sample is missing a non-empty image path",
        )

    image_value = image_value.strip()

    if Path(image_value).is_absolute():
        image_path = Path(image_value).expanduser().absolute()
    else:
        image_path = (image_root / image_value).absolute()

    resolved_root = image_root.resolve()
    resolved_image = image_path.resolve()

    try:
        resolved_image.relative_to(resolved_root)
    except ValueError as exc:
        raise SampleRejected(
            IssueCode.RAW_UNSAFE_IMAGE_PATH,
            "Image path resolves outside image_root",
            context={
                "image": image_value,
                "resolved_image": str(resolved_image),
                "image_root": str(resolved_root),
            },
        ) from exc

    if not image_path.is_file():
        raise SampleRejected(
            IssueCode.RAW_MISSING_IMAGE,
            "Image file does not exist",
            context={
                "image": str(image_path),
            },
        )

    if decode_image:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise SampleRejected(
                IssueCode.RAW_IMAGE_DECODE_ERROR,
                f"Image cannot be decoded: {exc}",
                context={
                    "image": str(image_path),
                    "exception_type": type(exc).__name__,
                },
            ) from exc

    return image_path


def normalize_llava_sample(
    sample: Any,
    *,
    image_root: Path,
    image_prefix: Optional[str],
    source_name: Optional[str],
    decode_image: bool = True,
) -> Optional[dict[str, Any]]:
    """Return a normalized row, or None for intentional prefix exclusion."""

    if not isinstance(sample, dict):
        raise SampleRejected(
            IssueCode.RAW_SCHEMA_MISMATCH,
            "Sample must be an object",
            context={
                "actual_type": type(sample).__name__,
            },
        )

    image_value = sample.get("image")

    if (
        image_prefix is not None
        and isinstance(image_value, str)
        and not image_value.startswith(image_prefix)
    ):
        return None

    sample_id = validate_sample_id(sample.get("id"))

    image_path = resolve_and_validate_image(
        image_root=image_root,
        image_value=image_value,
        decode_image=decode_image,
    )

    conversations = convert_llava_conversations(
        sample.get("conversations")
    )

    dataset_source = (
        source_name
        or sample.get("_dataset_source")
        or sample.get("source")
        or "unknown"
    )

    return {
        "id": sample_id,
        "_dataset_source": str(dataset_source),
        "image": str(image_path),
        "conversations": conversations,
    }
