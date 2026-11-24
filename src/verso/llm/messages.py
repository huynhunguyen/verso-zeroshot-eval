"""Chat messages in the OpenAI format, with images embedded as base64 data URIs (accepted by vLLM for Qwen3-VL)."""

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

from ..core.errors import VersoError

# An image as a file path, or as (bytes, MIME type).
ImageSource: TypeAlias = str | Path | tuple[bytes, str]

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


def image_part(image: ImageSource) -> dict[str, Any]:
    """Content part for an image given as a path or as (bytes, mime)."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime = _MIME.get(path.suffix.lower())
        if mime is None:
            raise VersoError(f"unsupported image type: {path.name}")
        data = path.read_bytes()
    else:
        data, mime = image
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}}


def system_message(text: str) -> dict[str, Any]:
    """A system turn."""
    return {"role": "system", "content": text}


def assistant_message(text: str) -> dict[str, Any]:
    """A turn of the model: a worked example's answer, or an earlier reply sent back for repair."""
    return {"role": "assistant", "content": text}


def user_message(text: str, images: Sequence[ImageSource] = ()) -> dict[str, Any]:
    """A user turn; images come before the text, as Qwen-VL prompts usually place them."""
    if not images:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [image_part(i) for i in images] + [{"type": "text", "text": text}]}
