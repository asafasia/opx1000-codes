"""Client for NVIDIA's Ising calibration vision-language NIM endpoint."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "nvidia/ising-calibration-1-35b-a3b"
DEFAULT_HOSTED_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_LOCAL_BASE_URL = "http://localhost:8000/v1"


class NvidiaIsingClient:
    """Small OpenAI-compatible chat-completions client.

    The same payload shape works for NVIDIA's hosted API and for a locally
    deployed NIM container. Use ``base_url`` to choose the target.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("NVIDIA_API_KEY")
        self.base_url = (base_url or os.getenv("NVIDIA_NIM_BASE_URL") or DEFAULT_HOSTED_BASE_URL).rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def local(cls, *, model: str = DEFAULT_MODEL, timeout_seconds: float = 120.0) -> "NvidiaIsingClient":
        """Create a client for a local NIM served on port 8000."""
        return cls(base_url=DEFAULT_LOCAL_BASE_URL, model=model, timeout_seconds=timeout_seconds)

    def chat_with_images(
        self,
        *,
        prompt: str,
        image_paths: list[Path],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Send a prompt and one or more PNG/JPEG figures to the model."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._image_data_url(path),
                    },
                }
            )

        return self.chat(
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Call the chat completions endpoint and return the decoded JSON response."""
        if self._uses_hosted_api() and not self.api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Set it in the same terminal before running "
                "AI review, or use --local / NVIDIA_NIM_BASE_URL for a local NIM."
            )
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"NVIDIA Ising calibration API failed: {error.code} {details}") from error

    def _uses_hosted_api(self) -> bool:
        return self.base_url.startswith(DEFAULT_HOSTED_BASE_URL)

    @staticmethod
    def _image_data_url(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            media_type = "image/jpeg"
        elif suffix == ".png":
            media_type = "image/png"
        else:
            raise ValueError(f"Unsupported image type for AI review: {path}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"
