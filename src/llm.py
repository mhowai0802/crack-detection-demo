"""Thin wrapper around the HKBU GenAI REST chat completions endpoint.

Credentials and defaults are read from a local ``.env`` file (loaded via
``python-dotenv``) so the secret never needs to be hardcoded. The public
surface is a single :func:`chat` function that takes a system + user
message and returns the assistant's text reply.

Environment variables
---------------------
HKBU_API_KEY
    Required. API key issued by the HKBU GenAI portal.
HKBU_BASE_URL
    Optional. Defaults to ``https://genai.hkbu.edu.hk/api/v0/rest``.
HKBU_MODEL
    Optional. Default deployment / model name. Defaults to ``gpt-4.1-mini``.
HKBU_API_VERSION
    Optional. Azure OpenAI API version. Defaults to ``2024-12-01-preview``.
"""

from __future__ import annotations

import os
from typing import List, Mapping, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://genai.hkbu.edu.hk/api/v0/rest"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_API_VERSION = "2024-12-01-preview"


class LLMConfigError(RuntimeError):
    """Raised when required credentials are missing."""


class LLMRequestError(RuntimeError):
    """Raised when the HKBU endpoint returns a non-200 response."""


def _require_api_key() -> str:
    api_key = os.getenv("HKBU_API_KEY")
    if not api_key:
        raise LLMConfigError(
            "HKBU_API_KEY is not set. Copy .env.example to .env and fill in "
            "your HKBU GenAI API key."
        )
    return api_key


def _post_chat(
    messages: List[Mapping[str, str]],
    *,
    model: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> str:
    """Shared HTTP + error-handling path for chat completions."""
    api_key = _require_api_key()
    base_url = os.getenv("HKBU_BASE_URL", DEFAULT_BASE_URL)
    model_name = model or os.getenv("HKBU_MODEL", DEFAULT_MODEL)
    api_version = os.getenv("HKBU_API_VERSION", DEFAULT_API_VERSION)

    url = (
        f"{base_url}/deployments/{model_name}/chat/completions"
        f"?api-version={api_version}"
    )
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload = {
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": False,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if response.status_code != 200:
        raise LLMRequestError(
            f"HKBU GenAI request failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMRequestError(f"Unexpected response payload: {data!r}") from exc


def chat(
    system: str,
    user: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 1.0,
    timeout: float = 30.0,
) -> str:
    """Send a single-turn chat completion and return the assistant text.

    Args:
        system: System-role instructions for the model.
        user: User-role message.
        model: Deployment name. Falls back to ``HKBU_MODEL`` or ``gpt-4.1-mini``.
        max_tokens: Response length cap.
        temperature: Sampling temperature in ``[0, 2]``.
        top_p: Nucleus sampling top-p.
        timeout: HTTP timeout in seconds.

    Returns:
        The assistant message content as a string.

    Raises:
        LLMConfigError: If ``HKBU_API_KEY`` is not set.
        LLMRequestError: If the API returns a non-200 response or a
            payload without a usable assistant message.
    """
    return _post_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
    )


def chat_messages(
    messages: List[Mapping[str, str]],
    *,
    model: Optional[str] = None,
    max_tokens: int = 400,
    temperature: float = 0.5,
    top_p: float = 1.0,
    timeout: float = 30.0,
) -> str:
    """Send a multi-turn chat completion and return the assistant text.

    Accepts a list of ``{"role": ..., "content": ...}`` dicts so callers
    can pass system + assistant + user turns (e.g. a chat box preserving
    history). Auth / URL / error handling are shared with :func:`chat`.

    Args:
        messages: Full conversation history in OpenAI-style chat format.
        model: Deployment name. Falls back to ``HKBU_MODEL`` or ``gpt-4.1-mini``.
        max_tokens: Response length cap.
        temperature: Sampling temperature in ``[0, 2]``.
        top_p: Nucleus sampling top-p.
        timeout: HTTP timeout in seconds.

    Returns:
        The assistant message content as a string.

    Raises:
        LLMConfigError: If ``HKBU_API_KEY`` is not set.
        LLMRequestError: If the API returns a non-200 response or a
            payload without a usable assistant message.
    """
    if not messages:
        raise ValueError("messages must be a non-empty list.")
    return _post_chat(
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
    )
