"""Text chat for Q&A over the search index.

Vision captioning during ingest still uses Ollama directly (``ingest.py``, etc.).

Backends (``PHOTO_INDEX_LLM_BACKEND``):
  * ``ollama`` (default) — ``ollama.chat``
  * ``openai`` — OpenAI-compatible HTTP (LM Studio, llama.cpp ``llama-server``)

LM Studio: enable **Local Server**, load a model, set e.g.::

  export PHOTO_INDEX_LLM_BACKEND=openai
  export PHOTO_INDEX_LLM_BASE_URL=http://127.0.0.1:1234/v1
  export PHOTO_INDEX_QA_MODEL=qwen2.5-3b-instruct
  export PHOTO_INDEX_QA_MODEL_SMALL=qwen2.5-3b-instruct

When only one model is loaded, LM Studio usually accepts any model name string.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_DEFAULT_OPENAI_BASE = "http://127.0.0.1:1234/v1"
_HTTP_TIMEOUT = float(os.environ.get("PHOTO_INDEX_LLM_TIMEOUT", "600"))

# Auto-shrink: when a server rejects a prompt for exceeding its context window,
# retry with the bulky middle of the longest message trimmed (keeps the leading
# instructions and the trailing user question).
_MAX_SHRINK_ATTEMPTS = int(os.environ.get("PHOTO_INDEX_LLM_SHRINK_ATTEMPTS", "5"))
_CHARS_PER_TOKEN_EST = 3.5
_INPUT_CONTEXT_FRACTION = 0.6  # leave room for the model's reply
_MIN_SHRINK_CHARS = 600
_NCTX_RE = re.compile(r"n_ctx[\"']?\s*[:=]\s*(\d+)")
_CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context window",
    "n_ctx",
    "tokens to keep",
    "too many tokens",
    "exceeds context",
    "maximum context",
)


class ContextOverflowError(RuntimeError):
    """Raised when the server rejects a request for exceeding its context window."""


def _looks_like_context_overflow(message: str) -> bool:
    m = (message or "").lower()
    return any(k in m for k in _CONTEXT_OVERFLOW_MARKERS)


def llm_backend() -> str:
    return os.environ.get("PHOTO_INDEX_LLM_BACKEND", "ollama").strip().lower()


def openai_base_url() -> str:
    return os.environ.get("PHOTO_INDEX_LLM_BASE_URL", _DEFAULT_OPENAI_BASE).rstrip("/")


def chat_completion_text(*, model: str, messages: list[dict[str, str]]) -> str:
    """Run a chat completion; return assistant text."""
    backend = llm_backend()
    if backend == "ollama":
        from ollama import chat

        response = chat(model=model, messages=messages)
        return (response.message.content or "").strip()
    if backend in ("openai", "lmstudio", "llama.cpp", "llamacpp"):
        return _openai_chat_with_shrink(model=model, messages=messages)
    raise ValueError(
        f"Unknown PHOTO_INDEX_LLM_BACKEND={backend!r} "
        "(use 'ollama' or 'openai')"
    )


def chat_user_prompt(*, model: str, prompt: str) -> str:
    return chat_completion_text(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )


def list_llm_models() -> list[str]:
    backend = llm_backend()
    if backend == "ollama":
        return _ollama_model_names()
    if backend in ("openai", "lmstudio", "llama.cpp", "llamacpp"):
        return _openai_compatible_model_ids()
    return []


def _ollama_model_names() -> list[str]:
    try:
        out = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception:
        return []
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return []
    models: list[str] = []
    for ln in lines[1:]:
        parts = ln.split()
        if parts:
            models.append(parts[0])
    return models


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            pass
        detail = body or e.reason
        if e.code == 400 and _looks_like_context_overflow(detail):
            raise ContextOverflowError(f"HTTP {e.code} from {url}: {detail}") from e
        raise RuntimeError(
            f"HTTP {e.code} from {url}: {detail}"
        ) from e
    except URLError as e:
        raise RuntimeError(
            f"Could not reach LLM server at {url} ({e.reason}). "
            "Is LM Studio Local Server running with a model loaded?"
        ) from e
    if not raw.strip():
        return {}
    return json.loads(raw)


def _middle_truncate(text: str, max_chars: int) -> str:
    """Trim the middle of ``text`` so the start and end survive."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    trimmed = len(text) - max_chars
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    head = text[:keep_head]
    tail = text[len(text) - keep_tail:]
    return f"{head}\n…[trimmed {trimmed} chars to fit model context]…\n{tail}"


def _shrink_messages(messages: list[dict[str, str]], total_char_budget: int) -> list[dict[str, str]]:
    """Trim the single longest message so the combined content fits the budget."""
    if not messages:
        return messages
    idx = max(
        range(len(messages)),
        key=lambda i: len(str(messages[i].get("content") or "")),
    )
    others = sum(
        len(str(m.get("content") or "")) for i, m in enumerate(messages) if i != idx
    )
    longest_budget = max(_MIN_SHRINK_CHARS, total_char_budget - others)
    new_msgs = [dict(m) for m in messages]
    new_msgs[idx]["content"] = _middle_truncate(
        str(new_msgs[idx].get("content") or ""), longest_budget
    )
    return new_msgs


def _openai_chat_with_shrink(*, model: str, messages: list[dict[str, str]]) -> str:
    attempt_messages = messages
    n_ctx_hint: int | None = None
    last_error: ContextOverflowError | None = None
    for _ in range(max(1, _MAX_SHRINK_ATTEMPTS)):
        try:
            return _openai_compatible_chat(model=model, messages=attempt_messages)
        except ContextOverflowError as e:
            last_error = e
            m = _NCTX_RE.search(str(e))
            if m:
                n_ctx_hint = int(m.group(1))
            total_chars = sum(
                len(str(mm.get("content") or "")) for mm in attempt_messages
            )
            if n_ctx_hint and n_ctx_hint > 0:
                target = int(n_ctx_hint * _CHARS_PER_TOKEN_EST * _INPUT_CONTEXT_FRACTION)
            else:
                target = int(total_chars * 0.5)
            target = max(target, _MIN_SHRINK_CHARS)
            if target >= total_chars:
                target = int(total_chars * 0.5)
            shrunk = _shrink_messages(attempt_messages, target)
            shrunk_chars = sum(
                len(str(mm.get("content") or "")) for mm in shrunk
            )
            if shrunk_chars >= total_chars:
                break  # can't shrink further
            attempt_messages = shrunk
    if last_error is not None:
        raise last_error
    return _openai_compatible_chat(model=model, messages=attempt_messages)


def _openai_compatible_chat(*, model: str, messages: list[dict[str, str]]) -> str:
    url = f"{openai_base_url()}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    data = _http_json("POST", url, payload)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response from {url}: not a JSON object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"No choices in LLM response from {url}: {data!r}")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"Invalid choice object from {url}")
    msg = first.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if content is not None:
            return str(content).strip()
    text = first.get("text")
    if text is not None:
        return str(text).strip()
    raise RuntimeError(f"Could not parse assistant text from {url}: {first!r}")


def _openai_compatible_model_ids() -> list[str]:
    url = f"{openai_base_url()}/models"
    try:
        data = _http_json("GET", url)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    items = data.get("data")
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            mid = str(item.get("id") or "").strip()
            if mid:
                out.append(mid)
    return out
