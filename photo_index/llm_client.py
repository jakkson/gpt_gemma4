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
_HTTP_TIMEOUT_BIG = float(os.environ.get("PHOTO_INDEX_LLM_TIMEOUT_BIG", "1800"))
_READ_IDLE_TIMEOUT = float(os.environ.get("PHOTO_INDEX_LLM_READ_TIMEOUT", "300"))
# Reasoning models (e.g. Qwen3.5-A3B) spend 400-600+ tokens "thinking" before the
# answer lands in `content`. The big-model budget must clear reasoning AND the answer,
# or `content` comes back empty (finish_reason=length mid-thought).
_DEFAULT_MAX_TOKENS = int(os.environ.get("PHOTO_INDEX_LLM_MAX_TOKENS", "2048"))
# Reasoning over many records can spend several thousand tokens "thinking" before
# the answer lands in `content`. Too low and `content` comes back empty mid-thought.
_DEFAULT_MAX_TOKENS_BIG = int(os.environ.get("PHOTO_INDEX_LLM_MAX_TOKENS_BIG", "8192"))

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


def is_big_qa_model(model: str) -> bool:
    """Heuristic: large/slow models that need longer timeouts and smaller prompts."""
    m = (model or "").lower()
    return any(tag in m for tag in ("32b", "26b", "27b", "30b", "34b", "35b", "70b"))


def inference_opts_for_model(model: str) -> dict[str, Any]:
    """Timeout / generation / prompt caps tuned per model size."""
    if is_big_qa_model(model):
        return {
            "timeout": _HTTP_TIMEOUT_BIG,
            "max_tokens": _DEFAULT_MAX_TOKENS_BIG,
            "stream": True,
        }
    return {
        "timeout": _HTTP_TIMEOUT,
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "stream": False,
    }


def _apply_no_think(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """When PHOTO_INDEX_NO_THINK is truthy, append Qwen3's ``/no_think`` switch so
    a HYBRID thinking model (e.g. dense Qwen3-32B) answers directly instead of
    emitting chain-of-thought that overflows the large RAG prompt. No-op for the
    non-thinking instruct models (30B-A3B, 235B-A22B) — only set the env for a
    thinking-capable model. Applied here so every answer path is covered at once.
    """
    flag = os.environ.get("PHOTO_INDEX_NO_THINK", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return messages
    out = [dict(m) for m in messages]
    tgt = next((m for m in out if m.get("role") == "system"), None)
    if tgt is None:
        tgt = next((m for m in reversed(out) if m.get("role") == "user"), None)
    if tgt is not None and "/no_think" not in str(tgt.get("content", "")):
        tgt["content"] = f"{str(tgt.get('content', '')).rstrip()}\n\n/no_think"
    return out


def chat_completion_text(
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout: float | None = None,
    max_tokens: int | None = None,
    stream: bool | None = None,
) -> str:
    """Run a chat completion; return assistant text."""
    messages = _apply_no_think(messages)
    opts = inference_opts_for_model(model)
    if timeout is None:
        timeout = float(opts["timeout"])
    if max_tokens is None:
        max_tokens = int(opts["max_tokens"])
    if stream is None:
        stream = bool(opts["stream"])
    backend = llm_backend()
    if backend == "ollama":
        from ollama import chat

        # Ollama defaults to a small context (2K-4K) and SILENTLY truncates longer
        # prompts, keeping only the tail — which drops most retrieved RAG records.
        # Pin an explicit context window so the model actually sees the records.
        num_ctx = int(os.environ.get("PHOTO_INDEX_LLM_NUM_CTX", "16384"))
        response = chat(
            model=model,
            messages=messages,
            options={
                "num_ctx": num_ctx,
                "temperature": float(os.environ.get("PHOTO_INDEX_LLM_TEMPERATURE", "0.2")),
            },
        )
        return (response.message.content or "").strip()
    if backend in ("openai", "lmstudio", "llama.cpp", "llamacpp"):
        return _openai_chat_with_shrink(
            model=model,
            messages=messages,
            timeout=timeout,
            max_tokens=max_tokens,
            stream=stream,
        )
    raise ValueError(
        f"Unknown PHOTO_INDEX_LLM_BACKEND={backend!r} "
        "(use 'ollama' or 'openai')"
    )


def chat_user_prompt(
    *,
    model: str,
    prompt: str,
    timeout: float | None = None,
    max_tokens: int | None = None,
    stream: bool | None = None,
) -> str:
    return chat_completion_text(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
        max_tokens=max_tokens,
        stream=stream,
    )


_DEFAULT_EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
# nomic-embed-text was trained with task prefixes; using them materially improves
# retrieval quality. Documents and queries get different prefixes.
_EMBED_DOC_PREFIX = "search_document: "
_EMBED_QUERY_PREFIX = "search_query: "
_EMBED_BATCH = int(os.environ.get("PHOTO_INDEX_EMBED_BATCH", "64"))
_EMBED_CHAR_CAP = int(os.environ.get("PHOTO_INDEX_EMBED_CHAR_CAP", "2000"))


def embed_model() -> str:
    return os.environ.get("PHOTO_INDEX_EMBED_MODEL", _DEFAULT_EMBED_MODEL)


def embed_texts(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    """Return embedding vectors for ``texts`` via the OpenAI-compatible server.

    Applies nomic task prefixes and caps each text length. Batches requests.
    """
    if not texts:
        return []
    prefix = _EMBED_QUERY_PREFIX if is_query else _EMBED_DOC_PREFIX
    prepared = [prefix + (t or "")[:_EMBED_CHAR_CAP] for t in texts]
    backend = llm_backend()
    if backend == "ollama":
        from ollama import embed as _ollama_embed

        out: list[list[float]] = []
        for t in prepared:
            r = _ollama_embed(model=embed_model(), input=t)
            out.append(list(r["embeddings"][0]))
        return out
    # OpenAI-compatible (LM Studio / llama.cpp)
    url = f"{openai_base_url()}/embeddings"
    out2: list[list[float]] = []
    for i in range(0, len(prepared), _EMBED_BATCH):
        chunk = prepared[i : i + _EMBED_BATCH]
        data = _http_json("POST", url, {"model": embed_model(), "input": chunk}, timeout=_HTTP_TIMEOUT)
        if not isinstance(data, dict) or "data" not in data:
            raise RuntimeError(f"Unexpected embeddings response from {url}: {str(data)[:200]}")
        items = sorted(data["data"], key=lambda x: x.get("index", 0))
        out2.extend([list(x["embedding"]) for x in items])
    return out2


def embed_query(text: str) -> list[float]:
    vecs = embed_texts([text], is_query=True)
    return vecs[0] if vecs else []


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


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float | None = None,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    t = _HTTP_TIMEOUT if timeout is None else timeout
    try:
        with urlopen(req, timeout=t) as resp:
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
        reason = str(e.reason or e)
        if "timed out" in reason.lower():
            raise RuntimeError(
                f"LLM request timed out after {t:.0f}s ({model_hint_for_timeout(url)}). "
                f"For 32B models: pre-load the model in LM Studio Developer tab before "
                f"Re-check (JIT swap can take 10+ min on 32 GB). "
                f"Increase PHOTO_INDEX_LLM_TIMEOUT_BIG (now {_HTTP_TIMEOUT_BIG:.0f}s)."
            ) from e
        raise RuntimeError(
            f"Could not reach LLM server at {url} ({reason}). "
            "Is LM Studio Local Server running with a model loaded?"
        ) from e
    if not raw.strip():
        return {}
    return json.loads(raw)


def model_hint_for_timeout(_url: str) -> str:
    return "LM Studio / OpenAI-compatible server"


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


def _openai_chat_with_shrink(
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout: float,
    max_tokens: int,
    stream: bool,
) -> str:
    attempt_messages = messages
    n_ctx_hint: int | None = None
    last_error: ContextOverflowError | None = None
    for _ in range(max(1, _MAX_SHRINK_ATTEMPTS)):
        try:
            return _openai_compatible_chat(
                model=model,
                messages=attempt_messages,
                timeout=timeout,
                max_tokens=max_tokens,
                stream=stream,
            )
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
    return _openai_compatible_chat(
        model=model,
        messages=attempt_messages,
        timeout=timeout,
        max_tokens=max_tokens,
        stream=stream,
    )


def _parse_openai_sse_line(line: str) -> tuple[str, str]:
    """Return (content_piece, reasoning_piece) from one SSE line."""
    if not line.startswith("data:"):
        return "", ""
    chunk = line[5:].strip()
    if not chunk or chunk == "[DONE]":
        return "", ""
    try:
        obj = json.loads(chunk)
    except json.JSONDecodeError:
        return "", ""
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", ""
    first = choices[0]
    if not isinstance(first, dict):
        return "", ""
    reasoning = ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        if delta.get("reasoning_content") is not None:
            reasoning = str(delta["reasoning_content"])
        if delta.get("content") is not None:
            return str(delta["content"]), reasoning
    msg = first.get("message")
    if isinstance(msg, dict) and msg.get("content") is not None:
        return str(msg["content"]), reasoning
    text = first.get("text")
    if text is not None:
        return str(text), reasoning
    return "", reasoning


def _openai_compatible_chat_stream(
    *,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> str:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    req = Request(url, data=data, headers=headers, method="POST")
    # Long wall-clock for JIT model load; per-read uses the same socket timeout.
    read_timeout = max(timeout, _READ_IDLE_TIMEOUT)
    parts: list[str] = []
    reasoning_parts: list[str] = []
    try:
        with urlopen(req, timeout=read_timeout) as resp:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                piece, reasoning = _parse_openai_sse_line(
                    raw_line.decode("utf-8", errors="replace").strip()
                )
                if piece:
                    parts.append(piece)
                if reasoning:
                    reasoning_parts.append(reasoning)
    except URLError as e:
        reason = str(e.reason or e)
        if "timed out" in reason.lower():
            raise RuntimeError(
                f"LLM stream timed out after {read_timeout:.0f}s waiting for tokens. "
                f"Pre-load qwen2.5-vl-32b-instruct in LM Studio before Re-check; "
                f"JIT load on 32 GB can exceed 10 minutes. "
                f"Set PHOTO_INDEX_LLM_TIMEOUT_BIG (now {_HTTP_TIMEOUT_BIG:.0f}s)."
            ) from e
        raise RuntimeError(
            f"Could not reach LLM server at {url} ({reason}). "
            "Is LM Studio Local Server running with a model loaded?"
        ) from e
    content = "".join(parts).strip()
    if content:
        return content
    # Reasoning model ran out of budget before the answer: surface the thinking
    # rather than returning blank or forcing an expensive non-stream re-generation.
    return "".join(reasoning_parts).strip()


def _openai_compatible_chat(
    *,
    model: str,
    messages: list[dict[str, str]],
    timeout: float,
    max_tokens: int,
    stream: bool,
) -> str:
    url = f"{openai_base_url()}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
        # Grounded RAG answers must quote records verbatim; the server-side
        # default (~0.8) makes 4-bit models invent dates/amounts/merchants.
        "temperature": float(os.environ.get("PHOTO_INDEX_LLM_TEMPERATURE", "0.2")),
    }
    if stream:
        text = _openai_compatible_chat_stream(url=url, payload=payload, timeout=timeout)
        if text:
            return text
        # Some servers ignore stream=true; fall back once.
        payload = dict(payload)
        payload["stream"] = False
    data = _http_json("POST", url, payload, timeout=timeout)
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
        if content is not None and str(content).strip():
            return str(content).strip()
        # Reasoning models put their chain-of-thought in `reasoning_content`. If the
        # final answer never materialized (e.g. max_tokens consumed by thinking),
        # surface the reasoning rather than returning a blank reply.
        reasoning = msg.get("reasoning_content")
        if reasoning is not None and str(reasoning).strip():
            return str(reasoning).strip()
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
