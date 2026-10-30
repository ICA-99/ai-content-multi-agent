"""Resilient access to Groq models.

Every LLM call in the project goes through one of two functions:

* generate()            plain text (blog, outline, notes, LinkedIn post)
* generate_structured() validated Pydantic object via Groq's native json_schema mode

Both share the same safety layer:

* A shared tokens-per-minute budget per model (prevents 429 errors).
* Automatic shrinking of oversized inputs (prevents 413 errors).
* Retries with backoff for rate limits, timeouts, 5xx and empty replies.
* Output-limit handling: a reply cut off by max_tokens is retried with a bigger budget.
* Model fallback when one model keeps failing.

Structured calls send the schema as `response_format` (constrained decoding with
strict=True, then best-effort strict=False) instead of describing JSON in the prompt.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from collections import defaultdict
from typing import Any, TypeVar

import groq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.config import get_settings
from app.progress import log

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

CHARS_PER_TOKEN = 3.5          # conservative estimate (real text is usually ~4)
MIN_USER_TOKENS = 800
MIN_OUTPUT_TOKENS = 512
STRUCTURED_TEMPERATURE = 0.2
MAX_LENGTH_RETRIES = 2
TRUNCATION_MARKER = "\n\n[... shortened to fit the token limit ...]"

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_WAIT_RE = re.compile(r"try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)", re.I)
_PART_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")

_EFFORT_ENABLED = True  # switched off automatically if the installed SDK rejects it


# ---------------- Errors ----------------

class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMFatalError(LLMError):
    """Retrying cannot help (invalid API key, outdated library)."""


class LLMUnavailableError(LLMError):
    """All retries and all fallback models failed."""


# ---------------- Token window (shared by all threads) ----------------

class _TokenWindow:
    """Sliding 60-second token counter per model, safe for parallel graph nodes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, list[list[float]]] = defaultdict(list)

    def _prune(self, model: str, now: float) -> list[list[float]]:
        events = [e for e in self._events[model] if now - e[0] < 60.0]
        self._events[model] = events
        return events

    def reserve(self, model: str, tokens: int, cap: int) -> tuple[float, list[float] | None]:
        """Reserve tokens atomically. Returns (0, entry) or (seconds_to_wait, None)."""
        with self._lock:
            now = time.monotonic()
            events = self._prune(model, now)
            used = sum(e[1] for e in events)

            if not events or used + tokens <= cap:
                entry = [now, float(tokens)]
                events.append(entry)
                return 0.0, entry

            freed = 0.0
            for ts, amount in events:  # chronological order
                freed += amount
                if used - freed + tokens <= cap:
                    return max(ts + 60.0 - now, 0.0) + 0.5, None
            return 60.0, None

    def settle(self, entry: list[float], actual_tokens: int) -> None:
        """Replace the reservation with the real usage (0 if the call failed)."""
        with self._lock:
            entry[1] = float(actual_tokens)


_WINDOW = _TokenWindow()


# ---------------- Small helpers ----------------

def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _truncate_head(text: str, max_tokens: int) -> str:
    """Keep the beginning of the text, cut at a natural boundary."""
    max_chars = max(int(max_tokens * CHARS_PER_TOKEN), 200)
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    for sep in ("\n\n", "\n", " "):
        idx = cut.rfind(sep)
        if idx > max_chars * 0.8:
            cut = cut[:idx]
            break
    return cut.rstrip() + TRUNCATION_MARKER


def _fit(fixed_tokens: int, user: str, out_tokens: int, cap: int, shrink: float) -> tuple[str, int]:
    """Make sure input + output tokens fit under the per-minute cap."""
    out_tokens = max(MIN_OUTPUT_TOKENS, min(out_tokens, cap - fixed_tokens - MIN_USER_TOKENS))
    user_budget = max(int((cap - fixed_tokens - out_tokens) * shrink), 200)

    if estimate_tokens(user) > user_budget:
        logger.warning(
            "Input too large (~%d tokens); shortening to ~%d tokens.",
            estimate_tokens(user),
            user_budget,
        )
        user = _truncate_head(user, user_budget)
    return user, out_tokens


def _parse_wait(message: str) -> float | None:
    """Parse Groq's 'try again in 18.02s' / '1m2.5s' / '250ms' hint."""
    match = _WAIT_RE.search(message)
    if not match:
        return None
    seconds = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(v) * seconds[u] for v, u in _PART_RE.findall(match.group(1)))


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        try:
            value = headers.get("retry-after")
            if value:
                return float(value)
        except (TypeError, ValueError):
            pass
    return _parse_wait(str(exc))


def _classify(exc: Exception) -> str:
    """Return: fatal | skip_model | rate | too_large | bad_request | retry."""
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()

    # Groq reports "request too large" as 413 with a rate-limit code, so check it first.
    if status == 413 or "request too large" in text or "context length" in text:
        return "too_large"
    if status == 429 or "rate_limit_exceeded" in text:
        return "rate"
    if status == 401:
        return "fatal"
    if status in (403, 404):
        return "skip_model"
    if isinstance(exc, groq.APIConnectionError) or (isinstance(status, int) and status >= 500):
        return "retry"
    if status == 400:
        return "bad_request"
    return "retry"


def _to_text(content: Any) -> str:
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return _THINK_RE.sub("", str(content or "")).strip()


def _total_tokens(response: Any) -> int:
    usage = getattr(response, "usage_metadata", None) or {}
    total = usage.get("total_tokens")
    return int(total) if total else 0


def _finish_reason(response: Any) -> str | None:
    return (getattr(response, "response_metadata", None) or {}).get("finish_reason")


def _build_llm(model: str, max_tokens: int | None, temperature: float | None = None) -> ChatGroq:
    s = get_settings()
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": s.temperature if temperature is None else temperature,
        "api_key": s.groq_api_key or None,
        "timeout": s.request_timeout,
        "max_retries": 0,  # retries are handled by this module
        "max_tokens": max_tokens,
    }
    if _EFFORT_ENABLED and s.reasoning_effort and "gpt-oss" in model:
        kwargs["reasoning_effort"] = s.reasoning_effort
    return ChatGroq(**kwargs)


def get_llm(model_name: str | None = None, max_tokens: int | None = None) -> ChatGroq:
    """Plain ChatGroq client (no retries). Prefer generate() / generate_structured()."""
    return _build_llm(model_name or get_settings().groq_model, max_tokens)


def _wait_for_capacity(model: str, needed: int, cap: int) -> list[float]:
    while True:
        wait, entry = _WINDOW.reserve(model, needed, cap)
        if entry is not None:
            return entry
        wait = min(wait, 60.0)
        log(f"⏳ Waiting {wait:.0f}s to stay under the tokens-per-minute limit...")
        time.sleep(wait)


def _invoke(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float | None,
    schema: type[BaseModel] | None,
    strict: bool,
) -> tuple[Any, BaseModel | None]:
    """One API call. Returns (raw AIMessage, parsed object or None)."""
    llm = _build_llm(model, max_tokens, temperature)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]

    if schema is None:
        return llm.invoke(messages), None

    try:
        structured = llm.with_structured_output(
            schema, method="json_schema", strict=strict, include_raw=True
        )
    except (TypeError, ValueError) as exc:
        raise LLMFatalError(
            "Your langchain-groq version does not support json_schema output. "
            "Run: pip install -U langchain-groq"
        ) from exc

    result = structured.invoke(messages)
    return result.get("raw"), result.get("parsed")


# ---------------- Core call (text and structured) ----------------

def _run_on_model(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float | None,
    schema: type[BaseModel] | None = None,
    strict: bool = True,
) -> Any:
    global _EFFORT_ENABLED

    s = get_settings()
    cap = int(s.tpm_limits.get(model, s.default_tpm_limit) * s.tpm_safety)

    # Tokens that are sent on every attempt: the system prompt plus the schema.
    fixed = estimate_tokens(system)
    if schema is not None:
        fixed += estimate_tokens(json.dumps(schema.model_json_schema(), separators=(",", ":")))

    out_tokens = max_tokens
    shrink = 1.0
    bad_requests = 0
    length_retries = 0
    partial = ""
    last_exc: Exception | None = None

    for attempt in range(1, s.max_attempts + 1):
        fitted_user, fitted_out = _fit(fixed, user, out_tokens, cap, shrink)
        needed = fixed + estimate_tokens(fitted_user) + fitted_out
        entry = _wait_for_capacity(model, needed, cap)

        try:
            raw, parsed = _invoke(
                model, system, fitted_user, fitted_out, temperature, schema, strict
            )
        except LLMFatalError:
            _WINDOW.settle(entry, 0)
            raise
        except Exception as exc:
            _WINDOW.settle(entry, 0)
            last_exc = exc
            kind = _classify(exc)
            logger.warning(
                "%s attempt %d/%d failed (%s): %s",
                model, attempt, s.max_attempts, kind, str(exc)[:300],
            )

            if _EFFORT_ENABLED and "reasoning_effort" in str(exc).lower():
                _EFFORT_ENABLED = False  # installed SDK does not support it
                continue

            if kind == "fatal":
                raise LLMFatalError(
                    "Groq rejected the API key (401). Check GROQ_API_KEY in your .env file."
                ) from exc

            if kind == "skip_model":
                raise

            if kind == "rate":
                wait = _retry_after(exc)
                delay = (wait if wait is not None else 10.0) + 1.0
                if delay > s.max_rate_wait:
                    raise  # most likely a daily quota: let a fallback model handle it
                log(f"⏳ Rate limit reached. Waiting {delay:.0f}s before retrying...")
                time.sleep(delay)
                continue

            if kind == "too_large":
                shrink *= 0.6
                continue

            if kind == "bad_request":
                bad_requests += 1
                if bad_requests >= 2:
                    raise
                out_tokens = int(fitted_out * 1.5)
                continue

            time.sleep(min(2 ** attempt, 20) + random.uniform(0, 1))
            continue

        _WINDOW.settle(entry, _total_tokens(raw) or needed)
        finish = _finish_reason(raw)

        # ----- Structured output -----
        if schema is not None:
            if parsed is not None:
                return parsed

            last_exc = LLMError(
                f"{model} returned output that does not match the schema "
                f"(finish_reason={finish})"
            )
            logger.warning("%s", last_exc)
            # Cut off by max_tokens (often reasoning ate the budget): give it more room.
            out_tokens = int(fitted_out * (2 if finish == "length" else 1.3))
            time.sleep(1.0)
            continue

        # ----- Plain text -----
        text = _to_text(getattr(raw, "content", ""))

        if text and finish == "length" and length_retries < MAX_LENGTH_RETRIES:
            length_retries += 1
            partial = text
            out_tokens = int(fitted_out * 2)
            logger.warning(
                "%s output hit max_tokens (retry %d/%d); doubling the budget to %d.",
                model, length_retries, MAX_LENGTH_RETRIES, out_tokens,
            )
            continue

        if text:
            if finish == "length":
                logger.warning("%s output is still truncated after retries; using it as is.", model)
            return text

        logger.warning("%s returned an empty reply; retrying with a larger budget.", model)
        last_exc = LLMError("empty response")
        out_tokens = int(fitted_out * 1.5)
        time.sleep(1.0)

    if partial:
        return partial
    raise LLMUnavailableError(f"{model} failed after {s.max_attempts} attempts.") from last_exc


# ---------------- Public API ----------------

def generate(
    system: str,
    user: str,
    model_name: str | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Plain text generation with throttling, retries and model fallback."""
    s = get_settings()
    primary = model_name or s.groq_model
    chain = [primary] + [m for m in s.fallback_models if m != primary]
    out_tokens = max_tokens or s.max_tokens_default

    last_exc: Exception | None = None
    for model in chain:
        try:
            return _run_on_model(model, system, user, out_tokens, temperature)
        except LLMFatalError:
            raise
        except Exception as exc:
            last_exc = exc
            logger.warning("Model %s unavailable (%s): %s", model, type(exc).__name__, str(exc)[:300])
            if model != chain[-1]:
                log(f"⚠️ {model} is unavailable. Switching to a fallback model...")

    raise LLMUnavailableError(
        "All configured models failed. This is usually a daily quota limit or a network "
        f"problem. Last error: {str(last_exc)[:300]}"
    ) from last_exc


def generate_structured(
    schema: type[T],
    system: str,
    user: str,
    model_name: str | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> T:
    """Structured output through Groq's native json_schema mode.

    For each structured-capable model: strict=True (constrained decoding) first, then
    strict=False (best effort). The result is always a validated Pydantic object.
    """
    s = get_settings()
    primary = model_name or s.groq_model

    chain: list[str] = []
    for model in [primary, *s.structured_models]:
        if model in s.structured_models and model not in chain:
            chain.append(model)
    if not chain:
        chain = list(s.structured_models)

    out_tokens = max_tokens or s.max_tokens_default
    temp = STRUCTURED_TEMPERATURE if temperature is None else temperature

    last_exc: Exception | None = None
    for model in chain:
        for strict in (True, False):
            try:
                return _run_on_model(model, system, user, out_tokens, temp, schema, strict)
            except LLMFatalError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Structured call failed on %s (strict=%s, %s): %s",
                    model, strict, type(exc).__name__, str(exc)[:300],
                )
                if _classify(exc) in ("rate", "skip_model"):
                    break  # strict=False would hit the same wall on this model
        if model != chain[-1]:
            log(f"⚠️ {model} could not return structured output. Trying another model...")

    raise LLMUnavailableError(
        f"Could not get structured output from any model. Last error: {str(last_exc)[:300]}"
    ) from last_exc