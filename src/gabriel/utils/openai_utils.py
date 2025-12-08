"""
This module reimplements the original GABRIEL `openai_utils.py` for the
OpenAI Responses API with several improvements:

* Rate limit introspection – a helper fetches the current token/request
  budget from the ``x‑ratelimit-*`` response headers returned by a cheap
  ``GET /v1/models`` call.  These values are used to display how many
  tokens and requests remain per minute.
* User‑friendly summary – before a long job starts, the module prints a
  summary showing the number of prompts, input words, remaining rate‑limit
  capacity, usage tier qualifications, and an estimated cost.  It also
  explains the purpose of the ``max_output_tokens`` parameter.
* Respect for explicit ``max_output_tokens`` – the helper no longer
  injects a default ceiling when the caller omits the parameter.  Earlier
  versions applied a 2 500 token cap once the remaining minute budget
  dipped below one million tokens, which confused users by silently
  truncating long responses.  Callers who want a limit can still provide
  one explicitly.
* Improved rate‑limit gating – the token limiter now estimates the worst
  possible output length when the cutoff is unspecified by assuming
  the response could be as long as the input.  This avoids grossly
  underestimating throughput while still honouring the per‑minute token
  budget.
* Exponential backoff with jitter – the retry logic uses a random
  exponential backoff when rate‑limit errors occur, following OpenAI’s
  guidelines for handling 429 responses.

The overall API surface remains compatible with the original file: the
public functions ``get_response`` and ``get_all_responses`` still
exist, but the argument ``max_tokens`` has been renamed to
``max_output_tokens`` to match the Responses API.  A legacy alias
``max_tokens`` is accepted for backward compatibility.
"""

from __future__ import annotations

import asyncio
import csv
import functools
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time
import subprocess
import sys
import textwrap
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict, deque
from collections.abc import Iterable
import pickle

from gabriel.utils.logging import get_logger, set_log_level
import logging
import math
import pandas as pd
from aiolimiter import AsyncLimiter
from tqdm.auto import tqdm
import openai
import statistics
import numpy as np
import tiktoken
from dataclasses import dataclass, fields

logger = get_logger(__name__)

# Track whether the verbose usage sheet has been shown to avoid repeating the
# static "info sheet" content on subsequent runs.
_USAGE_SHEET_PRINTED = False
_DEPENDENCIES_VERIFIED = False

# Cap the number of prompts we fully scan when estimating words/tokens.  Large
# datasets are sampled to keep start-up time predictable.
_ESTIMATION_SAMPLE_SIZE = 5000

# Try to import requests/httpx for rate‑limit introspection
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore
try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore

# Bring in specific error classes for granular handling
try:
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        InvalidRequestError,
        RateLimitError,
    )  # type: ignore
except Exception:
    APIConnectionError = Exception  # type: ignore
    APIError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore
    BadRequestError = Exception  # type: ignore
    InvalidRequestError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore

from gabriel.utils.parsing import safe_json

# single connection pool per process, keyed by base URL and created lazily
_clients_async: Dict[Optional[str], openai.AsyncOpenAI] = {}


def _progress_bar(*args: Any, verbose: bool = True, **kwargs: Any):
    """Construct a tqdm progress bar that degrades gracefully."""

    disable = kwargs.pop("disable", False) or not verbose
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, disable=disable, **kwargs)


def _in_notebook() -> bool:
    """Return True when running inside a Jupyter/colab notebook."""

    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _display_example_prompt(example_prompt: str, *, verbose: bool = True) -> None:
    """Show a concise example prompt with a collapsible view in notebooks."""

    if not verbose or not example_prompt:
        return
    saved_path: Optional[Path] = None
    try:
        saved_path = Path(tempfile.gettempdir()) / "gabriel_example_prompt.txt"
        saved_path.write_text(example_prompt)
    except Exception:
        saved_path = None
    print("Example prompt (full text):")
    print(textwrap.indent(example_prompt, "  "))
    if saved_path:
        print(f"(Saved full example prompt to {saved_path} for easy copy/paste.)")


def _get_client(base_url: Optional[str] = None) -> openai.AsyncOpenAI:
    """Return a cached ``AsyncOpenAI`` client for ``base_url``.

    When ``base_url`` is ``None`` the default OpenAI endpoint is used.  A client
    is created on first use and reused for subsequent calls with the same base
    URL to benefit from connection pooling.
    """

    url = base_url or os.getenv("OPENAI_BASE_URL")
    key: Optional[str] = url
    client = _clients_async.get(key)
    if client is None:
        kwargs: Dict[str, Any] = {}
        if url:
            kwargs["base_url"] = url
        if httpx is not None:
            try:
                kwargs.setdefault(
                    "timeout",
                    httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
                )
            except Exception:
                # Fall back to the SDK default if constructing the timeout fails
                pass
        client = openai.AsyncOpenAI(**kwargs)
        _clients_async[key] = client
    return client

# Default safety cutoff when token capacity is low
# Historical default used as a conservative upper bound when rate limits were
# not known.  We keep the constant so older call sites that import it continue
# to function, but the helper no longer applies this ceiling automatically.
DEFAULT_MAX_OUTPUT_TOKENS = 2500

# Estimated output tokens per prompt used for cost estimation when no cutoff is specified.
# When a user does not explicitly set ``max_output_tokens``, we assume that each response
# will contain roughly this many tokens.  This value is used solely for estimating cost
# and determining how many parallel requests can safely run under the token budget.
ESTIMATED_OUTPUT_TOKENS_PER_PROMPT = 250

# Conservative headroom when translating observed rate limits into concurrency and limiter budgets.
# Using less than the reported limit provides a buffer for short spikes and accounting inaccuracies.
RATE_LIMIT_HEADROOM = 0.85
# Additional planning buffer applied when translating reported rate limits into budgets.
PLANNING_RATE_LIMIT_BUFFER = 0.8
# Buffer applied to the estimated tokens per call to avoid optimistic throughput estimates.
TOKEN_ESTIMATE_BUFFER = 1.35
# Cushion applied to the expected output tokens when planning budgets so we never assume perfectly
# short responses for lower-tier accounts.
OUTPUT_TOKEN_HEADROOM = 1.2

# ---------------------------------------------------------------------------
# Helper dataclasses and token utilities


@dataclass
class StatusTracker:
    """Simple container for bookkeeping counters."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_timeout_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0.0


@dataclass
class DummyResponseSpec:
    """Configuration object describing synthetic responses for dummy runs."""

    responses: Optional[Any] = None
    duration: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    reasoning_summary: Optional[str] = None
    response_id: Optional[str] = None
    successful: Optional[bool] = None
    error_log: Optional[Union[str, List[str]]] = None
    warning: Optional[str] = None


class BackgroundTimeoutError(asyncio.TimeoutError):
    """Timeout raised while polling a background response."""

    def __init__(self, response_id: Optional[str], last_response: Any, message: str):
        super().__init__(message)
        self.response_id = response_id
        self.last_response = last_response


def _extract_retry_after_seconds(error: Exception) -> Optional[float]:
    """Return a retry-after duration in seconds when available."""

    for attr in ("retry_after", "retry_after_s", "retry_after_seconds"):
        retry_value = getattr(error, attr, None)
        if isinstance(retry_value, (int, float)) and retry_value > 0:
            return float(retry_value)
    retry_ms = getattr(error, "retry_after_ms", None)
    if isinstance(retry_ms, (int, float)) and retry_ms > 0:
        return float(retry_ms) / 1000.0
    message = str(error)
    if not message:
        return None
    match = re.search(r"after\s+([0-9]+(?:\.[0-9]+)?)\s*seconds", message)
    if match:
        try:
            parsed = float(match.group(1))
        except ValueError:
            return None
        if parsed > 0:
            return parsed
    return None


def _is_quota_error_message(message: str) -> bool:
    """Return True when the error text indicates an exhausted quota."""

    return bool(message) and "quota" in message.lower()


def _get_tokenizer(model_name: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding for the model or a sensible default."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        class _ApproxEncoder:
            def encode(self, text: str) -> List[int]:
                return [0] * max(1, _approx_tokens(text))

        return _ApproxEncoder()

# Usage tiers with qualifications and monthly limits for printing
TIER_INFO = [
    {
        "tier": "Free",
        "qualification": "User must be in an allowed geography",
        "monthly_quota": "$100 / month",
    },
    {"tier": "Tier 1", "qualification": "$5 paid", "monthly_quota": "$100 / month"},
    {
        "tier": "Tier 2",
        "qualification": "$50 paid and 7+ days since first payment",
        "monthly_quota": "$500 / month",
    },
    {
        "tier": "Tier 3",
        "qualification": "$100 paid and 7+ days since first payment",
        "monthly_quota": "$1 000 / month",
    },
    {
        "tier": "Tier 4",
        "qualification": "$250 paid and 14+ days since first payment",
        "monthly_quota": "$5 000 / month",
    },
    {
        "tier": "Tier 5",
        "qualification": "$1 000 paid and 30+ days since first payment",
        "monthly_quota": "$200 000 / month",
    },
]

# Truncated pricing table (USD per million tokens) for a few common models
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # model family       input   cached_input   output   batch_factor
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00, "batch": 0.5},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60, "batch": 0.5},
    "gpt-4.1-nano": {
        "input": 0.10,
        "cached_input": 0.025,
        "output": 0.40,
        "batch": 0.5,
    },
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00, "batch": 0.5},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60, "batch": 0.5},
    "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00, "batch": 0.5},
    "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40, "batch": 0.5},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00, "batch": 0.5},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00, "batch": 0.5},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40, "batch": 0.5},
    "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40, "batch": 0.5},
    "o3-deep-research": {
        "input": 10.00,
        "cached_input": 2.50,
        "output": 40.00,
        "batch": 0.5,
    },
    "o4-mini-deep-research": {
        "input": 2.00,
        "cached_input": 0.50,
        "output": 8.00,
        "batch": 0.5,
    },
}


def _print_tier_explainer(verbose: bool = True) -> None:
    """Print a helpful explanation of usage tiers and how to increase them.

    This helper can be called when a user encounters errors that may be
    related to low quotas or tier limitations.  It summarises the
    qualifications for each tier and encourages users to check their
    payment status and billing page.  The message is only printed when
    ``verbose`` is ``True``.
    """
    if not verbose:
        return
    print("\n===== Tier explainer =====")
    print(
        "Your organization’s ability to call the OpenAI API is governed by usage tiers."
    )
    print(
        "As you spend more on the API, you are automatically graduated to higher tiers with larger token and request limits."
    )
    print("Here are the current tiers and how to qualify:")
    for tier in TIER_INFO:
        print(
            f"  • {tier['tier']}: qualify by {tier['qualification']}; monthly quota {tier['monthly_quota']}"
        )
    print("If you are encountering rate limits or truncated outputs, consider:")
    print(
        "  – Checking your current spend and ensuring you have met the payment criteria for a higher tier."
    )
    print(
        "  – Adding funds or updating billing details at https://platform.openai.com/settings/organization/billing/."
    )
    print("  – Reducing the number of parallel requests or batching your workload.")


def _approx_tokens(text: str) -> int:
    """Roughly estimate the token count from a string by assuming ~1.5 tokens per word."""
    return int(len(str(text).split()) * 1.5)


def _lookup_model_pricing(model: str) -> Optional[Dict[str, float]]:
    """Find a pricing entry for ``model`` by prefix match (case‑insensitive)."""
    key = model.lower()
    # Find the most specific prefix match by selecting the longest matching prefix.
    best_match: Optional[Dict[str, float]] = None
    best_len = -1
    for prefix, pricing in MODEL_PRICING.items():
        if key.startswith(prefix) and len(prefix) > best_len:
            best_match = pricing
            best_len = len(prefix)
    return best_match


def _estimate_cost(
    prompts: List[str],
    n: int,
    max_output_tokens: Optional[int],
    model: str,
    use_batch: bool,
    *,
    sample_size: int = _ESTIMATION_SAMPLE_SIZE,
) -> Optional[Dict[str, float]]:
    """Estimate input/output tokens and cost for a set of prompts.

    Returns a dict with keys ``input_tokens``, ``output_tokens``, ``input_cost``, ``output_cost``, and ``total_cost``.
    If the model pricing is unavailable, returns ``None``.
    """
    pricing = _lookup_model_pricing(model)
    if pricing is None:
        return None
    # Estimate tokens: sample large datasets to avoid long start-up times.
    total_prompts = len(prompts)
    if total_prompts == 0:
        return None
    if sample_size and total_prompts > sample_size:
        # Deterministic sampling keeps estimates stable across runs.
        rng = random.Random(total_prompts)
        sampled = rng.sample(prompts, sample_size)
        avg_tokens = sum(_approx_tokens(p) for p in sampled) / float(sample_size)
        input_tokens = int(avg_tokens * total_prompts * max(1, n))
    else:
        input_tokens = sum(_approx_tokens(p) for p in prompts) * max(1, n)
    # Estimate output tokens: when no cutoff is provided we assume a reasonable default
    # number of output tokens per prompt.  This prevents the cost estimate from
    # ballooning for long inputs, which previously assumed the output could be as long
    # as the input.
    if max_output_tokens is None:
        # Use the per‑prompt estimate for each response
        output_tokens = ESTIMATED_OUTPUT_TOKENS_PER_PROMPT * max(1, n) * len(prompts)
    else:
        output_tokens = max_output_tokens * max(1, n) * len(prompts)
    cost_in = (input_tokens / 1_000_000) * pricing["input"]
    cost_out = (output_tokens / 1_000_000) * pricing["output"]
    if use_batch:
        cost_in *= pricing["batch"]
        cost_out *= pricing["batch"]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": cost_in,
        "output_cost": cost_out,
        "total_cost": cost_in + cost_out,
    }


def _estimate_dataset_stats(
    prompts: List[str],
    *,
    sample_size: int = _ESTIMATION_SAMPLE_SIZE,
) -> Dict[str, Any]:
    """Return rough totals for words and tokens without scanning massive datasets.

    The helper samples up to ``sample_size`` prompts and scales the totals to the
    full dataset.  This keeps initial reporting fast even for hundreds of
    thousands of prompts.
    """

    total_prompts = len(prompts)
    if total_prompts == 0:
        return {"word_count": 0, "token_count": 0, "sampled": False, "sample_size": 0}
    if sample_size and total_prompts > sample_size:
        rng = random.Random(total_prompts)
        sample = rng.sample(prompts, sample_size)
        avg_words = sum(len(str(p).split()) for p in sample) / float(sample_size)
        avg_tokens = sum(_approx_tokens(p) for p in sample) / float(sample_size)
        return {
            "word_count": int(avg_words * total_prompts),
            "token_count": int(avg_tokens * total_prompts),
            "sampled": True,
            "sample_size": sample_size,
        }
    return {
        "word_count": sum(len(str(p).split()) for p in prompts),
        "token_count": sum(_approx_tokens(p) for p in prompts),
        "sampled": False,
        "sample_size": total_prompts,
    }


def _ensure_runtime_dependencies(packages: Optional[List[str]] = None, *, verbose: bool = True) -> None:
    """Install missing runtime dependencies in a best-effort manner.

    The function is intentionally lightweight: it checks for a small set of
    packages and silently returns when everything is already present.  When a
    package is missing, ``pip`` is invoked to install only the missing items so
    the helper works in local, Colab, Databricks, and CI environments without
    user intervention.
    """

    global _DEPENDENCIES_VERIFIED
    if _DEPENDENCIES_VERIFIED:
        return
    pkgs = packages or ["wheel", "tiktoken", "aiolimiter", "httpx", "requests"]
    missing = [pkg for pkg in pkgs if importlib.util.find_spec(pkg) is None]
    if not missing:
        _DEPENDENCIES_VERIFIED = True
        return
    if verbose:
        print(
            "Installing missing dependencies for GABRIEL (once per session): "
            + ", ".join(sorted(missing))
        )
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", *missing],
            check=True,
        )
        _DEPENDENCIES_VERIFIED = True
    except Exception as exc:
        logger.warning("Automatic dependency installation failed: %s", exc)


def _print_run_banner(
    *,
    prompts: List[str],
    model: str,
    n: int,
    use_batch: bool,
    estimated_cost: Optional[Dict[str, float]],
    stats: Dict[str, Any],
    verbose: bool = True,
) -> None:
    """Print an immediate run overview so users see progress right away."""

    if not verbose:
        return
    print("\n===== Run kickoff =====")
    print(
        f"Prompts: {len(prompts):,} | Approx. words: {stats.get('word_count', 0):,}"
        f"{' (sampled)' if stats.get('sampled') else ''}"
    )
    print(f"Model: {model} | Mode: {'batch' if use_batch else 'streaming'}")
    if estimated_cost:
        print(
            f"Estimated {'batch' if use_batch else 'synchronous'} cost: ${estimated_cost['total_cost']:.4f} "
            f"(input: ${estimated_cost['input_cost']:.4f}, output: ${estimated_cost['output_cost']:.4f})"
        )
    else:
        print("Estimated cost unavailable for this model.")
    print("Preparing checkpoints and rate limits...")



def _require_api_key() -> str:
    """Return the API key or raise a runtime error if missing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable must be set or passed via OpenAIClient(api_key)."
        )
    return api_key


def _get_rate_limit_headers(
    model: str = "gpt-5-mini", base_url: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """Retrieve rate‑limit headers via a cheap API request.

    The OpenAI platform does not yet expose a dedicated endpoint for
    checking how many requests or tokens remain in your minute quota.  In
    practice, these values are only communicated via ``x‑ratelimit-*``
    headers on API responses.  The newer *Responses* API does not
    consistently include these headers as of mid‑2025【360365694688557†L209-L243】, but it
    may in the future.  To accommodate current and future behaviour, this
    helper first tries a minimal call against the Responses endpoint and
    falls back to a tiny call against the Chat completions endpoint when
    the headers are absent.  Both calls cap generation at one token to
    minimise usage.

    :param model: The model to use for the dummy request.  Matching the
      model you intend to use yields the most accurate limits.
    :returns: A dictionary containing limit and remaining values for
      requests and tokens if successful, otherwise ``None``.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    base = base.rstrip("/")
    # Define two candidate endpoints: the Responses API and the Chat
    # completions API.  In mid‑2025 the Responses API often omits rate‑limit
    # headers【360365694688557†L209-L243】, but OpenAI may add them in the future.  We try
    # the Responses endpoint first to see if headers are now included; if
    # missing, we fall back to a minimal call to the chat completions
    # endpoint.  Both calls cap generation at one token to minimise usage.
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    # Responses API payload (first attempt)
    candidates.append(
        (
            f"{base}/responses",
            {
                "model": model,
                "input": [
                    {"role": "user", "content": "Hello"},
                ],
                "truncation": "auto",
                "max_output_tokens": 1,
            },
        )
    )
    # Chat completions API payload (fallback)
    candidates.append(
        (
            f"{base}/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 1,
            },
        )
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for url, payload in candidates:
        for client in (requests, httpx):
            if client is None:
                continue
            try:
                resp = client.post(url, headers=headers, json=payload, timeout=10)  # type: ignore
                h = getattr(resp, "headers", {})  # type: ignore
                new_h = {k.lower(): v for k, v in h.items()}
                # Collect both standard and usage‑based headers.  If the
                # responses API is missing them, continue to the next
                # candidate.
                limit_requests = new_h.get("x-ratelimit-limit-requests")
                remaining_requests = new_h.get("x-ratelimit-remaining-requests")
                reset_requests = new_h.get("x-ratelimit-reset-requests")
                limit_tokens = new_h.get("x-ratelimit-limit-tokens") or new_h.get(
                    "x-ratelimit-limit-tokens_usage_based"
                )
                remaining_tokens = new_h.get("x-ratelimit-remaining-tokens") or new_h.get(
                    "x-ratelimit-remaining-tokens_usage_based"
                )
                reset_tokens = new_h.get("x-ratelimit-reset-tokens") or new_h.get(
                    "x-ratelimit-reset-tokens_usage_based"
                )
                # If any of the primary values are present, return them.  Some
                # providers may omit remaining values until you are close to
                # the limit, so we treat the presence of a limit value as
                # success.
                if limit_requests or limit_tokens:
                    return {
                        "limit_requests": limit_requests,
                        "remaining_requests": remaining_requests,
                        "reset_requests": reset_requests,
                        "limit_tokens": limit_tokens,
                        "remaining_tokens": remaining_tokens,
                        "reset_tokens": reset_tokens,
                    }
            except Exception:
                # Ignore any errors and try the next client or candidate
                continue
    return None


def _print_usage_overview(
    prompts: List[str],
    n: int,
    max_output_tokens: Optional[int],
    model: str,
    use_batch: bool,
    n_parallels: int,
    *,
    verbose: bool = True,
    rate_headers: Optional[Dict[str, str]] = None,
    base_url: Optional[str] = None,
    web_search_warning: Optional[str] = None,
    web_search_parallel_note: Optional[str] = None,
    show_static_sections: bool = True,
    stats: Optional[Dict[str, Any]] = None,
    sample_size: int = _ESTIMATION_SAMPLE_SIZE,
) -> None:
    """Print a summary of usage limits, cost estimate and tier information.

    Optionally takes a pre‑fetched ``rate_headers`` dict to avoid calling
    ``_get_rate_limit_headers`` multiple times per job.  When ``rate_headers``
    is ``None``, the helper will fetch the headers itself.  Optional web-search
    warnings can be provided to display additional caveats alongside the
    usage overview.
    """
    if not verbose:
        return
    print("\n===== OpenAI API usage summary =====")
    if web_search_warning:
        print(web_search_warning)
    if web_search_parallel_note:
        print(web_search_parallel_note)
    stats = stats or _estimate_dataset_stats(prompts, sample_size=sample_size)
    print(f"Prompts: {len(prompts)}")
    suffix = " (sampled)" if stats.get("sampled") else ""
    print(f"Approx. input words: {stats.get('word_count', 0):,}{suffix}")
    # Fetch fresh headers if not supplied.  Pass the model and base_url so the
    # helper knows which endpoint to probe when performing the dummy call.
    rl = rate_headers if rate_headers is not None else _get_rate_limit_headers(model, base_url=base_url)
    # Determine whether the headers include any meaningful limit values.  Some
    # endpoints (or API tiers) may omit rate‑limit headers, or return zero
    # values, which should be treated as unknown.
    def _parse_float(val: Optional[str]) -> Optional[float]:
        try:
            if val is None:
                return None
            # Treat empty strings as None
            s = str(val).strip()
            if not s:
                return None
            f = float(s)
            return f if f > 0 else None
        except Exception:
            return None

    # Parse rate‑limit values from the response headers.  If no headers are
    # returned or a value is zero/negative, leave the variable as None.
    lim_r_val = rem_r_val = None
    lim_t_val = rem_t_val = None
    reset_r = reset_t = None
    if rl:
        lim_r_val = _parse_float(rl.get("limit_requests"))
        rem_r_val = _parse_float(rl.get("remaining_requests"))
        lim_t_val = _parse_float(rl.get("limit_tokens"))
        rem_t_val = _parse_float(rl.get("remaining_tokens"))
        # Some accounts report usage‑based token limits
        if lim_t_val is None:
            lim_t_val = _parse_float(rl.get("limit_tokens_usage_based"))
        if rem_t_val is None:
            rem_t_val = _parse_float(rl.get("remaining_tokens_usage_based"))
        reset_r = rl.get("reset_requests")
        reset_t = rl.get("reset_tokens") or rl.get("reset_tokens_usage_based")
    # Print raw rate limit information without falling back to configured defaults.
    # If a value is unavailable, display "unknown" instead of substituting another number.
    def fmt(val: Optional[float]) -> str:
        return f"{int(val):,}" if val is not None else "unknown"
    # Print concise rate limit information.  Only the total per‑minute
    # capacities are shown; remaining quotas and reset timers are omitted
    # to reduce clutter.  Unknown values are labelled as "unknown".
    if lim_r_val is not None:
        print(f"Requests per minute: {fmt(lim_r_val)}")
    else:
        print("Requests per minute: unknown (API did not share a request limit)")
    if lim_t_val is not None:
        print(f"Tokens per minute: {fmt(lim_t_val)}")
        words_per_min = int(lim_t_val) // 2
        print(f"Approx. words per minute: {words_per_min:,}")
    else:
        print("Tokens per minute: unknown (API did not share a token limit)")
        print("Approx. words per minute: unknown")
    if lim_r_val is None or lim_t_val is None:
        warning_msg = (
            "⚠️ API did not return complete rate-limit headers. Running with conservative defaults. "
            "If you are on a free/low-balance plan, add funds to avoid quota blocks: "
            "https://platform.openai.com/settings/organization/billing/"
        )
        print(warning_msg)
        logger.warning(warning_msg)
    # Let users know about monthly usage caps in addition to per‑minute limits.
    if show_static_sections:
        print(
            "\nNote: your organization also has a monthly usage cap based on your tier. See the usage tiers below for details."
        )
        tier_lines = [
            f"• {tier['tier']}: qualifies by {tier['qualification']}; monthly quota {tier['monthly_quota']}"
            for tier in TIER_INFO
        ]
        print("\nUsage tiers:")
        for line in tier_lines:
            print(f"  {line}")
    pricing = _lookup_model_pricing(model)
    est = _estimate_cost(prompts, n, max_output_tokens, model, use_batch, sample_size=sample_size)
    if pricing and est:
        print(
            f"\nPricing for model '{model}': input ${pricing['input']}/1M, output ${pricing['output']}/1M"
        )
        if use_batch:
            print("Batch API prices are half the synchronous rates.")
        print(
            f"Estimated token usage: input {est['input_tokens']:,}, output {est['output_tokens']:,}"
        )
        print(
            f"Estimated {'batch' if use_batch else 'synchronous'} cost: ${est['total_cost']:.4f}"
        )
    else:
        print(f"\nPricing for model '{model}' is unavailable; cannot estimate cost.")
    # Compute concurrency based on the retrieved rate limits and token/request budgets.
    try:
        token_total = stats.get("token_count") if stats is not None else None
        avg_input_tokens = (token_total or sum(_approx_tokens(p) for p in prompts)) / max(
            1, len(prompts)
        )
        gating_output = max_output_tokens if max_output_tokens is not None else ESTIMATED_OUTPUT_TOKENS_PER_PROMPT
        tokens_per_call = (avg_input_tokens + gating_output) * max(1, n)
        # Extract numeric values from the rate limit headers.  If a value is missing or
        # zero, treat it as unknown (None).  We deliberately avoid falling back to
        # configured caps here because the user has requested that we rely solely
        # on the API‑provided limits.
        def _pf(val: Optional[str]) -> Optional[float]:
            try:
                if val is None:
                    return None
                s = str(val).strip()
                if not s:
                    return None
                f = float(s)
                return f if f > 0 else None
            except Exception:
                return None
        if rl:
            lim_r_val2 = _pf(rl.get("limit_requests"))
            rem_r_val2 = _pf(rl.get("remaining_requests"))
            lim_t_val2 = _pf(rl.get("limit_tokens")) or _pf(rl.get("limit_tokens_usage_based"))
            rem_t_val2 = _pf(rl.get("remaining_tokens")) or _pf(rl.get("remaining_tokens_usage_based"))
            # Track whether we are capping concurrency because of the minute's
            # remaining allowance or the hard per‑minute limit.  This helps us
            # explain the cap accurately to the caller.
            if rem_r_val2 is not None:
                allowed_req = rem_r_val2
                allowed_req_source = "remaining"
            else:
                allowed_req = lim_r_val2
                allowed_req_source = "limit" if lim_r_val2 is not None else None
            if rem_t_val2 is not None:
                allowed_tok = rem_t_val2
                allowed_tok_source = "remaining"
            else:
                allowed_tok = lim_t_val2
                allowed_tok_source = "limit" if lim_t_val2 is not None else None
        else:
            allowed_req = None
            allowed_tok = None
            allowed_req_source = None
            allowed_tok_source = None
        # Compute concurrency_possible (maximum possible parallelism based on
        # rate limits) before applying the user‑supplied ceiling.  If a value
        # is unknown, treat that dimension as unlimited.
        if allowed_req is None or allowed_req <= 0:
            concurrency_possible_from_requests: Optional[int] = None
        else:
            concurrency_possible_from_requests = int(max(1, allowed_req))
        if allowed_tok is None or allowed_tok <= 0:
            concurrency_possible_from_tokens: Optional[int] = None
        else:
            # Use floor division to avoid fractional parallelism
            concurrency_possible_from_tokens = int(max(1, allowed_tok // tokens_per_call))
        # Determine the theoretical maximum concurrency
        if concurrency_possible_from_requests is None and concurrency_possible_from_tokens is None:
            concurrency_possible: Optional[int] = None
        elif concurrency_possible_from_requests is None:
            concurrency_possible = concurrency_possible_from_tokens
        elif concurrency_possible_from_tokens is None:
            concurrency_possible = concurrency_possible_from_requests
        else:
            concurrency_possible = min(concurrency_possible_from_requests, concurrency_possible_from_tokens)
        # Now compute the concurrency cap based on the user‑specified ceiling
        if concurrency_possible is None:
            concurrency_cap = max(1, n_parallels)
        else:
            concurrency_cap = max(1, min(n_parallels, concurrency_possible))
    except Exception:
        concurrency_possible = None
        concurrency_cap = max(1, n_parallels)
    # Inform the user about dynamic concurrency.  If the calculated cap is lower
    # than the ceiling, explain that upgrading the tier would allow more
    # parallel requests.  If the cap equals the ceiling but additional capacity
    # remains, suggest that the user could increase n_parallels to make use of
    # their limits.  Otherwise, confirm that we will use the available cap.
    if concurrency_cap < n_parallels:
        limiting_messages: List[str] = []
        suggest_upgrade = False
        if (
            concurrency_possible is not None
            and concurrency_possible_from_requests is not None
            and concurrency_possible == concurrency_possible_from_requests
        ):
            if allowed_req_source == "remaining":
                limiting_messages.append(
                    f"the API reported only {int(allowed_req):,} request slots remaining in the current minute"
                )
            elif allowed_req_source == "limit":
                limiting_messages.append(
                    f"your per-minute request limit is {int(allowed_req):,}"
                )
                suggest_upgrade = True
        if (
            concurrency_possible is not None
            and concurrency_possible_from_tokens is not None
            and concurrency_possible == concurrency_possible_from_tokens
        ):
            approx_tokens = int(max(1, allowed_tok)) if allowed_tok is not None else None
            if allowed_tok_source == "remaining" and approx_tokens is not None:
                limiting_messages.append(
                    f"about {approx_tokens:,} tokens remain in the current minute"
                )
            elif allowed_tok_source == "limit" and approx_tokens is not None:
                limiting_messages.append(
                    f"your per-minute token limit is about {approx_tokens:,}"
                )
                suggest_upgrade = True
        if not limiting_messages:
            limiting_messages.append("of the reported rate limits")
        reason = " and ".join(limiting_messages)
        print(
            f"\nNote: we'll run up to {concurrency_cap} requests at the same time instead of {n_parallels} because {reason}."
        )
        if suggest_upgrade:
            print(
                "Upgrading your tier would allow more parallel requests and speed up processing."
            )
    else:
        # If concurrency_possible is larger than the user‑supplied ceiling, let the
        # user know they could increase n_parallels to utilise the available headroom.
        if concurrency_possible is not None and concurrency_possible > n_parallels:
            print(
                f"\nWe are running with {n_parallels} parallel requests, but your current limits could allow up to {int(concurrency_possible)} concurrent requests if desired."
            )
        else:
            print(
                f"\nWe can run up to {concurrency_cap} requests at the same time with your current settings."
            )
    print(
        "\nAdd funds or manage your billing here: https://platform.openai.com/settings/organization/billing/"
    )
    if max_output_tokens is None:
        print(
            f"\nNo explicit output token limit specified; cost estimate assumes about {ESTIMATED_OUTPUT_TOKENS_PER_PROMPT} tokens per prompt for the response."
        )
    else:
        print(
            f"\nmax_output_tokens: {max_output_tokens} (safety cutoff; generation will stop if this is reached)"
        )


def _decide_default_max_output_tokens(
    user_specified: Optional[int],
    rate_headers: Optional[Dict[str, str]] = None,
    *,
    base_url: Optional[str] = None,
) -> Optional[int]:
    """Return the caller supplied cutoff, or ``None`` if no preference.

    Earlier revisions attempted to infer a sensible default by inspecting the
    live rate‑limit headers.  That behaviour surprised users because the
    helper would silently clamp outputs once the remaining budget dipped below
    a hardcoded threshold.  The new policy is straightforward: when a caller
    does not ask for a ceiling we honour that request and allow the model to
    stream its full response.  The ``rate_headers`` and ``base_url`` arguments
    remain for backwards compatibility and to support future heuristics, but
    they are no longer consulted.
    """

    del rate_headers, base_url  # Unused; retained for backward compatibility.
    return user_specified


def _normalise_web_search_filters(
    filters: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Convert caller-friendly web-search filters to the Responses schema.

    ``filters`` mirrors the keyword arguments exposed by :func:`get_response`
    and the higher-level task wrappers.  Callers can supply an
    ``"allowed_domains"`` iterable together with optional location hints –
    ``city``, ``country``, ``region``, ``timezone`` and ``type`` (currently the
    API accepts ``"approximate"``).  Falsy values are stripped so the outgoing
    payload stays compact.

    The Responses API expects domain restrictions under ``filters`` and
    geography hints under ``user_location``.  This helper reshapes the mapping
    accordingly and ignores unknown keys to avoid forwarding unsupported
    filters.
    """

    if not filters:
        return {}

    result: Dict[str, Any] = {}

    allowed_domains = filters.get("allowed_domains")
    if allowed_domains:
        if isinstance(allowed_domains, (str, bytes)) or not isinstance(allowed_domains, Iterable):
            raise TypeError(
                "web_search_filters['allowed_domains'] must be an iterable of domain strings"
            )
        cleaned = [str(d) for d in allowed_domains if d]
        if cleaned:
            result["filters"] = {"allowed_domains": cleaned}

    location_keys = ("city", "country", "region", "timezone", "type")
    location = {
        key: value
        for key, value in ((k, filters.get(k)) for k in location_keys)
        if value
    }
    if location:
        result["user_location"] = location

    return result


def _merge_web_search_filters(
    base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Combine global and per-prompt web-search filter dictionaries.

    Both inputs follow the caller-facing schema accepted by
    :func:`get_all_responses`.  The override takes precedence, but falsy values
    are skipped so callers can opt out of specific fields.  ``allowed_domains``
    entries are normalised to a list of non-empty strings regardless of whether
    they were supplied as comma-separated text, tuples, or lists.
    """

    if not base and not override:
        return None

    merged: Dict[str, Any] = {}

    def _normalise_allowed(val: Any) -> Optional[List[str]]:
        if not val:
            return None
        if isinstance(val, (str, bytes)):
            candidates = [s.strip() for s in str(val).split(",") if s.strip()]
            return candidates or None
        if isinstance(val, Iterable):
            items = [str(item).strip() for item in val if str(item).strip()]
            return items or None
        return None

    for source in (base or {}, override or {}):
        for key, value in source.items():
            if not value:
                continue
            if key == "allowed_domains":
                normalised = _normalise_allowed(value)
                if normalised:
                    merged[key] = normalised
            else:
                merged[key] = value

    return merged or None


def _build_params(
    *,
    model: str,
    input_data: List[Dict[str, Any]],
    max_output_tokens: Optional[int],
    system_instruction: str,
    temperature: float,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    web_search: bool = False,
    web_search_filters: Optional[Dict[str, Any]] = None,
    search_context_size: str = "medium",
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Compose the keyword arguments for an OpenAI Responses API call.

    The function gathers together the many optional features supported by the
    Responses endpoint – such as tool use, web search, JSON formatting and
    reasoning controls – and emits a plain ``dict`` that mirrors the expected
    JSON payload.  ``None`` values are omitted so the underlying SDK can apply
    its own defaults.  This helper keeps the main :func:`get_response` function
    relatively small and easy to read.

    Parameters
    ----------
    model:
        Identifier of the model to query (e.g. ``"gpt-5-mini"``).
    input_data:
        A list representing the conversation so far.  Each element is a mapping
        with ``role`` and ``content`` keys in the format required by the API.
    max_output_tokens:
        Soft cap on the number of tokens the model may generate.  When ``None``
        the parameter is omitted and the model's server-side default applies.
    system_instruction:
        The system prompt used for non ``o``/``gpt-5`` models.  Included here so
        callers can pre-render it alongside the messages.
    temperature:
        Sampling temperature controlling randomness for models that honour it.
    tools, tool_choice:
        Optional tool specifications following the Responses API schema.
    web_search:
        When ``True`` a built-in web search tool is appended to the tool list.
    web_search_filters:
        Optional mapping with keys ``allowed_domains`` and/or any of
        ``city``, ``country``, ``region``, ``timezone`` and ``type``.  ``type``
        should match the Responses API expectation (currently ``"approximate"``
        for geographic hints).  Allowed domains are placed under
        ``filters.allowed_domains`` and location hints under ``user_location``
        to match the Responses API schema.  Keys with falsey values are
        ignored.
    search_context_size:
        Size of the search context when ``web_search`` is enabled.
    json_mode:
        If ``True`` the model is asked to produce structured JSON output.
    expected_schema:
        Optional JSON schema supplied when ``json_mode`` is requested.
    reasoning_effort, reasoning_summary:
        Additional settings for ``o``/``gpt-5`` models controlling hidden
        reasoning tokens and optional summaries.
    **extra:
        Any additional key-value pairs to forward directly to the API.

    Returns
    -------
    dict
        Dictionary ready to be expanded into
        :meth:`openai.AsyncOpenAI.responses.create`.
    """
    params: Dict[str, Any] = {
        "model": model,
        "input": input_data,
        "truncation": "auto",
    }
    if max_output_tokens is not None:
        params["max_output_tokens"] = max_output_tokens
    if json_mode:
        params["text"] = (
            {"format": {"type": "json_schema", "schema": expected_schema}}
            if expected_schema
            else {"format": {"type": "json_object"}}
        )
    all_tools = list(tools) if tools else []
    if web_search:
        tool: Dict[str, Any] = {"type": "web_search", "search_context_size": search_context_size}
        filters = _normalise_web_search_filters(web_search_filters)
        if filters:
            domains = filters.get("filters")
            if domains:
                tool["filters"] = domains
            user_location = filters.get("user_location")
            if user_location:
                tool["user_location"] = user_location
        all_tools.append(tool)
    if all_tools:
        params["tools"] = all_tools
    if tool_choice is not None:
        params["tool_choice"] = tool_choice
    # For o‑series and gpt-5 models, reasoning settings control hidden reasoning
    # tokens and optional summaries. gpt-5 models also ignore the ``temperature``
    # parameter, so we drop it and warn if a custom value was provided. Other
    # models retain temperature-based randomness.
    if model.startswith("o") or model.startswith("gpt-5"):
        reasoning: Dict[str, Any] = {}
        if reasoning_effort is not None:
            reasoning["effort"] = reasoning_effort
        if reasoning_summary is not None:
            reasoning["summary"] = reasoning_summary
        if reasoning:
            params["reasoning"] = reasoning
        if model.startswith("gpt-5") and temperature != 0.9:
            logger.warning(
                f"Model {model} does not support temperature; ignoring provided value."
            )
    else:
        params["temperature"] = temperature
    params.update(extra)
    return params


async def get_response(
    prompt: str,
    *,
    model: str = "gpt-5-mini",
    n: int = 1,
    max_output_tokens: Optional[int] = None,
    # legacy alias for backwards compatibility
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    temperature: float = 0.9,
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    web_search: bool = False,
    web_search_filters: Optional[Dict[str, Any]] = None,
    search_context_size: str = "medium",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    use_dummy: bool = False,
    base_url: Optional[str] = None,
    verbose: bool = True,
    images: Optional[List[str]] = None,
    audio: Optional[List[Dict[str, str]]] = None,
    return_raw: bool = False,
    logging_level: Optional[Union[str, int]] = None,
    background_mode: Optional[bool] = None,
    background_poll_interval: float = 10.0,
    **kwargs: Any,
):
    """Request one or more model completions from the OpenAI API.

    This coroutine is the main entry point for sending a single prompt to an
    OpenAI model.  It supports text-only prompts as well as prompts that include
    images or audio.  When ``use_dummy`` is ``True`` no network requests are
    made; instead a predictable placeholder response is returned, which is
    useful for tests.

    Internally the function prepares a parameter dictionary via
    :func:`_build_params` and dispatches the request using the asynchronous
    OpenAI SDK.  Audio inputs are routed through the chat-completions API,
    whereas all other requests use the newer Responses API.  Multiple
    completions can be retrieved in parallel by setting ``n`` greater than one.

    Parameters
    ----------
    prompt:
        The user question or instruction to send to the model.
    model:
        Name of the model to query.
    n:
        Number of completions to request.  Each completion is retrieved in
        parallel.
    max_output_tokens / max_tokens:
        Optional cap on the length of each completion in tokens.  The
        ``max_tokens`` alias is retained for backwards compatibility, but
        ``max_output_tokens`` takes precedence when both are supplied.
    timeout:
        Maximum time in seconds to wait for the API to respond.  ``None``
        disables client-side timeouts.
    temperature:
        Randomness control for non ``gpt-5`` models.
    json_mode, expected_schema:
        When ``json_mode`` is ``True`` the model is instructed to output JSON.
        ``expected_schema`` may provide a JSON schema to validate against.
    tools, tool_choice:
        Optional tool specifications to pass through to the API.
    web_search, search_context_size:
        Enable and configure the built-in web-search tool.
    web_search_filters:
        Optional mapping with ``allowed_domains`` and/or user location hints
        (``city``, ``country``, ``region``, ``timezone`` and ``type`` – typically
        ``"approximate"``) to guide search results when ``web_search`` is
        enabled.
    reasoning_effort, reasoning_summary:
        Additional reasoning controls for ``o`` and ``gpt-5`` models.
    use_dummy:
        If ``True`` return deterministic dummy responses instead of calling the
        external API.
    base_url:
        Optional custom OpenAI-compatible endpoint. If omitted, the default
        ``api.openai.com/v1`` or ``OPENAI_BASE_URL`` environment variable is
        used.
    verbose:
        When set, progress information is printed via the module logger.
    images, audio:
        Lists of base64-encoded media to include alongside the text prompt.
    return_raw:
        If ``True`` the raw SDK response objects are returned alongside the
        extracted text and timing information.
    logging_level:
        Optional override for the module's log level.
    background_mode:
        When ``True`` the helper submits the request in background mode and
        polls :meth:`openai.AsyncOpenAI.responses.retrieve` until completion.
        When ``None`` (default) the helper automatically enables background
        mode whenever ``timeout`` is ``None`` so long-running calls are resilient
        to transient HTTP disconnects.  Set to ``False`` to force the legacy
        behaviour of waiting on the initial HTTP response.
    background_poll_interval:
        How frequently (in seconds) to poll for background completion when
        background mode is active. Defaults to 10 seconds and automatically
        lengthens when rate-limit responses instruct a longer pause.
    **kwargs:
        Any additional parameters understood by the OpenAI SDK are forwarded
        transparently.

    Returns
    -------
    tuple
        ``([text, ...], duration)`` when ``return_raw`` is ``False``.  If
        ``return_raw`` is ``True`` the third element contains the raw response
        objects from the SDK.
    """
    if web_search_filters and not web_search:
        logger.debug(
            "web_search_filters were supplied but web_search is disabled; ignoring filters."
        )
    if web_search and json_mode:
        logger.warning(
            "Web search cannot be combined with JSON mode; disabling JSON mode."
        )
        json_mode = False
    # Use dummy for testing without calling the API
    if use_dummy:
        dummy = [f"DUMMY {prompt}" for _ in range(max(n, 1))]
        if return_raw:
            return dummy, 0.0, []
        return dummy, 0.0
    if logging_level is not None:
        set_log_level(logging_level)
    _require_api_key()
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    client_async = _get_client(base_url)

    try:
        poll_interval = float(background_poll_interval)
    except (TypeError, ValueError):
        poll_interval = 10.0
    if poll_interval <= 0:
        poll_interval = 10.0

    explicit_background = kwargs.pop("background", None)
    if explicit_background is not None:
        effective_background = bool(explicit_background)
    elif background_mode is not None:
        effective_background = bool(background_mode)
    else:
        effective_background = False
    background_argument: Optional[bool] = None
    if explicit_background is not None:
        background_argument = bool(explicit_background)
    elif effective_background:
        background_argument = True

    failure_statuses = {"failed", "cancelled", "expired"}

    def _background_error_message(resp: Any) -> str:
        err = _safe_get(resp, "error")
        if isinstance(err, dict):
            for key in ("message", "code", "type"):
                val = err.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            try:
                return json.dumps(err, ensure_ascii=False)
            except Exception:
                return str(err)
        if err:
            return str(err)
        status = _safe_get(resp, "status")
        identifier = _safe_get(resp, "id")
        return f"Response {identifier or '<unknown>'} failed with status {status}."

    async def _await_background_completion(
        response_obj: Any,
        start_time: float,
        *,
        poll: Optional[float] = None,
        client: Optional[openai.AsyncOpenAI] = None,
        should_poll: bool = False,
    ) -> Any:
        if not should_poll:
            return response_obj
        status = _safe_get(response_obj, "status")
        if status in (None, "completed"):
            return response_obj
        response_id = _safe_get(response_obj, "id")
        if not response_id:
            return response_obj
        poll_every = poll if poll is not None else poll_interval
        poll_every = max(0.5, float(poll_every))
        local_client = client or client_async
        last = response_obj
        consecutive_errors = 0
        while True:
            status = _safe_get(last, "status")
            if status == "completed":
                return last
            if status in failure_statuses or status == "requires_action":
                message = _background_error_message(last)
                raise APIError(message)
            if status not in {"queued", "in_progress", "cancelling"}:
                return last
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise BackgroundTimeoutError(
                        response_id,
                        last,
                        f"Background response {response_id} exceeded timeout of {timeout} s",
                    )
                sleep_for = min(poll_every, max(0.1, remaining))
            else:
                sleep_for = poll_every
            await asyncio.sleep(sleep_for)
            retrieve_kwargs: Dict[str, Any] = {}
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise BackgroundTimeoutError(
                        response_id,
                        last,
                        f"Background response {response_id} exceeded timeout of {timeout} s",
                    )
                retrieve_kwargs["timeout"] = max(1.0, min(30.0, remaining))
            try:
                last = await local_client.responses.retrieve(response_id, **retrieve_kwargs)
                consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except RateLimitError:
                logger.warning(
                    "[get_response] Polling %s hit rate limit; aborting background checks.",
                    response_id,
                )
                raise
            except Exception as exc:
                consecutive_errors += 1
                logger.warning(
                    "[get_response] Polling %s failed on attempt %d: %r",
                    response_id,
                    consecutive_errors,
                    exc,
                )
                if timeout is not None and (time.time() - start_time) >= timeout:
                    raise BackgroundTimeoutError(
                        response_id,
                        last,
                        f"Background response {response_id} exceeded timeout of {timeout} s",
                    )
                if consecutive_errors >= 5:
                    raise
                continue
    # Derive the effective cutoff
    cutoff = max_output_tokens if max_output_tokens is not None else max_tokens
    # Build system message only for non‑o series
    system_instruction = (
        "Please provide a helpful response to this inquiry for purposes of academic research."
    )
    if audio:
        logger.info(
            "Audio inputs require models gpt-4o-audio-preview, gpt-4o-mini-audio-preview, or gpt-audio"
        )
        contents: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images:
            for img in images:
                img_url = (
                    img if str(img).startswith("data:") else f"data:image/jpeg;base64,{img}"
                )
                contents.append(
                    {"type": "input_image", "image_url": {"url": img_url}}
                )
        for a in audio:
            contents.append({"type": "input_audio", "input_audio": a})
        messages = [{"role": "user", "content": contents}]
        # ``chat.completions`` infers the output modality from the request
        # content.  ``gpt-audio`` requires explicitly requesting text output
        # via ``modalities``; other models default to text when omitted.
        params_chat: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if model == "gpt-audio":
            params_chat["modalities"] = ["text"]
        if tools is not None:
            params_chat["tools"] = tools
        if tool_choice is not None:
            params_chat["tool_choice"] = tool_choice
        if cutoff is not None:
            params_chat["max_completion_tokens"] = cutoff
        params_chat.update(kwargs)
        start = time.time()
        tasks = [
            asyncio.create_task(
                client_async.chat.completions.create(
                    **params_chat, **({"timeout": timeout} if timeout is not None else {})
                )
            )
            for _ in range(max(n, 1))
        ]
        try:
            raw = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        except asyncio.TimeoutError as exc:
            message = (
                f"API call timed out after {timeout} s"
                if timeout is not None
                else "API call timed out"
            )
            logger.error(f"[get_response] {message}")
            raise asyncio.TimeoutError(message) from exc
        except Exception as e:
            logger.error(
                "[get_response] API call resulted in exception: %r", e, exc_info=True
            )
            raise
        texts = []
        for r in raw:
            msg = r.choices[0].message
            parts = getattr(msg, "content", None)
            if isinstance(parts, list):
                texts.append(
                    "".join(p.get("text", "") for p in parts if p.get("type") == "text")
                )
            else:
                texts.append(parts)
        duration = time.time() - start
        if return_raw:
            return texts, duration, raw
        return texts, duration
    else:
        if images:
            contents: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
            for img in images:
                img_url = (
                    img if str(img).startswith("data:") else f"data:image/jpeg;base64,{img}"
                )
                contents.append(
                    {"type": "input_image", "image_url": img_url}
                )
            input_data = (
                [{"role": "user", "content": contents}]
                if model.startswith("o") or model.startswith("gpt-5")
                else [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": contents},
                ]
            )
        else:
            input_data = (
                [{"role": "user", "content": prompt}]
                if model.startswith("o") or model.startswith("gpt-5")
                else [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ]
            )

        params = _build_params(
            model=model,
            input_data=input_data,
            max_output_tokens=cutoff,
            system_instruction=system_instruction,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            web_search=web_search,
            web_search_filters=web_search_filters,
            search_context_size=search_context_size,
            json_mode=json_mode,
            expected_schema=expected_schema,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            **kwargs,
        )
        if background_argument is not None:
            params["background"] = background_argument
        total_needed = max(n, 1)
        start = time.time()
        raw_new: List[Any] = []
        new_tasks: List[asyncio.Task] = [
            asyncio.create_task(
                client_async.responses.create(
                    **params, **({"timeout": timeout} if timeout is not None else {})
                )
            )
            for _ in range(total_needed)
        ]
        try:
            raw_new = await asyncio.gather(*new_tasks)
        except asyncio.CancelledError:
            for t in new_tasks:
                t.cancel()
            raise
        except asyncio.TimeoutError as exc:
            message = (
                f"API call timed out after {timeout} s"
                if timeout is not None
                else "API call timed out"
            )
            logger.error(f"[get_response] {message}")
            raise asyncio.TimeoutError(message) from exc
        except Exception as e:
            logger.error(
                "[get_response] API call resulted in exception: %r", e, exc_info=True
            )
            raise
        def _should_poll_response(resp: Any) -> bool:
            status = _safe_get(resp, "status")
            if status in failure_statuses or status == "requires_action":
                message = _background_error_message(resp)
                raise APIError(message)
            return status not in (None, "completed")

        completed_raw: List[Any] = []
        watcher_tasks: List[asyncio.Task] = []
        completion_durations: List[float] = []

        for response_obj in raw_new:
            needs_poll = _should_poll_response(response_obj)
            if needs_poll:
                watcher_tasks.append(
                    asyncio.create_task(
                        _await_background_completion(
                            response_obj,
                            start,
                            poll=poll_interval,
                            client=client_async,
                            should_poll=True,
                        )
                    )
                )
            else:
                completed_raw.append(response_obj)

        poll_error: Optional[BaseException] = None
        if watcher_tasks:
            try:
                for task in asyncio.as_completed(watcher_tasks):
                    try:
                        result_obj = await task
                        completion_durations.append(max(0.0, time.time() - start))
                        completed_raw.append(result_obj)
                        if len(completed_raw) >= total_needed:
                            break
                    except (BackgroundTimeoutError, RateLimitError, APIError) as exc:
                        poll_error = exc
                        break
                    except Exception as exc:
                        poll_error = exc
                        break
            finally:
                for task in watcher_tasks:
                    if task.done():
                        try:
                            task.result()
                        except Exception:
                            pass
                        continue
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, BackgroundTimeoutError, RateLimitError, APIError):
                        pass
                    except Exception:
                        pass
        if poll_error is not None:
            raise poll_error
        if len(completed_raw) < total_needed:
            raise asyncio.TimeoutError("Background responses did not complete")
        raw = completed_raw[:total_needed]
        # Extract ``output_text`` from the responses.  For Responses API
        # the SDK returns an object with an ``output_text`` attribute.
        texts = [r.output_text for r in raw]
        duration = time.time() - start
        if completion_durations:
            duration = max(duration, max(completion_durations))
        if return_raw:
            return texts, duration, raw
        return texts, duration


def _ser(x: Any) -> Optional[str]:
    """Serialize Python objects deterministically."""
    return None if x is None else json.dumps(x, ensure_ascii=False)


def _de(x: Any) -> Any:
    """Deserialize JSON strings back to Python objects."""
    if pd.isna(x):
        return None
    parsed = safe_json(x)
    return parsed if parsed else None


def response_to_text(value: Any) -> str:
    """Coerce a Response payload into plain text.

    The OpenAI Responses API frequently wraps the textual output in one or
    more layers of lists and dictionaries.  This helper mirrors the extraction
    heuristics used when the responses are first collected, but in a reusable
    form so downstream tasks can reliably access the human-readable text.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        for item in value:
            text = response_to_text(item)
            if text:
                return text
        return ""

    if isinstance(value, dict):
        for key in ("text", "content", "output_text"):
            if key in value:
                text = response_to_text(value.get(key))
                if text:
                    return text
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


async def get_embedding(
    text: str,
    *,
    model: str = "text-embedding-3-small",
    timeout: Optional[float] = None,
    use_dummy: bool = False,
    base_url: Optional[str] = None,
    return_raw: bool = False,
    logging_level: Optional[Union[str, int]] = None,
    **kwargs: Any,
) -> Tuple[List[float], float]:
    """Retrieve a numeric embedding vector for ``text``.

    The OpenAI embedding endpoint converts a piece of text into a list of
    floating‑point numbers that capture semantic meaning.  This helper wraps
    that API in a small asynchronous function and returns both the embedding and
    the time taken to obtain it.  When ``use_dummy`` is ``True`` a synthetic
    embedding is produced instead of contacting the network – handy for unit
    tests or offline experimentation.

    Parameters
    ----------
    text:
        The string to embed.
    model:
        Which embedding model to use.  Defaults to ``"text-embedding-3-small"``.
    timeout:
        Optional request timeout in seconds.  ``None`` waits indefinitely.
    use_dummy:
        Return a short deterministic vector instead of calling the API.
    base_url:
        Optional custom endpoint for the OpenAI-compatible API.
    return_raw:
        When ``True`` the raw SDK response object is also returned.
    logging_level:
        Optional log level override for this call.
    **kwargs:
        Additional parameters forwarded to
        :meth:`openai.AsyncOpenAI.embeddings.create`.

    Returns
    -------
    tuple
        ``(embedding, duration)`` or ``(embedding, duration, raw)`` when
        ``return_raw`` is ``True``.
    """

    if use_dummy:
        dummy = [float(len(text))]
        return (dummy, 0.0, {}) if return_raw else (dummy, 0.0)

    if logging_level is not None:
        set_log_level(logging_level)
    _require_api_key()

    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    client_async = _get_client(base_url)

    start = time.time()
    try:
        raw = await client_async.embeddings.create(
            model=model,
            input=text,
            **({"timeout": timeout} if timeout is not None else {}),
            **kwargs,
        )
    except asyncio.TimeoutError as exc:
        message = (
            f"API call timed out after {timeout} s"
            if timeout is not None
            else "API call timed out"
        )
        logger.error(f"[get_embedding] {message}")
        raise asyncio.TimeoutError(message) from exc
    except APITimeoutError as e:
        logger.error(
            "[get_embedding] API call resulted in client timeout: %r", e, exc_info=True
        )
        raise
    except Exception as e:
        logger.error(
            "[get_embedding] API call resulted in exception: %r", e, exc_info=True
        )
        raise

    embed = raw.data[0].embedding
    duration = time.time() - start
    if return_raw:
        return embed, duration, raw
    return embed, duration


async def get_all_embeddings(
    texts: List[str],
    identifiers: Optional[List[str]] = None,
    *,
    model: str = "text-embedding-3-small",
    save_path: str = "embeddings.pkl",
    reset_file: bool = False,
    n_parallels: int = 150,
    timeout: float = 30.0,
    save_every_x: int = 5000,
    use_dummy: bool = False,
    dummy_embeddings: Optional[Dict[str, List[float]]] = None,
    base_url: Optional[str] = None,
    verbose: bool = True,
    logging_level: Union[str, int] = "warning",
    max_retries: int = 3,
    global_cooldown: int = 15,
    **get_embedding_kwargs: Any,
) -> Dict[str, List[float]]:
    """Compute embeddings for many pieces of text and persist the results.

    The function accepts a list of input strings and queries the OpenAI
    embedding API concurrently.  Progress is periodically written to
    ``save_path`` so long‑running jobs can be resumed.  The routine adapts the
    number of parallel workers based on observed successes and handles common
    failure modes such as timeouts or rate‑limit errors by retrying with an
    exponential backoff.

    Parameters
    ----------
    texts:
        Iterable of strings to embed.
    identifiers:
        Optional identifiers corresponding to ``texts``; defaults to using the
        text itself.  These keys are used when saving and resuming work.
    model:
        Embedding model name.
    save_path:
        File path of a pickle used to store intermediate and final results.
    reset_file:
        When ``True`` any existing ``save_path`` is ignored and overwritten.
    n_parallels:
        Upper bound on the number of concurrent API calls.
    timeout:
        Per‑request timeout in seconds.
    save_every_x:
        Frequency (in processed texts) at which the pickle file is updated.
    use_dummy:
        Generate fake embeddings instead of calling the API.
    dummy_embeddings:
        Optional mapping from identifiers (or ``"*"`` for a fallback) to
        deterministic vectors used when ``use_dummy`` is ``True``.  Supplying
        this allows tests to control the synthetic embeddings instead of
        relying on the default ``[len(text)]`` stub.
    base_url:
        Optional custom OpenAI-compatible endpoint used for requests.
    verbose:
        If ``True`` a progress bar is displayed.
    logging_level:
        Logging verbosity for this helper.
    max_retries:
        Number of times to retry a failed request before giving up.
    global_cooldown:
        Seconds to pause new work after encountering a rate‑limit error.
    **get_embedding_kwargs:
        Additional keyword arguments passed to :func:`get_embedding`.

    Returns
    -------
    dict
        Mapping from identifier to embedding vector.
    """

    if not use_dummy:
        _require_api_key()
    set_log_level(logging_level)
    logger = get_logger(__name__)
    base_url = base_url or os.getenv("OPENAI_BASE_URL")

    if identifiers is None:
        identifiers = texts
    dummy_embeddings_map: Dict[str, List[float]] = {}
    dummy_embedding_default: Optional[List[float]] = None
    if dummy_embeddings:
        for key, value in dummy_embeddings.items():
            if value is None:
                continue
            vector = [float(v) for v in value]
            dummy_embeddings_map[str(key)] = vector
        if "*" in dummy_embeddings_map:
            dummy_embedding_default = list(dummy_embeddings_map["*"])
        elif "__default__" in dummy_embeddings_map:
            dummy_embedding_default = list(dummy_embeddings_map["__default__"])

    save_path = os.path.expanduser(os.path.expandvars(save_path))
    embeddings: Dict[str, List[float]] = {}
    if not reset_file and os.path.exists(save_path):
        try:
            with open(save_path, "rb") as f:
                embeddings = pickle.load(f)
            print(
                f"[get_all_embeddings] Loaded {len(embeddings)} existing embeddings from {save_path}"
            )
        except Exception:
            embeddings = {}

    if len(texts) > 50_000:
        msg = (
            "[get_all_embeddings] Warning: more than 50k texts supplied; the"
            " resulting embeddings file may be very large."
        )
        print(msg)
        logger.warning(msg)

    items = [
        (i, t) for i, t in zip(identifiers, texts) if i not in embeddings
    ]
    if not items:
        print(
            f"[get_all_embeddings] Using cached embeddings from {save_path}; no new texts to process"
        )
        return embeddings

    tokenizer = _get_tokenizer(model)
    get_embedding_kwargs.setdefault("base_url", base_url)
    error_logs: Dict[str, List[str]] = defaultdict(list)
    queue: asyncio.Queue[Tuple[str, str, int]] = asyncio.Queue()
    for item in items:
        queue.put_nowait((item[1], item[0], max_retries))

    processed = 0
    pbar = _progress_bar(
        total=len(items),
        desc="Getting embeddings",
        leave=True,
        verbose=verbose,
    )
    cooldown_until = 0.0
    active_workers = 0
    concurrency_cap = max(1, min(n_parallels, queue.qsize()))
    print(f"[init] Starting with {concurrency_cap} parallel workers")
    logger.info(f"[init] Starting with {concurrency_cap} parallel workers")
    rate_limit_errors_since_adjust = 0
    successes_since_adjust = 0

    def maybe_adjust_concurrency() -> None:
        nonlocal concurrency_cap, rate_limit_errors_since_adjust, successes_since_adjust
        total_events = rate_limit_errors_since_adjust + successes_since_adjust
        if rate_limit_errors_since_adjust > 0:
            min_samples = max(20, int(math.ceil(concurrency_cap * 0.3)))
            if total_events >= min_samples:
                error_ratio = rate_limit_errors_since_adjust / max(1, total_events)
                if error_ratio >= 0.25 or rate_limit_errors_since_adjust >= max(8, int(math.ceil(concurrency_cap * 0.2))):
                    decrement = max(1, int(math.ceil(max(concurrency_cap * 0.15, 1))))
                    new_cap = max(1, concurrency_cap - decrement)
                    if new_cap != concurrency_cap:
                        msg = (
                            f"[scale down] Reducing parallel workers from {concurrency_cap} to {new_cap} due to repeated rate limit errors."
                        )
                        print(msg)
                        logger.warning(msg)
                    concurrency_cap = new_cap
                    rate_limit_errors_since_adjust = 0
                    successes_since_adjust = 0
                    return
        if rate_limit_errors_since_adjust == 0 and concurrency_cap < n_parallels:
            success_threshold = max(15, int(math.ceil(concurrency_cap * 0.75)))
            if successes_since_adjust >= success_threshold:
                increment = max(1, int(math.ceil(max(concurrency_cap * 0.25, 1))))
                new_cap = min(n_parallels, concurrency_cap + increment)
                if new_cap != concurrency_cap:
                    msg = (
                        f"[scale up] Increasing parallel workers from {concurrency_cap} to {new_cap} after sustained success."
                    )
                    print(msg)
                    logger.info(msg)
                concurrency_cap = new_cap
                successes_since_adjust = 0
                rate_limit_errors_since_adjust = 0

    async def worker() -> None:
        nonlocal processed, cooldown_until, active_workers, concurrency_cap
        nonlocal rate_limit_errors_since_adjust, successes_since_adjust
        while True:
            try:
                text, ident, attempts_left = await queue.get()
            except asyncio.CancelledError:
                break
            try:
                now = time.time()
                if now < cooldown_until:
                    await asyncio.sleep(cooldown_until - now)
                while active_workers >= concurrency_cap:
                    await asyncio.sleep(0.01)
                active_workers += 1
                error_logs.setdefault(ident, [])
                call_timeout = timeout
                start = time.time()
                override_embedding: Optional[List[float]] = None
                if use_dummy and (dummy_embeddings_map or dummy_embedding_default):
                    override_embedding = dummy_embeddings_map.get(str(ident))
                    if override_embedding is None:
                        override_embedding = dummy_embedding_default
                    if override_embedding is not None:
                        embeddings[ident] = list(override_embedding)
                        processed += 1
                        successes_since_adjust += 1
                        rate_limit_errors_since_adjust = 0
                        maybe_adjust_concurrency()
                        if processed % save_every_x == 0:
                            with open(save_path, "wb") as f:
                                pickle.dump(embeddings, f)
                        pbar.update(1)
                        continue
                task = asyncio.create_task(
                    get_embedding(
                        text,
                        model=model,
                        timeout=call_timeout,
                        use_dummy=use_dummy,
                        **get_embedding_kwargs,
                    )
                )
                emb, _ = await task
                embeddings[ident] = emb
                processed += 1
                successes_since_adjust += 1
                rate_limit_errors_since_adjust = 0
                maybe_adjust_concurrency()
                if processed % save_every_x == 0:
                    with open(save_path, "wb") as f:
                        pickle.dump(embeddings, f)
                pbar.update(1)
            except (asyncio.TimeoutError, APITimeoutError) as e:
                elapsed = time.time() - start
                if isinstance(e, APITimeoutError):
                    error_message = (
                        f"OpenAI client timed out after {elapsed:.2f} s; "
                        "consider increasing the timeout or reducing concurrency."
                    )
                    detail = str(e)
                    if detail:
                        error_logs[ident].append(detail)
                else:
                    error_message = f"API call timed out after {elapsed:.2f} s"
                error_logs[ident].append(error_message)
                logger.warning(f"Timeout error for {ident}: {error_message}")
                if attempts_left - 1 > 0:
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    await asyncio.sleep(backoff)
                    queue.put_nowait((text, ident, attempts_left - 1))
                else:
                    logger.error(f"[get_all_embeddings] {ident} failed: {e}")
                    processed += 1
                    pbar.update(1)
                    if processed % save_every_x == 0:
                        with open(save_path, "wb") as f:
                            pickle.dump(embeddings, f)
            except RateLimitError as e:
                error_logs[ident].append(str(e))
                logger.warning(f"Rate limit error for {ident}: {e}")
                cooldown_until = time.time() + global_cooldown
                rate_limit_errors_since_adjust += 1
                successes_since_adjust = 0
                maybe_adjust_concurrency()
                if attempts_left - 1 > 0:
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    await asyncio.sleep(backoff)
                    queue.put_nowait((text, ident, attempts_left - 1))
                else:
                    logger.error(f"[get_all_embeddings] {ident} failed: {e}")
                    processed += 1
                    pbar.update(1)
                    if processed % save_every_x == 0:
                        with open(save_path, "wb") as f:
                            pickle.dump(embeddings, f)
            except APIConnectionError as e:
                error_logs[ident].append(str(e))
                logger.warning(f"Connection error for {ident}: {e}")
                if attempts_left - 1 > 0:
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    await asyncio.sleep(backoff)
                    queue.put_nowait((text, ident, attempts_left - 1))
                else:
                    logger.error(f"[get_all_embeddings] {ident} failed: {e}")
                    processed += 1
                    pbar.update(1)
                    if processed % save_every_x == 0:
                        with open(save_path, "wb") as f:
                            pickle.dump(embeddings, f)
            except (
                APIError,
                BadRequestError,
                AuthenticationError,
                InvalidRequestError,
            ) as e:
                error_logs[ident].append(str(e))
                logger.warning(f"API error for {ident}: {e}")
                processed += 1
                pbar.update(1)
                if processed % save_every_x == 0:
                    with open(save_path, "wb") as f:
                        pickle.dump(embeddings, f)
            except Exception as e:
                error_logs[ident].append(str(e))
                logger.error(f"Unexpected error for {ident}: {e}")
                raise
            finally:
                active_workers -= 1
                queue.task_done()

    workers = [
        asyncio.create_task(worker())
        for _ in range(max(1, min(n_parallels, queue.qsize())))
    ]
    try:
        await queue.join()
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Cancellation requested, shutting down workers...")
        raise
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        pbar.close()
        with open(save_path, "wb") as f:
            pickle.dump(embeddings, f)

    return embeddings


def _coerce_to_list(value: Any) -> List[Any]:
    """Return ``value`` as a list while preserving common container types."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Retrieve ``attr`` from ``obj`` supporting dicts and objects uniformly."""

    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _resolve_effective_timeout(
    nonlocal_timeout: float, task_timeout: float, dynamic_timeout: bool
) -> float:
    """Return the timeout that should apply when evaluating in-flight tasks."""

    if not dynamic_timeout:
        return task_timeout
    if math.isinf(task_timeout):
        return nonlocal_timeout
    return task_timeout


def _should_cancel_inflight_task(
    start_time: float,
    now: float,
    nonlocal_timeout: float,
    task_timeout: float,
    dynamic_timeout: bool,
) -> bool:
    """Determine whether an in-flight task should be cancelled for timeout."""

    limit = _resolve_effective_timeout(nonlocal_timeout, task_timeout, dynamic_timeout)
    if math.isinf(limit):
        return False
    return now - start_time > limit


def _normalize_response_result(result: Any) -> Tuple[List[Any], Optional[float], List[Any]]:
    """Normalize outputs from ``response_fn`` into ``(responses, duration, raw)``."""

    responses_obj: Any = None
    duration: Optional[float] = None
    raw_obj: Any = []
    if isinstance(result, dict):
        responses_obj = result.get("responses") or result.get("response")
        duration = result.get("duration")
        raw_obj = result.get("raw", result.get("raw_responses", []))
    elif isinstance(result, tuple):
        seq = list(result)
        responses_obj = seq[0] if seq else None
        if len(seq) >= 2:
            candidate = seq[1]
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                duration = float(candidate)
                tail = seq[2:]
                if len(tail) == 1:
                    raw_obj = tail[0]
                elif tail:
                    raw_obj = tail
            elif candidate is None:
                duration = None
                tail = seq[2:]
                if len(tail) == 1:
                    raw_obj = tail[0]
                elif tail:
                    raw_obj = tail
            else:
                raw_obj = seq[1:]
    else:
        responses_obj = result
    if responses_obj is None:
        responses_obj = [] if isinstance(result, tuple) else result
    responses_list = _coerce_to_list(responses_obj)
    raw_list = _coerce_to_list(raw_obj)
    if duration is not None:
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = None
    if not raw_list or (len(raw_list) == 1 and raw_list[0] is None):
        raw_list = []
    return responses_list, duration, raw_list


def _coerce_dummy_response_spec(
    value: Optional[Union[DummyResponseSpec, Dict[str, Any]]]
) -> Optional[DummyResponseSpec]:
    """Return ``value`` as a :class:`DummyResponseSpec` instance when possible."""

    if value is None:
        return None
    if isinstance(value, DummyResponseSpec):
        return value
    if isinstance(value, dict):
        allowed = {field.name for field in fields(DummyResponseSpec)}
        filtered = {k: v for k, v in value.items() if k in allowed}
        return DummyResponseSpec(**filtered)
    raise TypeError(
        "dummy_responses values must be DummyResponseSpec instances or dictionaries"
    )


def _merge_dummy_specs(
    primary: Optional[DummyResponseSpec], fallback: Optional[DummyResponseSpec]
) -> Optional[DummyResponseSpec]:
    """Combine ``primary`` with ``fallback`` preferring explicit ``primary`` values."""

    if primary is None:
        return fallback
    if fallback is None:
        return primary
    return DummyResponseSpec(
        responses=primary.responses
        if primary.responses is not None
        else fallback.responses,
        duration=primary.duration
        if primary.duration is not None
        else fallback.duration,
        input_tokens=primary.input_tokens
        if primary.input_tokens is not None
        else fallback.input_tokens,
        output_tokens=primary.output_tokens
        if primary.output_tokens is not None
        else fallback.output_tokens,
        reasoning_tokens=primary.reasoning_tokens
        if primary.reasoning_tokens is not None
        else fallback.reasoning_tokens,
        reasoning_summary=primary.reasoning_summary
        if primary.reasoning_summary is not None
        else fallback.reasoning_summary,
        response_id=primary.response_id
        if primary.response_id is not None
        else fallback.response_id,
        successful=primary.successful
        if primary.successful is not None
        else fallback.successful,
        error_log=primary.error_log
        if primary.error_log is not None
        else fallback.error_log,
        warning=primary.warning
        if primary.warning is not None
        else fallback.warning,
    )


def _auto_dummy_usage(prompt: str, responses: List[Any]) -> DummyResponseSpec:
    """Generate a fallback :class:`DummyResponseSpec` based on prompt/response length."""

    approx_in = max(1, _approx_tokens(str(prompt)))
    approx_out = max(
        1,
        sum(max(1, _approx_tokens(str(resp))) for resp in responses) or 1,
    )
    return DummyResponseSpec(
        input_tokens=approx_in,
        output_tokens=approx_out,
        reasoning_tokens=0,
    )


def _listify_error_log(value: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalise ``value`` into a list of human-readable error log entries."""

    if value is None:
        return []
    return [str(item) for item in _coerce_to_list(value)]


def _synthesise_dummy_raw(
    identifier: str, spec: DummyResponseSpec, responses: List[Any]
) -> List[Dict[str, Any]]:
    """Create a faux Responses payload so downstream code sees usage metrics."""

    usage = {
        "input_tokens": int(spec.input_tokens or 0),
        "output_tokens": int(spec.output_tokens or 0),
        "output_tokens_details": {
            "reasoning_tokens": int(spec.reasoning_tokens or 0)
        },
    }
    output_blocks: List[Dict[str, Any]] = []
    if responses:
        content = [
            {"type": "output_text", "text": str(resp)} for resp in responses
        ]
        output_blocks.append(
            {"type": "message", "role": "assistant", "content": content}
        )
    if spec.reasoning_summary:
        output_blocks.append(
            {
                "type": "reasoning",
                "summary": [
                    {"type": "output_text", "text": spec.reasoning_summary}
                ],
            }
        )
    response_id = spec.response_id or f"dummy-{identifier}"
    return [
        {
            "id": response_id,
            "status": "completed" if spec.successful is not False else "failed",
            "output": output_blocks,
            "usage": usage,
        }
    ]

async def get_all_responses(
    prompts: List[str],
    identifiers: Optional[List[str]] = None,
    prompt_images: Optional[Dict[str, List[str]]] = None,
    prompt_audio: Optional[Dict[str, List[Dict[str, str]]]] = None,
    prompt_web_search_filters: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    model: str = "gpt-5-mini",
    n: int = 1,
    max_output_tokens: Optional[int] = None,
    # legacy alias
    max_tokens: Optional[int] = None,
    temperature: float = 0.9,
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    web_search: Optional[bool] = None,
    web_search_filters: Optional[Dict[str, Any]] = None,
    search_context_size: str = "medium",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    dummy_responses: Optional[Dict[str, Union[DummyResponseSpec, Dict[str, Any]]]] = None,
    use_dummy: bool = False,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    print_example_prompt: bool = True,
    save_path: str = "responses.csv",
    reset_files: bool = False,
    # Maximum number of parallel worker tasks to spawn.  This value
    # represents a ceiling; the actual number of concurrent requests
    # will be adjusted downward based on your API rate limits and
    # average prompt length.  See `_print_usage_overview` for more
    # details on how the concurrency cap is calculated.  When web
    # search is enabled the helper automatically lowers this ceiling to
    # one-third of the requested value to avoid overwhelming the search
    # tool.
    n_parallels: int = 750,
    max_retries: int = 3,
    timeout_factor: float = 2.25,
    max_timeout: Optional[float] = None,
    dynamic_timeout: bool = True,
    background_mode: Optional[bool] = None,
    background_poll_interval: float = 2.0,
    # Note: we no longer accept user‑supplied requests_per_minute, tokens_per_minute,
    # dynamic_rate_limit, or rate_limit_factor parameters.  Concurrency is
    # automatically determined from the OpenAI API’s rate‑limit headers and
    # adjusted internally when encountering rate‑limit errors.
    cancel_existing_batch: bool = False,
    use_batch: bool = False,
    batch_completion_window: str = "24h",
    batch_poll_interval: int = 10,
    batch_wait_for_completion: bool = False,
    max_batch_requests: int = 50_000,
    max_batch_file_bytes: int = 100 * 1024 * 1024,
    save_every_x_responses: int = 100,
    verbose: bool = True,
    quiet: bool = False,
    global_cooldown: int = 15,
    rate_limit_window: float = 30.0,
    token_sample_size: int = 20,
    status_report_interval: Optional[float] = 120.0,
    logging_level: Union[str, int] = "warning",
    **get_response_kwargs: Any,
) -> pd.DataFrame:
    """Retrieve model responses for a collection of prompts.

    For each prompt the function contacts the OpenAI API and stores the returned
    text, token counts and timing information in ``save_path``.  It can either
    send requests directly using an asynchronous worker pool or, when
    ``use_batch`` is ``True``, upload the prompts to the OpenAI Batch API and
    periodically poll for completion.  In both modes the helper automatically
    obeys rate limits, retries transient failures with exponential backoff and
    writes partial results to disk so interrupted runs can be resumed.
    The API base URL can be overridden per call via ``base_url`` or globally
    with the ``OPENAI_BASE_URL`` environment variable.

    A dynamic timeout mechanism keeps long‑running jobs efficient: the function
    initially allows unlimited time for each request, then observes how long
    successful responses take and sets a timeout based on the 90th percentile of
    observed durations.  Subsequent calls use this timeout (capped by
    ``max_timeout``) and it is increased if later responses are slower.  Any
    request exceeding the current limit is cancelled and retried.  While the
    timeout is unbounded the helper automatically submits requests in
    background mode and polls for completion so that connections closed by the
    server or networking layer do not strand in-flight prompts.  You can force
    or disable this behaviour with ``background_mode`` and adjust the polling
    cadence via ``background_poll_interval``.

    Concurrency adapts gently to sustained rate‑limit pressure.  A rolling
    window (``rate_limit_window``, default 30 seconds) tracks recent rate‑limit
    errors and only reduces the parallel worker cap when many errors occur
    within that window or when a long streak of consecutive errors is
    observed.  After a reduction the helper waits for another full window
    before scaling down again so brief spikes do not trigger runaway
    throttling, while successful calls reset the counters and allow the pool to
    scale back up.

    Because every prompt that uses web search fans out into additional tool
    calls, the helper automatically lowers the requested ``n_parallels`` to one
    third of its original value whenever ``web_search`` is enabled.  This guard
    reduces the chance of exhausting the search tool’s own quotas and keeps the
    Responses API from being flooded with the much longer prompts that web
    search produces.  You can still request a smaller value manually if needed,
    and the message printed at the start of each run explains the adjustment so
    it can be revisited in the future if the limitation becomes unnecessary.

    Long‑running jobs can also emit periodic status updates.  The
    ``status_report_interval`` parameter controls how frequently the helper
    prints the current concurrency cap, number of active workers, queue size and
    failure counts (default: every five minutes).  Set the interval to ``None``
    or ``0`` to disable these reports.

    The worker pool responds promptly to user cancellation (e.g. pressing
    stop/``Ctrl+C``) by signalling all workers to halt before any new API
    requests are issued.  Transient network disruptions such as lost
    connections are retried with exponential backoff so long‑running jobs can
    resume automatically once connectivity returns.

    For organisations that route prompts through an internal LLM gateway the
    ``response_fn`` parameter exposes a lightweight dependency‑injection
    point.  When provided, the callable is awaited for every prompt instead of
    :func:`get_response`.  Only the keyword arguments accepted by the callable
    are forwarded, allowing simple signatures (e.g. ``async def fn(prompt)``)
    while still supporting advanced features for fully compatible adapters.
    The callable may return a list of responses, a ``(responses, duration)``
    pair, or the traditional ``(responses, duration, raw)`` tuple.  Missing
    duration or raw values simply disable the associated timeout and
    token‑tracking heuristics, keeping the worker resilient to alternative
    backends without forcing callers to mirror the OpenAI API exactly.

    Offline test runs frequently rely on ``use_dummy`` to avoid network calls.
    The optional ``dummy_responses`` mapping refines this mode by letting you
    describe the synthetic payload for each identifier (or ``"*"`` as a
    fallback) via :class:`DummyResponseSpec`.  These specs control the response
    text, duration, token usage, warnings and error logs so tests can exercise
    cost reporting and failure handling paths deterministically.

    The function remains backwards compatible with the original version, except
    that the parameter ``max_tokens`` has been renamed to ``max_output_tokens``.
    When both are provided, ``max_output_tokens`` takes precedence.  The former
    ``use_web_search`` flag is still accepted but ``web_search`` should be used
    going forward.  Additional web search options (allowed domains and user
    location hints such as ``city``, ``country``, ``region``, ``timezone`` and
    ``type`` – usually ``"approximate"``) can be supplied together via
    ``web_search_filters``.  Per-identifier
    overrides can be passed through ``prompt_web_search_filters`` where the
    mapping keys correspond to prompt identifiers and values follow the same
    schema as ``web_search_filters``.  These overrides are merged with the
    global filters before each request, enabling DataFrame-driven location hints
    without hand-crafting separate dictionaries.
"""
    global _USAGE_SHEET_PRINTED
    message_verbose = bool(verbose and not quiet)
    if message_verbose:
        print("Initializing model calls and loading data...")
    if api_key is not None:
        os.environ["OPENAI_API_KEY"] = api_key
    response_callable = response_fn or get_response
    provided_api_key = api_key
    underlying_callable = response_callable
    if isinstance(underlying_callable, functools.partial):
        underlying_callable = underlying_callable.func  # type: ignore[attr-defined]
    try:
        underlying_callable = inspect.unwrap(underlying_callable)  # type: ignore[arg-type]
    except Exception:
        pass
    using_custom_response_fn = response_fn is not None and underlying_callable is not get_response
    manage_rate_limits = not using_custom_response_fn
    if not use_dummy and not using_custom_response_fn:
        _require_api_key()
    set_log_level(logging_level)
    logger = get_logger(__name__)
    _ensure_runtime_dependencies(verbose=message_verbose)
    dataset_stats = _estimate_dataset_stats(prompts)
    cost_estimate = _estimate_cost(
        prompts,
        n,
        max_output_tokens,
        model,
        use_batch,
        sample_size=_ESTIMATION_SAMPLE_SIZE,
    )
    _print_run_banner(
        prompts=prompts,
        model=model,
        n=n,
        use_batch=use_batch,
        estimated_cost=cost_estimate,
        stats=dataset_stats,
        verbose=message_verbose,
    )
    response_param_names: Set[str] = set()
    response_accepts_var_kw = False
    response_accepts_return_raw = False
    prompt_param_kind: Optional[inspect._ParameterKind] = None
    has_generic_positional_slot = False
    has_var_positional = False
    has_var_keyword = False
    prompt_call_via_keyword = False
    try:
        sig = inspect.signature(response_callable)
    except (TypeError, ValueError):
        response_accepts_var_kw = True
        response_accepts_return_raw = True
        prompt_call_via_keyword = False
    else:
        for name, param in sig.parameters.items():
            if name in {"self", "cls"}:
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                has_var_positional = True
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                response_accepts_var_kw = True
                has_var_keyword = True
                continue
            if name == "prompt":
                prompt_param_kind = param.kind
                continue
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                response_param_names.add(name)
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):
                has_generic_positional_slot = True
        response_accepts_return_raw = response_accepts_var_kw or ("return_raw" in response_param_names)
        prompt_can_be_positional = False
        prompt_can_be_keyword = False
        if prompt_param_kind is not None:
            if prompt_param_kind == inspect.Parameter.POSITIONAL_ONLY:
                prompt_can_be_positional = True
            elif prompt_param_kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                prompt_can_be_positional = True
                prompt_can_be_keyword = True
            elif prompt_param_kind == inspect.Parameter.KEYWORD_ONLY:
                prompt_can_be_keyword = True
            elif prompt_param_kind == inspect.Parameter.VAR_POSITIONAL:
                prompt_can_be_positional = True
            elif prompt_param_kind == inspect.Parameter.VAR_KEYWORD:
                prompt_can_be_keyword = True
        else:
            if has_generic_positional_slot or has_var_positional:
                prompt_can_be_positional = True
            if has_var_keyword:
                prompt_can_be_keyword = True
        if not prompt_can_be_positional and not prompt_can_be_keyword:
            raise TypeError(
                "Custom response_fn must accept a `prompt` argument as a positional or keyword parameter."
            )
        prompt_call_via_keyword = prompt_can_be_keyword and not prompt_can_be_positional
    dummy_response_specs: Dict[str, DummyResponseSpec] = {}
    dummy_default_spec: Optional[DummyResponseSpec] = None
    if dummy_responses:
        for key, value in dummy_responses.items():
            spec = _coerce_dummy_response_spec(value)
            if spec is None:
                continue
            dummy_response_specs[str(key)] = spec
        dummy_default_spec = dummy_response_specs.get("*") or dummy_response_specs.get("__default__")
        if not use_dummy:
            logger.warning(
                "`dummy_responses` were provided but `use_dummy` is False; ignoring synthetic payloads."
            )
    else:
        dummy_response_specs = {}
    if status_report_interval is not None:
        try:
            status_report_interval = float(status_report_interval)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid `status_report_interval=%r`; disabling periodic status reports.",
                status_report_interval,
            )
            status_report_interval = None
        else:
            if status_report_interval <= 0:
                status_report_interval = None
    if quiet:
        status_report_interval = None
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    # ``use_web_search`` was the original parameter name; ``web_search`` is the
    # preferred modern spelling.  If both are supplied we favour ``web_search``
    # but emit a warning for awareness.
    legacy_use_web_search = get_response_kwargs.pop("use_web_search", None)
    if web_search is None and legacy_use_web_search is not None:
        web_search = bool(legacy_use_web_search)
    elif legacy_use_web_search is not None and bool(legacy_use_web_search) != web_search:
        logger.warning(
            "`use_web_search` is deprecated; please use `web_search` instead."
        )
    if web_search_filters and not web_search and not get_response_kwargs.get("web_search", False):
        logger.debug(
            "web_search_filters were supplied but web_search is disabled; filters will be ignored."
        )

    if using_custom_response_fn and use_batch:
        logger.warning(
            "Custom response_fn cannot be combined with batch mode; falling back to per-request execution."
        )
        use_batch = False

    if get_response_kwargs.get("web_search", web_search) and get_response_kwargs.get(
        "json_mode", json_mode
    ):
        logger.warning(
            "Web search cannot be combined with JSON mode; disabling JSON mode."
        )
        get_response_kwargs["json_mode"] = False
    # httpx logs a success line for every request at INFO level, which
    # interferes with tqdm's progress display.  Silence these messages
    # so only warnings and errors surface.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    status = StatusTracker()
    requested_n_parallels = max(1, n_parallels)
    user_requested_n_parallels = requested_n_parallels
    tokenizer = _get_tokenizer(model)
    # Backwards compatibility for identifiers
    if identifiers is None:
        identifiers = prompts
    # Pull default values into kwargs for get_response
    get_response_kwargs.setdefault("web_search", web_search)
    if web_search_filters is not None:
        get_response_kwargs.setdefault("web_search_filters", web_search_filters)
    get_response_kwargs.setdefault("search_context_size", search_context_size)
    get_response_kwargs.setdefault("tools", tools)
    get_response_kwargs.setdefault("tool_choice", tool_choice)
    get_response_kwargs.setdefault("json_mode", json_mode)
    get_response_kwargs.setdefault("expected_schema", expected_schema)
    get_response_kwargs.setdefault("temperature", temperature)
    get_response_kwargs.setdefault("reasoning_effort", reasoning_effort)
    get_response_kwargs.setdefault("reasoning_summary", reasoning_summary)
    # Pass the chosen model through to get_response by default
    get_response_kwargs.setdefault("model", model)
    get_response_kwargs.setdefault("base_url", base_url)
    if background_mode is not None:
        get_response_kwargs.setdefault("background_mode", background_mode)
    get_response_kwargs.setdefault("background_poll_interval", background_poll_interval)
    base_web_search_filters = get_response_kwargs.get("web_search_filters")
    web_search_active = bool(get_response_kwargs.get("web_search"))
    web_search_warning_text: Optional[str] = None
    web_search_parallel_note: Optional[str] = None
    if web_search_active:
        web_search_warning_text = (
            "⚠️ Web search is enabled: tool lookups incur extra fees and tokens beyond this estimate, "
            "so actual costs may be significantly higher. Reduce `n_parallels` manually if tool errors occur."
        )
        logger.warning(web_search_warning_text)
    if web_search_active and not use_batch:
        reduced = max(1, requested_n_parallels // 3)
        if reduced < requested_n_parallels:
            requested_n_parallels = reduced
            web_search_parallel_note = (
                f"Web search mode automatically capped parallel workers at {requested_n_parallels} "
                f"(requested {user_requested_n_parallels}) to reduce load on the search tool. "
                "This safeguard helps avoid rate-limit and tool errors and may be relaxed in the future."
            )
            logger.info(web_search_parallel_note)
    web_search_warning_displayed = False
    web_search_note_displayed = False
    # Decide default cutoff once per job using cached rate headers
    # Fetch rate headers once to avoid multiple API calls
    # Retrieve rate‑limit headers for the chosen model.  Passing the model
    # ensures the helper performs a dummy call with the correct model
    # rather than probing the unsupported ``/v1/models`` endpoint.
    rate_headers = (
        _get_rate_limit_headers(model, base_url=base_url)
        if manage_rate_limits
        else {}
    )
    user_cutoff = max_output_tokens if max_output_tokens is not None else max_tokens
    cutoff = (
        _decide_default_max_output_tokens(user_cutoff, rate_headers, base_url=base_url)
        if manage_rate_limits
        else user_cutoff
    )
    get_response_kwargs.setdefault("max_output_tokens", cutoff)
    initial_estimated_output_tokens = (
        cutoff if cutoff is not None else ESTIMATED_OUTPUT_TOKENS_PER_PROMPT
    )
    # Always load or initialise the CSV
    # Expand variables in save_path and ensure the parent directory exists.
    save_path = os.path.expandvars(os.path.expanduser(save_path))
    save_dir = Path(save_path).expanduser().resolve().parent
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.debug("Could not create directory %s", save_dir)
    if reset_files:
        for p in (save_path, save_path + ".batch_state.json"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    csv_header_written = os.path.exists(save_path) and not reset_files and os.path.getsize(save_path) > 0
    if os.path.exists(save_path) and not reset_files:
        if message_verbose:
            print(f"Reading from existing files at {save_path}...")
        df = pd.read_csv(save_path)
        df = df.drop_duplicates(subset=["Identifier"], keep="last")
        df["Response"] = df["Response"].apply(_de)
        if "Error Log" in df.columns:
            df["Error Log"] = df["Error Log"].apply(_de)
        expected_cols = [
            "Input Tokens",
            "Reasoning Tokens",
            "Output Tokens",
            "Reasoning Effort",
            "Successful",
            "Error Log",
        ]
        if reasoning_summary is not None:
            expected_cols.insert(4, "Reasoning Summary")
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA
        if reasoning_summary is None and "Reasoning Summary" in df.columns:
            df = df.drop(columns=["Reasoning Summary"])
        # Only skip identifiers that previously succeeded so failures can be retried
        if "Successful" in df.columns:
            done = set(df.loc[df["Successful"] == True, "Identifier"])
        else:
            done = set(df["Identifier"])
        if message_verbose:
            print(f"Loaded {len(df):,} rows; {len(done):,} already marked complete.")
    else:
        cols = [
            "Identifier",
            "Response",
            "Time Taken",
            "Input Tokens",
            "Reasoning Tokens",
            "Output Tokens",
            "Reasoning Effort",
            "Successful",
            "Error Log",
        ]
        if reasoning_summary is not None:
            cols.insert(7, "Reasoning Summary")
        df = pd.DataFrame(columns=cols)
        done = set()
    written_identifiers: Set[Any] = set(df["Identifier"]) if not df.empty else set()
    # Helper to calculate and report final run cost
    def _report_cost() -> None:
        nonlocal df
        pricing = _lookup_model_pricing(model)
        required_cols = {"Input Tokens", "Output Tokens"}
        if not pricing or not required_cols.issubset(df.columns):
            return
        inp = pd.to_numeric(df["Input Tokens"], errors="coerce").fillna(0)
        out = pd.to_numeric(df["Output Tokens"], errors="coerce").fillna(0)
        if "Reasoning Tokens" in df:
            reason = pd.to_numeric(df["Reasoning Tokens"], errors="coerce").fillna(0)
        else:
            reason = pd.Series([0] * len(df))
        df["Cost"] = (inp / 1_000_000) * pricing["input"] + ((out + reason) / 1_000_000) * pricing["output"]
        total_cost = df["Cost"].sum()
        if len(df) > 0:
            avg_row = total_cost / len(df)
            avg_1000 = avg_row * 1000
        else:
            avg_row = 0.0
            avg_1000 = 0.0
        msg = (
            f"Actual total cost: ${total_cost:.4f}; average per row: ${avg_row:.4f}; average per 1000 rows: ${avg_1000:.4f}"
        )
        if message_verbose:
            print(msg)
        logger.info(msg)
    # Filter prompts/identifiers based on what is already completed
    todo_pairs = [(p, i) for p, i in zip(prompts, identifiers) if i not in done]
    if not todo_pairs:
        _report_cost()
        return df
    if len(todo_pairs) >= 10_000:
        effective_save_every = save_every_x_responses
        if len(todo_pairs) >= 50_000:
            effective_save_every = max(save_every_x_responses, 2000)
        elif len(todo_pairs) >= 20_000:
            effective_save_every = max(save_every_x_responses, 1000)
        elif len(todo_pairs) >= 10_000:
            effective_save_every = max(save_every_x_responses, 500)
        if effective_save_every != save_every_x_responses:
            if message_verbose:
                print(
                    f"Large run detected ({len(todo_pairs):,} rows); autoscaling checkpoint frequency "
                    f"to every {effective_save_every} responses (was {save_every_x_responses})."
                )
            save_every_x_responses = effective_save_every
    status.num_tasks_started = len(todo_pairs)
    status.num_tasks_in_progress = len(todo_pairs)
    if prompt_audio and any(prompt_audio.get(str(i)) for _, i in todo_pairs):
        if use_batch:
            logger.warning(
                "Batch mode is not supported for audio inputs; falling back to non-batch processing."
            )
        use_batch = False
    # Warn the user if the input dataset is very large.  Processing more
    # than 50,000 prompts in a single run can lead to very long execution
    # times and increased risk of rate‑limit throttling.  We still proceed
    # with the run, but advise the user to split the input into smaller
    # batches when possible.
    if len(todo_pairs) > 50_000:
        logger.warning(
            f"You are attempting to process {len(todo_pairs):,} prompts in one go. For better performance and reliability, we recommend splitting jobs into 50k‑row chunks or fewer."
        )
    show_example_prompt = bool(print_example_prompt and not quiet)
    # Print usage summary and example prompt
    if show_example_prompt and todo_pairs:
        # Build prompt list for cost estimate
        prompt_list = [p for p, _ in todo_pairs]
        todo_stats = _estimate_dataset_stats(prompt_list, sample_size=_ESTIMATION_SAMPLE_SIZE)
        if not using_custom_response_fn:
            _print_usage_overview(
                prompts=prompt_list,
                n=n,
                max_output_tokens=cutoff,
                model=model,
                use_batch=use_batch,
                n_parallels=requested_n_parallels,
                verbose=message_verbose,
                rate_headers=rate_headers,
                base_url=base_url,
                web_search_warning=web_search_warning_text,
                web_search_parallel_note=web_search_parallel_note,
                show_static_sections=not _USAGE_SHEET_PRINTED,
                stats=todo_stats,
                sample_size=_ESTIMATION_SAMPLE_SIZE,
            )
            _USAGE_SHEET_PRINTED = True
            if web_search_warning_text:
                web_search_warning_displayed = True
            if web_search_parallel_note:
                web_search_note_displayed = True
        elif message_verbose:
            print(
                "\n===== Job summary ====="
                f"\nNumber of prompts: {len(prompt_list)}"
                f"\nParallel workers (requested): {requested_n_parallels}"
            )
            if web_search_warning_text:
                print(web_search_warning_text)
                web_search_warning_displayed = True
            if web_search_parallel_note:
                print(web_search_parallel_note)
                web_search_note_displayed = True
            logger.info(
                "Skipping OpenAI usage overview because a custom response_fn was supplied."
            )
        example_prompt, _ = todo_pairs[0]
        _display_example_prompt(example_prompt, verbose=message_verbose)
        if not message_verbose:
            logger.info("Example prompt omitted from logs because verbose output is disabled.")
    if message_verbose and web_search_warning_text and not web_search_warning_displayed:
        print(web_search_warning_text)
        web_search_warning_displayed = True
    if message_verbose and web_search_parallel_note and not web_search_note_displayed:
        print(web_search_parallel_note)
        web_search_note_displayed = True
    # Dynamically adjust the maximum number of parallel workers based on rate
    # limits.  We base the concurrency on your API’s per‑minute request and
    # token budgets and the average prompt length.  This calculation only
    # runs once at the start of a non‑batch run.  The resulting value acts
    # as the true upper bound on parallelism; it will be used to size the
    # worker pool and to configure the request/token limiters below.
    max_parallel_ceiling = requested_n_parallels
    concurrency_cap = requested_n_parallels
    allowed_req_pm = max(1, requested_n_parallels)
    estimated_tokens_per_call = max(
        1.0, (initial_estimated_output_tokens + 1) * max(1, n)
    )
    allowed_tok_pm = int(max(1, requested_n_parallels * estimated_tokens_per_call))
    if not use_batch and manage_rate_limits:
        try:
            # Estimate the average number of tokens per call using tiktoken
            # for more accurate gating.  We include the expected output length
            # to ensure that long prompts reduce available parallelism.
            sample_for_tokens = (
                random.sample(todo_pairs, min(len(todo_pairs), _ESTIMATION_SAMPLE_SIZE))
                if len(todo_pairs) > _ESTIMATION_SAMPLE_SIZE
                else todo_pairs
            )
            avg_input_tokens = (
                sum(len(tokenizer.encode(p)) for p, _ in sample_for_tokens)
                / max(1, len(sample_for_tokens))
            )
            gating_output = initial_estimated_output_tokens * OUTPUT_TOKEN_HEADROOM
            tokens_per_call = max(1.0, (avg_input_tokens + gating_output) * max(1, n))
            tokens_per_call *= TOKEN_ESTIMATE_BUFFER

            def _pf(val: Optional[str]) -> Optional[float]:
                try:
                    if val is None:
                        return None
                    s = str(val).strip()
                    if not s:
                        return None
                    f = float(s)
                    return f if f > 0 else None
                except Exception:
                    return None

            lim_r: Optional[float] = None
            rem_r: Optional[float] = None
            lim_t: Optional[float] = None
            rem_t: Optional[float] = None
            if rate_headers:
                lim_r = _pf(rate_headers.get("limit_requests"))
                rem_r = _pf(rate_headers.get("remaining_requests"))
                lim_t = _pf(rate_headers.get("limit_tokens")) or _pf(
                    rate_headers.get("limit_tokens_usage_based")
                )
                rem_t = _pf(rate_headers.get("remaining_tokens")) or _pf(
                    rate_headers.get("remaining_tokens_usage_based")
                )
            def _with_headroom(val: Optional[float], *, buffer: float = RATE_LIMIT_HEADROOM) -> Optional[int]:
                if val is None:
                    return None
                return int(max(1, math.floor(val * buffer)))

            def _select_budget(limit_val: Optional[float], remaining_val: Optional[float]) -> Optional[float]:
                candidates = [v for v in (remaining_val, limit_val) if v is not None and v > 0]
                if not candidates:
                    return None
                return min(candidates)

            def _select_ceiling(limit_val: Optional[float], remaining_val: Optional[float]) -> Optional[float]:
                candidates = [v for v in (limit_val, remaining_val) if v is not None and v > 0]
                if not candidates:
                    return None
                return max(candidates)

            initial_req_budget = _select_budget(lim_r, rem_r)
            initial_tok_budget = _select_budget(lim_t, rem_t)
            ceiling_req = _select_ceiling(lim_r, rem_r)
            ceiling_tok = _select_ceiling(lim_t, rem_t)
            req_budget_for_cap = _with_headroom(initial_req_budget, buffer=PLANNING_RATE_LIMIT_BUFFER)
            tok_budget_for_cap = _with_headroom(initial_tok_budget, buffer=PLANNING_RATE_LIMIT_BUFFER)
            concurrency_candidates = [requested_n_parallels]
            if req_budget_for_cap is not None:
                concurrency_candidates.append(req_budget_for_cap)
            if tok_budget_for_cap is not None:
                concurrency_candidates.append(
                    int(max(1, tok_budget_for_cap // tokens_per_call))
                )
            concurrency_cap = max(1, min(concurrency_candidates))
            ceiling_candidates = [requested_n_parallels]
            ceiling_req_budget = _with_headroom(ceiling_req, buffer=PLANNING_RATE_LIMIT_BUFFER)
            ceiling_tok_budget = _with_headroom(ceiling_tok, buffer=PLANNING_RATE_LIMIT_BUFFER)
            if ceiling_req_budget is not None:
                ceiling_candidates.append(ceiling_req_budget)
            if ceiling_tok_budget is not None:
                ceiling_candidates.append(
                    int(max(1, ceiling_tok_budget // tokens_per_call))
                )
            max_parallel_ceiling = max(1, min(ceiling_candidates))
            if max_parallel_ceiling < concurrency_cap:
                max_parallel_ceiling = concurrency_cap
            if concurrency_cap < requested_n_parallels:
                logger.info(
                    f"[parallel reduction] Limiting parallel workers from {requested_n_parallels} to {concurrency_cap} based on your current rate limits. Consider upgrading your plan for faster processing."
                )
            if ceiling_req_budget is not None:
                allowed_req_pm = ceiling_req_budget
            elif req_budget_for_cap is not None:
                allowed_req_pm = req_budget_for_cap
            else:
                allowed_req_pm = max(1, max_parallel_ceiling)
            if ceiling_tok_budget is not None:
                allowed_tok_pm = ceiling_tok_budget
            elif tok_budget_for_cap is not None:
                allowed_tok_pm = tok_budget_for_cap
            else:
                allowed_tok_pm = int(max(1, max_parallel_ceiling * tokens_per_call))
            estimated_tokens_per_call = tokens_per_call
        except Exception:
            concurrency_cap = max(1, requested_n_parallels)
            max_parallel_ceiling = concurrency_cap
            allowed_req_pm = max(1, requested_n_parallels)
            allowed_tok_pm = int(max(1, requested_n_parallels * estimated_tokens_per_call))
    elif use_batch:
        # In batch mode we don't set concurrency or limiters here; they are
        # handled by the batch API submission.
        allowed_req_pm = 1
        allowed_tok_pm = 1

    # Batch submission path
    if use_batch:
        state_path = save_path + ".batch_state.json"

        # Helper to append batch rows
        def _append_results(rows: List[Dict[str, Any]]) -> None:
            nonlocal df, csv_header_written, written_identifiers
            if not rows:
                return
            batch_df = pd.DataFrame(rows)
            batch_df = batch_df[~batch_df["Identifier"].isin(written_identifiers)]
            if batch_df.empty:
                return
            to_save = batch_df.copy()
            to_save["Response"] = to_save["Response"].apply(_ser)
            to_save["Error Log"] = to_save["Error Log"].apply(_ser)
            to_save.to_csv(
                save_path,
                mode="a" if csv_header_written else "w",
                header=not csv_header_written,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            csv_header_written = True
            if df.empty:
                df = batch_df.reset_index(drop=True)
            else:
                df = pd.concat([df, batch_df], ignore_index=True)
            written_identifiers.update(batch_df["Identifier"])

        client = _get_client(base_url)
        # Load existing state
        if os.path.exists(state_path) and not reset_files:
            with open(state_path, "r") as f:
                state = json.load(f)
        else:
            state = {}
        # Convert single batch format
        if state.get("batch_id"):
            state = {
                "batches": [
                    {
                        "batch_id": state["batch_id"],
                        "input_file_id": state.get("input_file_id"),
                        "total": None,
                        "submitted_at": None,
                    }
                ]
            }
        # Cancel unfinished batches if requested
        if cancel_existing_batch and state.get("batches"):
            logger.info("Cancelling unfinished batch jobs...")
            for b in state["batches"]:
                bid = b.get("batch_id")
                try:
                    await client.batches.cancel(bid)
                    logger.info(f"Cancelled batch {bid}.")
                except Exception as exc:
                    logger.warning(f"Failed to cancel batch {bid}: {exc}")
            try:
                os.remove(state_path)
            except OSError:
                pass
            state = {}
        # If there are no unfinished batches, create new ones
        if not state.get("batches"):
            tasks: List[Dict[str, Any]] = []
            for prompt, ident in todo_pairs:
                imgs = prompt_images.get(str(ident)) if prompt_images else None
                if imgs:
                    contents: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
                    for img in imgs:
                        img_url = img if str(img).startswith("data:") else f"data:image/jpeg;base64,{img}"
                        contents.append({"type": "input_image", "image_url": img_url})
                    input_data = (
                        [{"role": "user", "content": contents}]
                        if (
                            m := get_response_kwargs.get("model", "gpt-5-mini")
                        ).startswith("o")
                        or m.startswith("gpt-5")
                        else [
                            {
                                "role": "system",
                                "content": "Please provide a helpful response to this inquiry for purposes of academic research.",
                            },
                            {"role": "user", "content": contents},
                        ]
                    )
                else:
                    input_data = (
                        [{"role": "user", "content": prompt}]
                        if (
                            m := get_response_kwargs.get("model", "gpt-5-mini")
                        ).startswith("o")
                        or m.startswith("gpt-5")
                        else [
                            {
                                "role": "system",
                                "content": "Please provide a helpful response to this inquiry for purposes of academic research.",
                            },
                            {"role": "user", "content": prompt},
                        ]
                    )
                per_prompt_filters = (
                    prompt_web_search_filters.get(str(ident))
                    if prompt_web_search_filters
                    else None
                )
                merged_filters = _merge_web_search_filters(
                    base_web_search_filters, per_prompt_filters
                )
                body = _build_params(
                    model=get_response_kwargs.get("model", "gpt-5-mini"),
                    input_data=input_data,
                    max_output_tokens=cutoff,
                    system_instruction="Please provide a helpful response to this inquiry for purposes of academic research.",
                    temperature=get_response_kwargs.get("temperature", 0.9),
                    tools=get_response_kwargs.get("tools"),
                    tool_choice=get_response_kwargs.get("tool_choice"),
                    web_search=get_response_kwargs.get("web_search", False),
                    web_search_filters=merged_filters,
                    search_context_size=get_response_kwargs.get(
                        "search_context_size", "medium"
                    ),
                    json_mode=get_response_kwargs.get("json_mode", False),
                    expected_schema=get_response_kwargs.get("expected_schema"),
                    reasoning_effort=get_response_kwargs.get("reasoning_effort"),
                    reasoning_summary=get_response_kwargs.get(
                        "reasoning_summary"
                    ),
                )
                tasks.append(
                    {
                        "custom_id": str(ident),
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": body,
                    }
                )
            if tasks:
                batches: List[List[Dict[str, Any]]] = []
                current_batch: List[Dict[str, Any]] = []
                current_size = 0
                for obj in tasks:
                    line_bytes = (
                        len(json.dumps(obj, ensure_ascii=False).encode("utf-8")) + 1
                    )
                    if (
                        len(current_batch) >= max_batch_requests
                        or current_size + line_bytes > max_batch_file_bytes
                    ):
                        if current_batch:
                            batches.append(current_batch)
                        current_batch = []
                        current_size = 0
                    current_batch.append(obj)
                    current_size += line_bytes
                if current_batch:
                    batches.append(current_batch)
                state["batches"] = []
                for batch_tasks in batches:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".jsonl"
                    ) as tmp:
                        for obj in batch_tasks:
                            tmp.write(json.dumps(obj).encode("utf-8") + b"\n")
                        input_filename = tmp.name
                    uploaded = await client.files.create(
                        file=open(input_filename, "rb"), purpose="batch"
                    )
                    batch = await client.batches.create(
                        input_file_id=uploaded.id,
                        endpoint="/v1/responses",
                        completion_window=batch_completion_window,
                    )
                    state["batches"].append(
                        {
                            "batch_id": batch.id,
                            "input_file_id": uploaded.id,
                            "total": len(batch_tasks),
                            "submitted_at": int(time.time()),
                        }
                    )
                    logger.info(
                        f"Submitted batch {batch.id} with {len(batch_tasks)} requests."
                    )
                with open(state_path, "w") as f:
                    json.dump(state, f)
        # Return immediately if not waiting for completion
        if not batch_wait_for_completion:
            return df
        unfinished_batches: List[Dict[str, Any]] = list(state.get("batches", []))
        completed_rows: List[Dict[str, Any]] = []
        while unfinished_batches:
            for b in list(unfinished_batches):
                bid = b.get("batch_id")
                try:
                    job = await client.batches.retrieve(bid)
                except Exception as exc:
                    logger.warning(f"Failed to retrieve batch {bid}: {exc}")
                    continue
                status = job.status
                if status == "completed":
                    output_file_id = job.output_file_id
                    error_file_id = job.error_file_id
                    logger.info(f"Batch {bid} completed. Downloading results...")
                    try:
                        file_response = await client.files.content(output_file_id)
                    except Exception as exc:
                        logger.warning(
                            f"Failed to download output file for batch {bid}: {exc}"
                        )
                        unfinished_batches.remove(b)
                        continue
                    # Normalize file response to plain text
                    text_data: Optional[str] = None
                    try:
                        if isinstance(file_response, str):
                            text_data = file_response
                        elif isinstance(file_response, bytes):
                            text_data = file_response.decode("utf-8", errors="replace")
                        elif hasattr(file_response, "text"):
                            attr = getattr(file_response, "text")
                            text_data = await attr() if callable(attr) else attr  # type: ignore
                        if text_data is None and hasattr(file_response, "read"):
                            content_bytes = await file_response.read()  # type: ignore
                            text_data = (
                                content_bytes.decode("utf-8", errors="replace")
                                if isinstance(content_bytes, bytes)
                                else str(content_bytes)
                            )
                    except Exception:
                        pass
                    if text_data is None:
                        logger.warning(f"No data found in output file for batch {bid}.")
                        unfinished_batches.remove(b)
                        continue
                    errors: Dict[str, Any] = {}
                    if error_file_id:
                        try:
                            err_response = await client.files.content(error_file_id)
                        except Exception as exc:
                            logger.warning(
                                f"Failed to download error file for batch {bid}: {exc}"
                            )
                            err_response = None
                        if err_response is not None:
                            err_text: Optional[str] = None
                            try:
                                if isinstance(err_response, str):
                                    err_text = err_response
                                elif isinstance(err_response, bytes):
                                    err_text = err_response.decode(
                                        "utf-8", errors="replace"
                                    )
                                elif hasattr(err_response, "text"):
                                    attr = getattr(err_response, "text")
                                    err_text = await attr() if callable(attr) else attr  # type: ignore
                                if err_text is None and hasattr(err_response, "read"):
                                    content_bytes = await err_response.read()  # type: ignore
                                    err_text = (
                                        content_bytes.decode("utf-8", errors="replace")
                                        if isinstance(content_bytes, bytes)
                                        else str(content_bytes)
                                    )
                            except Exception:
                                err_text = None
                            if err_text:
                                for line in err_text.splitlines():
                                    try:
                                        rec = json.loads(line)
                                        errors[rec.get("custom_id")] = rec.get("error")
                                    except Exception:
                                        pass
                    for line in text_data.splitlines():
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        ident = rec.get("custom_id")
                        if not ident:
                            continue
                        if rec.get("response") is None:
                            err = rec.get("error") or errors.get(ident)
                            row = {
                                "Identifier": ident,
                                "Response": None,
                                "Time Taken": None,
                                "Input Tokens": None,
                                "Reasoning Tokens": None,
                                "Output Tokens": None,
                                "Reasoning Effort": get_response_kwargs.get(
                                    "reasoning_effort", reasoning_effort
                                ),
                                "Successful": False,
                                "Error Log": [err] if err else [],
                            }
                            if reasoning_summary is not None:
                                row["Reasoning Summary"] = None
                            completed_rows.append(row)
                            continue
                        resp_obj = rec["response"]
                        resp_text: Optional[str] = None
                        summary_text: Optional[str] = None
                        usage = {}
                        if isinstance(resp_obj, dict):
                            usage = resp_obj.get("usage", {}) or {}
                        input_tok = usage.get("input_tokens") if isinstance(usage, dict) else None
                        output_tok = usage.get("output_tokens") if isinstance(usage, dict) else None
                        reason_tok = None
                        if isinstance(usage, dict):
                            otd = usage.get("output_tokens_details") or {}
                            if isinstance(otd, dict):
                                reason_tok = otd.get("reasoning_tokens")
                        # Determine candidate payload
                        candidate = (
                            resp_obj.get("body", resp_obj)
                            if isinstance(resp_obj, dict)
                            else None
                        )
                        search_objs: List[Dict[str, Any]] = []
                        if isinstance(candidate, dict):
                            search_objs.append(candidate)
                        if isinstance(resp_obj, dict):
                            search_objs.append(resp_obj)
                        for obj in search_objs:
                            if resp_text is None and isinstance(
                                obj.get("output_text"), (str, bytes)
                            ):
                                resp_text = obj["output_text"]
                                break
                            if resp_text is None and isinstance(
                                obj.get("choices"), list
                            ):
                                choices = obj.get("choices")
                                if choices:
                                    choice = choices[0]
                                    if isinstance(choice, dict):
                                        msg = (
                                            choice.get("message")
                                            or choice.get("delta")
                                            or {}
                                        )
                                        if isinstance(msg, dict):
                                            content = msg.get("content")
                                            if isinstance(content, str):
                                                resp_text = content
                                                break
                                    if resp_text is None and isinstance(
                                        obj.get("output"), list
                                    ):
                                        out_list = obj.get("output")
                                        for item in out_list:
                                            if not isinstance(item, dict):
                                                continue
                                            content_list = item.get("content")
                                            if isinstance(content_list, list):
                                                for piece in content_list:
                                                    if (
                                                        isinstance(piece, dict)
                                                        and "text" in piece
                                                    ):
                                                        txt = piece.get("text")
                                                        if isinstance(txt, str):
                                                            resp_text = txt
                                                            break
                                                if resp_text is not None:
                                                    break
                                            if resp_text is None and isinstance(
                                                item.get("text"), str
                                            ):
                                                resp_text = item["text"]
                                                break
                                            if resp_text is not None:
                                                break
                                        if resp_text is not None:
                                            break
                        for obj in search_objs:
                            out_list = obj.get("output")
                            if isinstance(out_list, list):
                                for piece in out_list:
                                    if (
                                        isinstance(piece, dict)
                                        and piece.get("type") == "reasoning"
                                    ):
                                        summ = piece.get("summary")
                                        if isinstance(summ, list) and summ:
                                            first = summ[0]
                                            if isinstance(first, dict):
                                                txt = first.get("text")
                                                if isinstance(txt, str):
                                                    summary_text = txt
                                                    break
                                if summary_text is not None:
                                    break
                        row = {
                            "Identifier": ident,
                            "Response": [resp_text],
                            "Time Taken": None,
                            "Input Tokens": input_tok,
                            "Reasoning Tokens": reason_tok,
                            "Output Tokens": output_tok,
                            "Reasoning Effort": get_response_kwargs.get(
                                "reasoning_effort", reasoning_effort
                            ),
                            "Successful": True,
                            "Error Log": [],
                        }
                        if reasoning_summary is not None:
                            row["Reasoning Summary"] = summary_text
                        completed_rows.append(row)
                    unfinished_batches.remove(b)
                    state["batches"] = [
                        bb
                        for bb in state.get("batches", [])
                        if bb.get("batch_id") != bid
                    ]
                    with open(state_path, "w") as f:
                        json.dump(state, f)
                elif status in {"failed", "cancelled", "expired"}:
                    logger.warning(f"Batch {bid} finished with status {status}.")
                    unfinished_batches.remove(b)
                    state["batches"] = [
                        bb
                        for bb in state.get("batches", [])
                        if bb.get("batch_id") != bid
                    ]
                    with open(state_path, "w") as f:
                        json.dump(state, f)
                else:
                    rc = job.request_counts
                    logger.info(
                        f"Batch {bid} in progress: {status}; completed {rc.completed}/{rc.total}."
                    )
            if unfinished_batches:
                await asyncio.sleep(batch_poll_interval)
        # Append and return
        _append_results(completed_rows)
        _report_cost()
        return df
    # Non‑batch path
    # Initialise limiters using the per‑minute budgets derived above.  These
    # limiters control the rate of API requests and the number of tokens
    # consumed per minute.  By setting the budgets based on your account’s
    # remaining limits (or sensible defaults when limits are unknown), we
    # ensure that tasks yield gracefully when the budget is exhausted rather
    # than overrunning the API’s quota.  We do not apply any dynamic
    # scaling factor here; concurrency has already been capped based on
    # the budgets and average prompt length.
    max_timeout_val = float("inf") if max_timeout is None else float(max_timeout)
    nonlocal_timeout: float = float("inf") if dynamic_timeout else max_timeout_val
    req_lim: Optional[AsyncLimiter] = None
    tok_lim: Optional[AsyncLimiter] = None
    if not use_batch and manage_rate_limits:
        req_lim = AsyncLimiter(allowed_req_pm, 60)
        tok_lim = AsyncLimiter(allowed_tok_pm, 60)
    success_times: List[float] = []
    timeout_initialized = False
    observed_latency_p90 = float("inf")
    inflight: Dict[str, Tuple[float, asyncio.Task, float]] = {}
    error_logs: Dict[str, List[str]] = defaultdict(list)
    call_count = 0
    samples_for_timeout = max(
        1,
        int(
            0.90
            * min(
                len(todo_pairs),
                max_parallel_ceiling,
                requested_n_parallels,
            )
        ),
    )
    queue: asyncio.Queue[Tuple[str, str, int]] = asyncio.Queue()
    for item in todo_pairs:
        queue.put_nowait((item[0], item[1], max_retries))
    results: List[Dict[str, Any]] = []
    processed = 0
    pbar = _progress_bar(
        total=len(todo_pairs),
        desc="Processing prompts",
        leave=True,
        verbose=verbose,
    )
    cooldown_until = 0.0
    stop_event = asyncio.Event()
    timeout_cancellations: Set[str] = set()
    # Counters used for the gentle concurrency adaptation below
    rate_limit_errors_since_adjust = 0
    successes_since_adjust = 0
    active_workers = 0
    rate_limit_window = max(1.0, float(rate_limit_window))
    rate_limit_error_times: Deque[float] = deque()
    last_concurrency_scale_down = 0.0
    last_concurrency_scale_up = 0.0
    usage_samples: List[Tuple[int, int, int]] = []
    estimated_output_tokens = initial_estimated_output_tokens
    limiter_wait_durations: Deque[float] = deque(maxlen=max(10, token_sample_size))
    limiter_wait_ratios: Deque[float] = deque(maxlen=max(10, token_sample_size))
    limiter_wait_ratio_threshold = 0.35
    limiter_wait_duration_threshold = 0.6
    token_adjust_cooldown = max(25.0, rate_limit_window)
    token_adjust_eval_interval = max(5.0, rate_limit_window * 0.25)
    last_token_adjust = 0.0
    last_token_adjust_eval = 0.0
    last_token_adjust_log = 0.0
    token_adjust_log_cooldown = 20.0
    token_adjust_min_delta = max(1, int(math.ceil(max_parallel_ceiling * 0.05)))

    def _aggregate_usage(raw_items: List[Any]) -> Tuple[int, int, int]:
        total_in = total_out = total_reason = 0
        for item in raw_items:
            usage = _safe_get(item, "usage")
            if usage is None:
                continue
            input_tokens = _safe_get(usage, "input_tokens")
            if input_tokens in (None, 0):
                input_tokens = _safe_get(usage, "prompt_tokens")
            output_tokens = _safe_get(usage, "output_tokens")
            if output_tokens in (None, 0):
                output_tokens = _safe_get(usage, "completion_tokens")
            details = _safe_get(usage, "output_tokens_details")
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens") or 0
            else:
                reasoning_tokens = _safe_get(details, "reasoning_tokens", 0)
            try:
                total_in += int(input_tokens or 0)
            except Exception:
                pass
            try:
                total_out += int(output_tokens or 0)
            except Exception:
                pass
            try:
                total_reason += int(reasoning_tokens or 0)
            except Exception:
                pass
        return total_in, total_out, total_reason

    def emit_parallelization_status(reason: str, *, force: bool = False) -> None:
        """Print and log a snapshot of the current worker utilisation."""

        if not force and status_report_interval is None and not verbose:
            return
        if quiet and not force:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        msg = (
            f"[parallelization] {timestamp} | {reason}: "
            f"cap={concurrency_cap}, active={active_workers}, inflight={len(inflight)}, "
            f"queue={queue.qsize()}, processed={processed}/{status.num_tasks_started}, "
            f"rate_limit_errors={status.num_rate_limit_errors}"
        )
        if message_verbose:
            print(msg)
        logger.info(msg)

    emit_parallelization_status("Initial parallelization settings", force=True)

    async def flush() -> None:
        nonlocal results, df, processed, csv_header_written, written_identifiers
        if results:
            batch_df = pd.DataFrame(results)
            batch_df = batch_df[~batch_df["Identifier"].isin(written_identifiers)]
            if not batch_df.empty:
                to_save = batch_df.copy()
                to_save["Response"] = to_save["Response"].apply(_ser)
                to_save["Error Log"] = to_save["Error Log"].apply(_ser)
                to_save.to_csv(
                    save_path,
                    mode="a" if csv_header_written else "w",
                    header=not csv_header_written,
                    index=False,
                    quoting=csv.QUOTE_MINIMAL,
                )
                csv_header_written = True
                if df.empty:
                    df = batch_df.reset_index(drop=True)
                else:
                    df = pd.concat([df, batch_df], ignore_index=True)
                written_identifiers.update(batch_df["Identifier"])
            results = []
        if logger.isEnabledFor(logging.INFO) and processed:
            logger.info(
                f"Processed {processed}/{status.num_tasks_started} prompts; "
                f"failures: {status.num_tasks_failed} "
                f"(timeouts: {status.num_timeout_errors}, "
                f"rate limits: {status.num_rate_limit_errors}, "
                f"API: {status.num_api_errors}, other: {status.num_other_errors})"
            )

    async def adjust_timeout() -> None:
        nonlocal nonlocal_timeout, timeout_initialized, observed_latency_p90
        if not dynamic_timeout:
            return
        if len(success_times) < samples_for_timeout:
            return
        try:
            p90 = float(np.percentile(success_times, 90))
            observed_latency_p90 = p90
            new_timeout = min(max_timeout_val, timeout_factor * p90)
            if math.isinf(nonlocal_timeout):
                nonlocal_timeout = new_timeout
                if not timeout_initialized:
                    timeout_display = (
                        "inf"
                        if math.isinf(nonlocal_timeout)
                        else f"{nonlocal_timeout:.1f}s"
                    )
                    p90_display = (
                        "inf" if math.isinf(p90) else f"{p90:.1f}s"
                    )
                    msg = (
                        "[dynamic timeout] Initialized timeout to "
                        f"{timeout_display} (p90={p90_display}, factor={timeout_factor:.2f})."
                    )
                    print(msg)
                    logger.info(msg)
                    timeout_initialized = True
            elif new_timeout > nonlocal_timeout:
                p90_display = "inf" if math.isinf(p90) else f"{p90:.1f}s"
                logger.debug(
                    "[dynamic timeout] Updating timeout to %s (p90=%s, factor=%.2f).",
                    "inf" if math.isinf(new_timeout) else f"{new_timeout:.1f}s",
                    p90_display,
                    timeout_factor,
                )
                nonlocal_timeout = new_timeout
            if not math.isinf(nonlocal_timeout):
                now = time.time()
                for ident, (start, task, t_out) in list(inflight.items()):
                    limit = _resolve_effective_timeout(
                        nonlocal_timeout, t_out, dynamic_timeout
                    )
                    if now - start > limit and not task.done():
                        timeout_cancellations.add(ident)
                        task.cancel()
        except Exception:
            pass

    # The per‑minute AsyncLimiter budgets remain fixed.  When rate limits are
    # hit we only adapt the number of in‑flight worker tasks without
    # rebuilding the limiters themselves, keeping the gating logic simple.
    async def rebuild_limiters() -> None:
        return None

    def maybe_adjust_concurrency() -> None:
        nonlocal concurrency_cap, rate_limit_errors_since_adjust, successes_since_adjust, max_parallel_ceiling, last_concurrency_scale_down, last_concurrency_scale_up
        if not manage_rate_limits:
            return
        now = time.time()
        window_start = now - rate_limit_window
        while rate_limit_error_times and rate_limit_error_times[0] < window_start:
            rate_limit_error_times.popleft()
        recent_errors = len(rate_limit_error_times)
        error_window_threshold = max(6, int(math.ceil(concurrency_cap * 0.10)))
        consecutive_threshold = max(3, int(math.ceil(concurrency_cap * 0.08)))
        should_scale_down = False
        if recent_errors >= error_window_threshold:
            should_scale_down = True
        elif rate_limit_errors_since_adjust >= consecutive_threshold:
            should_scale_down = True
        if should_scale_down and (now - last_concurrency_scale_down) >= rate_limit_window:
            decrement = max(1, int(math.ceil(max(concurrency_cap * 0.35, 2))))
            new_cap = max(1, concurrency_cap - decrement)
            if new_cap != concurrency_cap:
                old_cap = concurrency_cap
                concurrency_cap = new_cap
                reason = (
                    f"[scale down] Reducing parallel workers from {old_cap} to {new_cap} "
                    f"after {recent_errors} rate limit errors in the last {int(round(rate_limit_window))}s."
                )
                logger.warning(reason)
                emit_parallelization_status(reason, force=True)
            else:
                concurrency_cap = new_cap
            rate_limit_errors_since_adjust = 0
            successes_since_adjust = 0
            rate_limit_error_times.clear()
            last_concurrency_scale_down = now
            return
        quiet_since_last_error = (now - status.time_of_last_rate_limit_error) >= rate_limit_window
        if (
            rate_limit_errors_since_adjust == 0
            and concurrency_cap < max_parallel_ceiling
            and quiet_since_last_error
            and (now - last_concurrency_scale_down) >= rate_limit_window
            and (now - last_concurrency_scale_up) >= rate_limit_window
        ):
            success_threshold = max(40, int(math.ceil(concurrency_cap * 1.5)))
            if successes_since_adjust >= success_threshold:
                increment = max(1, int(math.ceil(max(concurrency_cap * 0.08, 1))))
                new_cap = min(max_parallel_ceiling, concurrency_cap + increment)
                if new_cap != concurrency_cap:
                    old_cap = concurrency_cap
                    concurrency_cap = new_cap
                    reason = (
                        f"[scale up] Increasing parallel workers from {old_cap} to {new_cap} after sustained success."
                    )
                    logger.info(reason)
                    emit_parallelization_status(reason, force=True)
                    last_concurrency_scale_up = now
                else:
                    concurrency_cap = new_cap
                successes_since_adjust = 0
                rate_limit_errors_since_adjust = 0

    async def worker() -> None:
        nonlocal processed, call_count, nonlocal_timeout, active_workers, concurrency_cap, cooldown_until, estimated_output_tokens, rate_limit_errors_since_adjust, successes_since_adjust, stop_event, max_parallel_ceiling, last_token_adjust
        while True:
            if stop_event.is_set():
                break
            try:
                prompt, ident, attempts_left = await queue.get()
            except asyncio.CancelledError:
                break
            if stop_event.is_set():
                queue.task_done()
                break
            try:
                now = time.time()
                if now < cooldown_until:
                    await asyncio.sleep(cooldown_until - now)
                while active_workers >= concurrency_cap:
                    await asyncio.sleep(0.01)
                active_workers += 1
                input_tokens = len(tokenizer.encode(prompt))
                gating_output = estimated_output_tokens
                limiter_wait_time = 0.0
                if req_lim is not None:
                    wait_start = time.perf_counter()
                    await req_lim.acquire()
                    limiter_wait_time += time.perf_counter() - wait_start
                if tok_lim is not None:
                    wait_start = time.perf_counter()
                    await tok_lim.acquire((input_tokens + gating_output) * n)
                    limiter_wait_time += time.perf_counter() - wait_start
                call_count += 1
                error_logs.setdefault(ident, [])
                start = time.time()
                base_timeout = nonlocal_timeout
                multiplier = 1.5 ** (max_retries - attempts_left)
                call_timeout = None if math.isinf(base_timeout) else base_timeout * multiplier
                if call_timeout is not None and dynamic_timeout and not math.isinf(max_timeout_val):
                    call_timeout = min(call_timeout, max_timeout_val)
                images_payload = prompt_images.get(str(ident)) if prompt_images else None
                audio_payload = prompt_audio.get(str(ident)) if prompt_audio else None
                call_kwargs = dict(get_response_kwargs)
                per_prompt_filters = (
                    prompt_web_search_filters.get(str(ident))
                    if prompt_web_search_filters
                    else None
                )
                merged_filters = _merge_web_search_filters(
                    base_web_search_filters, per_prompt_filters
                )
                if merged_filters is not None:
                    call_kwargs["web_search_filters"] = merged_filters
                else:
                    call_kwargs.pop("web_search_filters", None)
                if images_payload is not None:
                    call_kwargs["images"] = images_payload
                if audio_payload is not None:
                    call_kwargs["audio"] = audio_payload
                call_kwargs.update(
                    {
                        "n": n,
                        "timeout": call_timeout,
                        "use_dummy": use_dummy,
                    }
                )
                if using_custom_response_fn and provided_api_key is not None:
                    call_kwargs.setdefault("api_key", provided_api_key)
                if response_accepts_return_raw:
                    call_kwargs.setdefault("return_raw", True)
                else:
                    call_kwargs.pop("return_raw", None)
                if not response_accepts_var_kw:
                    call_kwargs = {
                        k: v for k, v in call_kwargs.items() if k in response_param_names
                    }
                if prompt_call_via_keyword:
                    call_kwargs = dict(call_kwargs)
                    call_kwargs["prompt"] = prompt
                    call_args = ()
                else:
                    call_args = (prompt,)
                task = asyncio.create_task(response_callable(*call_args, **call_kwargs))
                inflight[ident] = (
                    start,
                    task,
                    call_timeout if call_timeout is not None else float("inf"),
                )
                response_ids: List[str] = []
                try:
                    result = await task
                except asyncio.CancelledError:
                    inflight.pop(ident, None)
                    if ident in timeout_cancellations:
                        timeout_cancellations.discard(ident)
                        raise asyncio.TimeoutError(
                            f"API call timed out after {call_timeout} s"
                        )
                    raise
                inflight.pop(ident, None)
                resps, duration, raw = _normalize_response_result(result)
                success_override: Optional[bool] = None
                if use_dummy:
                    selected_spec = dummy_response_specs.get(str(ident))
                    if selected_spec is None:
                        selected_spec = dummy_default_spec
                    auto_spec: Optional[DummyResponseSpec] = None
                    if not raw:
                        auto_spec = _auto_dummy_usage(prompt, resps)
                    selected_spec = _merge_dummy_specs(selected_spec, auto_spec)
                    if selected_spec is not None:
                        override_responses = selected_spec.responses
                        if override_responses is not None:
                            resps = _coerce_to_list(override_responses)
                        if selected_spec.duration is not None:
                            duration = selected_spec.duration
                        if selected_spec.warning:
                            logger.warning(selected_spec.warning)
                        extra_errors = _listify_error_log(selected_spec.error_log)
                        if extra_errors:
                            error_logs[ident].extend(extra_errors)
                        raw = _synthesise_dummy_raw(str(ident), selected_spec, resps)
                        success_override = selected_spec.successful
                if duration is not None:
                    success_times.append(duration)
                await adjust_timeout()
                limiter_wait_ratio = 0.0
                if (
                    limiter_wait_time > 0
                    and duration is not None
                    and duration > 0
                ):
                    limiter_wait_ratio = min(
                        1.0, limiter_wait_time / max(duration, 1e-6)
                    )
                limiter_wait_durations.append(limiter_wait_time)
                limiter_wait_ratios.append(limiter_wait_ratio)
                total_input, total_output, total_reasoning = _aggregate_usage(raw)
                for item in _coerce_to_list(raw):
                    rid = _safe_get(item, "id")
                    if rid:
                        response_ids.append(rid)
                summary_text = None
                try:
                    for r in raw:
                        out_items = _coerce_to_list(_safe_get(r, "output", []))
                        if not out_items:
                            continue
                        for item in out_items:
                            if _safe_get(item, "type") == "reasoning":
                                summary_list = _coerce_to_list(
                                    _safe_get(item, "summary", [])
                                )
                                if summary_list:
                                    txt = _safe_get(summary_list[0], "text")
                                    if isinstance(txt, str):
                                        summary_text = txt
                                        break
                        if summary_text is not None:
                            break
                except Exception:
                    summary_text = None
                usage_samples.append((total_input, total_output, total_reasoning))
                if len(usage_samples) > token_sample_size:
                    usage_samples.pop(0)
                if manage_rate_limits and len(usage_samples) >= token_sample_size:
                    now = time.time()
                    if (now - last_token_adjust_eval) >= token_adjust_eval_interval:
                        last_token_adjust_eval = now
                        avg_in = statistics.mean(u[0] for u in usage_samples)
                        avg_out = statistics.mean(u[1] for u in usage_samples)
                        avg_reason = statistics.mean(u[2] for u in usage_samples)
                        observed_output = avg_out + avg_reason
                        if observed_output > 0:
                            estimated_output_tokens = max(
                                estimated_output_tokens, observed_output * OUTPUT_TOKEN_HEADROOM
                            )
                        tokens_per_call_est = (
                            avg_in + max(estimated_output_tokens, observed_output)
                        ) * max(1, n)
                        token_limited = int(
                            max(1, allowed_tok_pm // max(1, tokens_per_call_est))
                        )
                        req_limited = int(max(1, allowed_req_pm))
                        new_cap = min(max_parallel_ceiling, req_limited, token_limited)
                        if new_cap < 1:
                            new_cap = 1
                        safe_to_increase = (now - status.time_of_last_rate_limit_error) >= rate_limit_window
                        if new_cap > concurrency_cap and safe_to_increase:
                            max_increase = max(2, int(math.ceil(concurrency_cap * 0.15)))
                            new_cap = min(new_cap, concurrency_cap + max_increase)
                        limiter_pressure = False
                        if (
                            limiter_wait_ratio >= limiter_wait_ratio_threshold
                            or limiter_wait_time >= limiter_wait_duration_threshold
                        ):
                            limiter_pressure = True
                        else:
                            sample_count = len(limiter_wait_durations)
                            min_samples = max(8, min(token_sample_size, 24))
                            if sample_count >= min_samples:
                                try:
                                    avg_ratio = statistics.mean(limiter_wait_ratios)
                                    avg_wait = statistics.mean(limiter_wait_durations)
                                except statistics.StatisticsError:
                                    avg_ratio = 0.0
                                    avg_wait = 0.0
                                high_ratio_events = sum(
                                    1
                                    for r in limiter_wait_ratios
                                    if r >= limiter_wait_ratio_threshold
                                )
                                high_wait_events = sum(
                                    1
                                    for d in limiter_wait_durations
                                    if d >= limiter_wait_duration_threshold
                                )
                                limiter_pressure = (
                                    avg_ratio >= limiter_wait_ratio_threshold
                                    or avg_wait >= limiter_wait_duration_threshold
                                    or high_ratio_events >= max(4, math.ceil(sample_count * 0.45))
                                    or high_wait_events >= max(4, math.ceil(sample_count * 0.45))
                                )
                        change = abs(new_cap - concurrency_cap)
                        recently_adjusted = (now - last_token_adjust) < token_adjust_cooldown
                        if new_cap > concurrency_cap and change < token_adjust_min_delta and not limiter_pressure:
                            # allow small upward nudges even during cooldown, but only when healthy
                            new_cap = concurrency_cap if recently_adjusted else min(concurrency_cap + 1, max_parallel_ceiling)
                        if new_cap < concurrency_cap:
                            # soften downward moves for limiter-driven path
                            max_drop = max(1, int(math.ceil(concurrency_cap * 0.20)))
                            new_cap = max(new_cap, concurrency_cap - max_drop)
                            if not limiter_pressure or (recently_adjusted and change < token_adjust_min_delta):
                                new_cap = concurrency_cap
                        if new_cap != concurrency_cap:
                            old_cap = concurrency_cap
                            concurrency_cap = new_cap
                            last_token_adjust = now
                            should_log = (now - last_token_adjust_log) >= token_adjust_log_cooldown or change >= token_adjust_min_delta
                            limiter_pct = limiter_wait_ratio * 100
                            reason_detail = (
                                "based on sustained limiter waits (no new errors detected)"
                                if limiter_pressure
                                else "after re-evaluating observed token usage"
                            )
                            reason = (
                                f"[token-based adaptation] Adjusting parallel workers from {old_cap} to {new_cap} "
                                f"{reason_detail}. Recent limiter wait ≈ {limiter_wait_time:.2f}s (~{limiter_pct:.0f}% of call)."
                            )
                            if should_log:
                                logger.info(reason)
                                emit_parallelization_status(reason, force=True)
                                last_token_adjust_log = now
                            else:
                                logger.debug(reason)
                        else:
                            concurrency_cap = new_cap
                if resps and all((isinstance(r, str) and not r.strip()) for r in resps):
                    if call_timeout is not None:
                        logger.warning(
                            f"Timeout for {ident} after {call_timeout:.1f}s. Consider increasing 'max_timeout'."
                        )
                else:
                    row = {
                        "Identifier": ident,
                        "Response": resps,
                        "Time Taken": duration,
                        "Input Tokens": total_input,
                        "Reasoning Tokens": total_reasoning,
                        "Output Tokens": total_output,
                        "Reasoning Effort": get_response_kwargs.get(
                            "reasoning_effort", reasoning_effort
                        ),
                        "Error Log": error_logs.get(ident, []),
                    }
                    if response_ids:
                        row["Response IDs"] = response_ids
                    if reasoning_summary is not None:
                        row["Reasoning Summary"] = summary_text
                    is_success = True if success_override is None else bool(success_override)
                    row["Successful"] = is_success
                    results.append(row)
                    processed += 1
                    status.num_tasks_in_progress -= 1
                    pbar.update(1)
                    error_logs.pop(ident, None)
                    if is_success:
                        status.num_tasks_succeeded += 1
                        successes_since_adjust += 1
                        rate_limit_errors_since_adjust = 0
                        maybe_adjust_concurrency()
                        if processed % save_every_x_responses == 0:
                            await flush()
                    else:
                        status.num_tasks_failed += 1
                        await flush()
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, APITimeoutError) as e:
                status.num_timeout_errors += 1
                elapsed = time.time() - start
                inflight.pop(ident, None)
                await adjust_timeout()
                if isinstance(e, APITimeoutError):
                    error_message = (
                        f"OpenAI client timed out after {elapsed:.2f} s; "
                        "consider increasing max_timeout or reducing concurrency."
                    )
                    detail = str(e)
                    if detail:
                        error_logs[ident].append(detail)
                else:
                    error_message = f"API call timed out after {elapsed:.2f} s"
                logger.warning(f"Timeout error for {ident}: {error_message}")
                error_logs[ident].append(error_message)
                if attempts_left - 1 > 0:
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    # Retry the same prompt after a delay.  We sleep within the
                    # worker so the task remains accounted for in ``queue.join``
                    # and ensure the new task is enqueued before ``task_done``
                    # is called.  This mirrors the legacy retry behaviour and
                    # prevents retries from being dropped prematurely.
                    await asyncio.sleep(backoff)
                    queue.put_nowait((prompt, ident, attempts_left - 1))
                else:
                    row = {
                        "Identifier": ident,
                        "Response": None,
                        "Time Taken": None,
                        "Input Tokens": input_tokens,
                        "Reasoning Tokens": None,
                        "Output Tokens": None,
                        "Reasoning Effort": get_response_kwargs.get(
                            "reasoning_effort", reasoning_effort
                        ),
                        "Successful": False,
                        "Error Log": error_logs.get(ident, []),
                    }
                    if response_ids:
                        row["Response IDs"] = response_ids
                    if reasoning_summary is not None:
                        row["Reasoning Summary"] = None
                    results.append(row)
                    processed += 1
                    status.num_tasks_failed += 1
                    status.num_tasks_in_progress -= 1
                    pbar.update(1)
                    error_logs.pop(ident, None)
                    await flush()
            except RateLimitError as e:
                inflight.pop(ident, None)
                status.num_rate_limit_errors += 1
                status.time_of_last_rate_limit_error = time.time()
                cooldown_until = status.time_of_last_rate_limit_error + global_cooldown
                error_text = str(e)
                logger.warning(f"Rate limit error for {ident}: {e}")
                error_logs[ident].append(error_text)
                rate_limit_error_times.append(time.time())
                rate_limit_errors_since_adjust += 1
                successes_since_adjust = 0
                if _is_quota_error_message(error_text):
                    fatal_msg = (
                        "Quota exceeded (billing or credit balance likely exhausted). "
                        "Add funds at https://platform.openai.com/settings/organization/billing/. "
                        "Stopping remaining requests."
                    )
                    logger.error(fatal_msg)
                    error_logs[ident].append(fatal_msg)
                    row = {
                        "Identifier": ident,
                        "Response": None,
                        "Time Taken": None,
                        "Input Tokens": input_tokens,
                        "Reasoning Tokens": None,
                        "Output Tokens": None,
                        "Reasoning Effort": get_response_kwargs.get(
                            "reasoning_effort", reasoning_effort
                        ),
                        "Successful": False,
                        "Error Log": error_logs.get(ident, []),
                    }
                    if response_ids:
                        row["Response IDs"] = response_ids
                    if reasoning_summary is not None:
                        row["Reasoning Summary"] = None
                    results.append(row)
                    processed += 1
                    status.num_tasks_failed += 1
                    status.num_tasks_in_progress -= 1
                    pbar.update(1)
                    drained = 0
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                            queue.task_done()
                            drained += 1
                        except asyncio.QueueEmpty:
                            break
                    if drained:
                        status.num_tasks_failed += drained
                        status.num_tasks_in_progress -= drained
                        processed += drained
                        pbar.update(drained)
                    stop_event.set()
                    await flush()
                    raise RuntimeError(fatal_msg)
                maybe_adjust_concurrency()
                if attempts_left - 1 > 0:
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    await asyncio.sleep(backoff)
                    queue.put_nowait((prompt, ident, attempts_left - 1))
                else:
                    row = {
                        "Identifier": ident,
                        "Response": None,
                        "Time Taken": None,
                        "Input Tokens": input_tokens,
                        "Reasoning Tokens": None,
                        "Output Tokens": None,
                        "Reasoning Effort": get_response_kwargs.get(
                            "reasoning_effort", reasoning_effort
                        ),
                        "Successful": False,
                        "Error Log": error_logs.get(ident, []),
                    }
                    if response_ids:
                        row["Response IDs"] = response_ids
                    if reasoning_summary is not None:
                        row["Reasoning Summary"] = None
                    results.append(row)
                    processed += 1
                    status.num_tasks_failed += 1
                    status.num_tasks_in_progress -= 1
                    pbar.update(1)
                    error_logs.pop(ident, None)
                    await flush()
            except APIConnectionError as e:
                inflight.pop(ident, None)
                status.num_api_errors += 1
                logger.warning(f"Connection error for {ident}: {e}")
                error_logs[ident].append(str(e))
                if attempts_left - 1 > 0 and not stop_event.is_set():
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    await asyncio.sleep(backoff)
                    queue.put_nowait((prompt, ident, attempts_left - 1))
                else:
                    row = {
                        "Identifier": ident,
                        "Response": None,
                        "Time Taken": None,
                        "Input Tokens": input_tokens,
                        "Reasoning Tokens": None,
                        "Output Tokens": None,
                        "Reasoning Effort": get_response_kwargs.get(
                            "reasoning_effort", reasoning_effort
                        ),
                        "Successful": False,
                        "Error Log": error_logs.get(ident, []),
                    }
                    if reasoning_summary is not None:
                        row["Reasoning Summary"] = None
                    results.append(row)
                    processed += 1
                    status.num_tasks_failed += 1
                    status.num_tasks_in_progress -= 1
                    pbar.update(1)
                    error_logs.pop(ident, None)
                    await flush()
            except (
                APIError,
                BadRequestError,
                AuthenticationError,
                InvalidRequestError,
            ) as e:
                inflight.pop(ident, None)
                status.num_api_errors += 1
                logger.warning(f"API error for {ident}: {e}")
                error_logs[ident].append(str(e))
                row = {
                    "Identifier": ident,
                    "Response": None,
                    "Time Taken": None,
                    "Input Tokens": input_tokens,
                    "Reasoning Tokens": None,
                    "Output Tokens": None,
                    "Reasoning Effort": get_response_kwargs.get(
                        "reasoning_effort", reasoning_effort
                    ),
                    "Successful": False,
                    "Error Log": error_logs.get(ident, []),
                }
                if response_ids:
                    row["Response IDs"] = response_ids
                if reasoning_summary is not None:
                    row["Reasoning Summary"] = None
                results.append(row)
                processed += 1
                status.num_tasks_failed += 1
                status.num_tasks_in_progress -= 1
                pbar.update(1)
                error_logs.pop(ident, None)
                await flush()
            except Exception as e:
                inflight.pop(ident, None)
                status.num_other_errors += 1
                logger.error(f"Unexpected error for {ident}: {e}")
                await flush()
                raise
            finally:
                active_workers -= 1
                queue.task_done()

    async def timeout_watcher() -> None:
        try:
            while True:
                await asyncio.sleep(0.5)
                if stop_event.is_set():
                    break
                now = time.time()
                current_timeout = nonlocal_timeout
                for ident, (start, task, t_out) in list(inflight.items()):
                    if task.done():
                        continue
                    if _should_cancel_inflight_task(
                        start,
                        now,
                        current_timeout,
                        t_out,
                        dynamic_timeout,
                    ):
                        timeout_cancellations.add(ident)
                        task.cancel()
        except asyncio.CancelledError:
            pass

    async def status_reporter() -> None:
        if status_report_interval is None:
            return
        try:
            while not stop_event.is_set():
                await asyncio.sleep(status_report_interval)
                if stop_event.is_set() or processed >= status.num_tasks_started:
                    break
                emit_parallelization_status("Periodic status update", force=True)
        except asyncio.CancelledError:
            pass

    # Spawn workers and ensure they are cleaned up on exit or cancellation
    watcher = asyncio.create_task(timeout_watcher())
    status_task: Optional[asyncio.Task] = None
    if status_report_interval is not None:
        status_task = asyncio.create_task(status_reporter())
    initial_worker_count = max(1, min(max_parallel_ceiling, queue.qsize()))
    workers = [asyncio.create_task(worker()) for _ in range(initial_worker_count)]
    try:
        await queue.join()
    except (asyncio.CancelledError, KeyboardInterrupt):
        stop_event.set()
        logger.info("Cancellation requested, shutting down workers...")
        raise
    finally:
        stop_event.set()
        for w in workers:
            w.cancel()
        watcher.cancel()
        if status_task is not None:
            status_task.cancel()
        worker_results = await asyncio.gather(*workers, return_exceptions=True)
        await asyncio.gather(watcher, return_exceptions=True)
        if status_task is not None:
            await asyncio.gather(status_task, return_exceptions=True)
        for res in worker_results:
            if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                # flush partial results before raising
                await flush()
                pbar.close()
                raise res
        # Flush remaining results and close progress bar
        await flush()
        pbar.close()

    logger.info(
        f"Processing complete. {status.num_tasks_succeeded}/{status.num_tasks_started} requests succeeded."
    )
    if status.num_tasks_failed > 0:
        logger.warning(f"{status.num_tasks_failed} requests failed.")
    if status.num_rate_limit_errors > 0:
        logger.warning(
            f"{status.num_rate_limit_errors} rate limit errors encountered; consider reducing concurrency."
        )
    if status.num_timeout_errors > 0:
        logger.warning(f"{status.num_timeout_errors} timeouts encountered.")
    if status.num_api_errors > 0:
        logger.warning(f"{status.num_api_errors} API errors encountered.")
    if status.num_other_errors > 0:
        logger.warning(f"{status.num_other_errors} unexpected errors encountered.")
    _report_cost()
    return df
