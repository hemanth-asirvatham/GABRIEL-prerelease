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
* Dynamic ``max_output_tokens`` – when a user does not specify
  ``max_output_tokens`` explicitly, the library inspects the current
  token quota.  If fewer than one million tokens remain in the minute
  budget, a safety cutoff of 2 500 tokens is applied; otherwise, the
  parameter is left ``None`` so the model’s default output limit is used.
  This prevents long responses from being rejected due to an overly high
  token estimate, while removing unnecessary complexity when there is
  ample capacity.
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
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
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
from dataclasses import dataclass

logger = get_logger(__name__)

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
        AuthenticationError,
        BadRequestError,
        InvalidRequestError,
        RateLimitError,
    )  # type: ignore
except Exception:
    APIConnectionError = Exception  # type: ignore
    APIError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore
    BadRequestError = Exception  # type: ignore
    InvalidRequestError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore

from gabriel.utils.parsing import safe_json

# single connection pool per process, keyed by base URL and created lazily
_clients_async: Dict[Optional[str], openai.AsyncOpenAI] = {}


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
        client = openai.AsyncOpenAI(**kwargs)
        _clients_async[key] = client
    return client

# Default safety cutoff when token capacity is low
DEFAULT_MAX_OUTPUT_TOKENS = 2500

# Estimated output tokens per prompt used for cost estimation when no cutoff is specified.
# When a user does not explicitly set ``max_output_tokens``, we assume that each response
# will contain roughly this many tokens.  This value is used solely for estimating cost
# and determining how many parallel requests can safely run under the token budget.
ESTIMATED_OUTPUT_TOKENS_PER_PROMPT = 2500

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
) -> Optional[Dict[str, float]]:
    """Estimate input/output tokens and cost for a set of prompts.

    Returns a dict with keys ``input_tokens``, ``output_tokens``, ``input_cost``, ``output_cost``, and ``total_cost``.
    If the model pricing is unavailable, returns ``None``.
    """
    pricing = _lookup_model_pricing(model)
    if pricing is None:
        return None
    # Estimate tokens: input tokens are sum of tokens per prompt times number of responses
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
) -> None:
    """Print a summary of usage limits, cost estimate and tier information.

    Optionally takes a pre‑fetched ``rate_headers`` dict to avoid calling
    ``_get_rate_limit_headers`` multiple times per job.  When ``rate_headers``
    is ``None``, the helper will fetch the headers itself.
    """
    if not verbose:
        return
    print("\n===== OpenAI API usage summary =====")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Total input words: {sum(len(str(p).split()) for p in prompts):,}")
    # Fetch fresh headers if not supplied.  Pass the model and base_url so the
    # helper knows which endpoint to probe when performing the dummy call.
    rl = rate_headers or _get_rate_limit_headers(model, base_url=base_url)
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
        print(
            f"Requests per minute: {fmt(lim_r_val)} – maximum API calls you can make each minute."
        )
    else:
        print("Requests per minute: unknown – maximum API calls you can make each minute.")
    if lim_t_val is not None:
        print(
            f"Tokens per minute: {fmt(lim_t_val)} – maximum input + output tokens allowed per minute."
        )
        words_per_min = int(lim_t_val) // 2
        print(
            f"Words per minute: {words_per_min:,} – approximate number of words you can process per minute (2 tokens ≈ 1 word)."
        )
    else:
        print(
            "Tokens per minute: unknown – maximum input + output tokens allowed per minute."
        )
        print(
            "Words per minute: unknown – approximate number of words you can process per minute."
        )
    # Let users know about monthly usage caps in addition to per‑minute limits.
    print(
        "\nNote: your organization also has a monthly usage cap based on your tier. See the usage tiers below for details."
    )
    # Display usage tiers succinctly
    print("\nUsage tiers:")
    for tier in TIER_INFO:
        print(
            f"  • {tier['tier']}: qualifies by {tier['qualification']}; monthly quota {tier['monthly_quota']}"
        )
    pricing = _lookup_model_pricing(model)
    est = _estimate_cost(prompts, n, max_output_tokens, model, use_batch)
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
        avg_input_tokens = sum(_approx_tokens(p) for p in prompts) / max(1, len(prompts))
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
            # Allowed requests are whichever remaining value is available, otherwise
            # the limit.  If both are missing, None indicates unknown.
            allowed_req = rem_r_val2 if rem_r_val2 is not None else lim_r_val2
            allowed_tok = rem_t_val2 if rem_t_val2 is not None else lim_t_val2
        else:
            allowed_req = None
            allowed_tok = None
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
        print(
            f"\nNote: based on your current plan and rate limits, we'll run up to {concurrency_cap} requests at the same time instead of {n_parallels}. Upgrading your tier would allow more parallel requests and speed up processing."
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
    """Decide a default ``max_output_tokens`` based on current token budget.

    If ``user_specified`` is not ``None``, return it unchanged.  Otherwise,
    use the supplied ``rate_headers`` dict (or fetch one if ``None``) to
    determine how many tokens remain in the per‑minute budget.  If fewer than
    one million tokens remain, return ``DEFAULT_MAX_OUTPUT_TOKENS``; else
    return ``None`` to indicate no cutoff.
    """
    if user_specified is not None:
        return user_specified
    # When rate headers are not supplied, fall back to fetching them using
    # the default model.  Passing a model ensures the helper uses the
    # minimal chat call rather than the unsupported ``/v1/models`` endpoint.
    rl = rate_headers or _get_rate_limit_headers(base_url=base_url)
    if rl and rl.get("remaining_tokens"):
        try:
            rem = int(float(rl["remaining_tokens"]))
            if rem < 1_000_000:
                return DEFAULT_MAX_OUTPUT_TOKENS
        except Exception:
            pass
    return None


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
        all_tools.append(
            {"type": "web_search_preview", "search_context_size": search_context_size}
        )
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
    # Derive the effective cutoff
    cutoff = max_output_tokens if max_output_tokens is not None else max_tokens
    # Build system message only for non‑o series
    system_instruction = (
        "Please provide a helpful response to this inquiry for purposes of academic research."
    )
    if audio:
        logger.info(
            "Audio inputs require models gpt-4o-audio-preview or gpt-4o-mini-audio-preview"
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
        # ``chat.completions`` currently infers the response modality from
        # the request content.  Supplying an explicit ``modalities`` field
        # leads to ``Unknown parameter`` errors on newer models (e.g. gpt‑5),
        # so we omit it here and default to text output.
        params_chat: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools is not None:
            params_chat["tools"] = tools
        if tool_choice is not None:
            params_chat["tool_choice"] = tool_choice
        if cutoff is not None:
            params_chat["max_completion_tokens"] = cutoff
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
        except asyncio.TimeoutError:
            err = Exception(f"API call timed out after {timeout} s")
            logger.error(f"[get_response] {err}")
            raise err
        except Exception as e:
            err = Exception(f"API call resulted in exception: {e!r}")
            logger.error(f"[get_response] {err}")
            raise err
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
            search_context_size=search_context_size,
            json_mode=json_mode,
            expected_schema=expected_schema,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            **kwargs,
        )
        start = time.time()
        # Create parallel tasks for `n` completions
        tasks = [
            asyncio.create_task(
                client_async.responses.create(
                    **params, **({"timeout": timeout} if timeout is not None else {})
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
        except asyncio.TimeoutError:
            err = Exception(f"API call timed out after {timeout} s")
            logger.error(f"[get_response] {err}")
            raise err
        except Exception as e:
            err = Exception(f"API call resulted in exception: {e!r}")
            logger.error(f"[get_response] {err}")
            raise err
        # Extract ``output_text`` from the responses.  For Responses API
        # the SDK returns an object with an ``output_text`` attribute.
        texts = [r.output_text for r in raw]
        duration = time.time() - start
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
    except asyncio.TimeoutError:
        err = Exception(f"API call timed out after {timeout} s")
        logger.error(f"[get_embedding] {err}")
        raise err
    except Exception as e:
        err = Exception(f"API call resulted in exception: {e!r}")
        logger.error(f"[get_embedding] {err}")
        raise err

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

    save_path = os.path.expanduser(os.path.expandvars(save_path))
    embeddings: Dict[str, List[float]] = {}
    if not reset_file and os.path.exists(save_path):
        try:
            with open(save_path, "rb") as f:
                embeddings = pickle.load(f)
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
        return embeddings

    tokenizer = _get_tokenizer(model)
    get_embedding_kwargs.setdefault("base_url", base_url)
    error_logs: Dict[str, List[str]] = defaultdict(list)
    queue: asyncio.Queue[Tuple[str, str, int]] = asyncio.Queue()
    for item in items:
        queue.put_nowait((item[1], item[0], max_retries))

    processed = 0
    pbar = tqdm(
        total=len(items),
        disable=not verbose,
        desc="Getting embeddings",
        leave=True,
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
        decrease_threshold = 20
        increase_threshold = 200
        if rate_limit_errors_since_adjust >= decrease_threshold:
            new_cap = max(1, int(math.floor(concurrency_cap * 0.75)))
            if new_cap != concurrency_cap:
                msg = (
                    f"[scale down] Reducing parallel workers from {concurrency_cap} to {new_cap} due to repeated rate limit errors."
                )
                print(msg)
                logger.warning(msg)
            concurrency_cap = new_cap
            rate_limit_errors_since_adjust = 0
            successes_since_adjust = 0
        elif rate_limit_errors_since_adjust == 0 and successes_since_adjust >= increase_threshold:
            new_cap = min(n_parallels, int(math.ceil(concurrency_cap * 1.25)))
            if new_cap != concurrency_cap:
                msg = (
                    f"[scale up] Increasing parallel workers from {concurrency_cap} to {new_cap} after sustained success."
                )
                print(msg)
                logger.warning(msg)
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
            except asyncio.TimeoutError as e:
                error_logs[ident].append(str(e))
                logger.warning(f"Timeout error for {ident}: {e}")
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


async def get_all_responses(
    prompts: List[str],
    identifiers: Optional[List[str]] = None,
    prompt_images: Optional[Dict[str, List[str]]] = None,
    prompt_audio: Optional[Dict[str, List[Dict[str, str]]]] = None,
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
    use_web_search: Optional[bool] = None,
    search_context_size: str = "medium",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    use_dummy: bool = False,
    base_url: Optional[str] = None,
    print_example_prompt: bool = True,
    save_path: str = "responses.csv",
    reset_files: bool = False,
    # Maximum number of parallel worker tasks to spawn.  This value
    # represents a ceiling; the actual number of concurrent requests
    # will be adjusted downward based on your API rate limits and
    # average prompt length.  See `_print_usage_overview` for more
    # details on how the concurrency cap is calculated.
    n_parallels: int = 750,
    max_retries: int = 3,
    timeout_factor: float = 2.00,
    max_timeout: Optional[float] = None,
    dynamic_timeout: bool = True,
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
    global_cooldown: int = 15,
    token_sample_size: int = 20,
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
    request exceeding the current limit is cancelled and retried.

    The worker pool responds promptly to user cancellation (e.g. pressing
    stop/``Ctrl+C``) by signalling all workers to halt before any new API
    requests are issued.  Transient network disruptions such as lost
    connections are retried with exponential backoff so long‑running jobs can
    resume automatically once connectivity returns.

    The function remains backwards compatible with the original version, except
    that the parameter ``max_tokens`` has been renamed to ``max_output_tokens``.
    When both are provided, ``max_output_tokens`` takes precedence.  The former
    ``use_web_search`` flag is still accepted but ``web_search`` should be used
    going forward.
"""
    if not use_dummy:
        _require_api_key()
    set_log_level(logging_level)
    logger = get_logger(__name__)
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    # ``use_web_search`` was the original parameter name; ``web_search`` is the
    # preferred modern spelling.  If both are supplied we favour ``web_search``
    # but emit a warning for awareness.
    if web_search is None:
        web_search = bool(use_web_search)
    elif use_web_search is not None and use_web_search != web_search:
        logger.warning(
            "`use_web_search` is deprecated; please use `web_search` instead."
        )

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
    tokenizer = _get_tokenizer(model)
    # Backwards compatibility for identifiers
    if identifiers is None:
        identifiers = prompts
    # Pull default values into kwargs for get_response
    get_response_kwargs.setdefault("web_search", web_search)
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
    # Decide default cutoff once per job using cached rate headers
    # Fetch rate headers once to avoid multiple API calls
    # Retrieve rate‑limit headers for the chosen model.  Passing the model
    # ensures the helper performs a dummy call with the correct model
    # rather than probing the unsupported ``/v1/models`` endpoint.
    rate_headers = _get_rate_limit_headers(model, base_url=base_url)
    user_cutoff = max_output_tokens if max_output_tokens is not None else max_tokens
    cutoff = _decide_default_max_output_tokens(user_cutoff, rate_headers, base_url=base_url)
    get_response_kwargs.setdefault("max_output_tokens", cutoff)
    # Always load or initialise the CSV
    # Expand variables in save_path and ensure the parent directory exists.
    save_path = os.path.expandvars(os.path.expanduser(save_path))
    if reset_files:
        for p in (save_path, save_path + ".batch_state.json"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    if os.path.exists(save_path) and not reset_files:
        print("Reading from existing files...")
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
        print(msg)
        logger.info(msg)
    # Filter prompts/identifiers based on what is already completed
    todo_pairs = [(p, i) for p, i in zip(prompts, identifiers) if i not in done]
    if not todo_pairs:
        _report_cost()
        return df
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
    # Print usage summary and example prompt
    if print_example_prompt and todo_pairs:
        # Build prompt list for cost estimate
        prompt_list = [p for p, _ in todo_pairs]
        _print_usage_overview(
            prompts=prompt_list,
            n=n,
            max_output_tokens=cutoff,
            model=model,
            use_batch=use_batch,
            n_parallels=n_parallels,
            verbose=verbose,
            rate_headers=rate_headers,
            base_url=base_url,
        )
        example_prompt, _ = todo_pairs[0]
        logger.warning(f"Example prompt: {example_prompt}")
    # Dynamically adjust the maximum number of parallel workers based on rate
    # limits.  We base the concurrency on your API’s per‑minute request and
    # token budgets and the average prompt length.  This calculation only
    # runs once at the start of a non‑batch run.  The resulting value acts
    # as the true upper bound on parallelism; it will be used to size the
    # worker pool and to configure the request/token limiters below.
    if not use_batch:
        try:
            # Estimate the average number of tokens per call using tiktoken
            # for more accurate gating.  We include the expected output length
            # to ensure that long prompts reduce available parallelism.
            avg_input_tokens = (
                sum(len(tokenizer.encode(p)) for p, _ in todo_pairs)
                / max(1, len(todo_pairs))
            )
            gating_output = cutoff if cutoff is not None else ESTIMATED_OUTPUT_TOKENS_PER_PROMPT
            tokens_per_call = (avg_input_tokens + gating_output) * max(1, n)
            # Parse limits from the rate headers.  If the API returns both a
            # limit and a remaining value, we prefer the remaining value (it
            # reflects your remaining quota for the current minute).  Missing
            # or zero values are treated as unknown.
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
            if rate_headers:
                lim_r = _pf(rate_headers.get("limit_requests"))
                rem_r = _pf(rate_headers.get("remaining_requests"))
                allowed_req = rem_r if rem_r is not None else lim_r
                lim_t = _pf(rate_headers.get("limit_tokens")) or _pf(rate_headers.get("limit_tokens_usage_based"))
                rem_t = _pf(rate_headers.get("remaining_tokens")) or _pf(rate_headers.get("remaining_tokens_usage_based"))
                allowed_tok = rem_t if rem_t is not None else lim_t
            else:
                allowed_req = None
                allowed_tok = None
            # Compute the theoretical parallelism from request and token budgets
            if allowed_req is None:
                concurrency_from_requests: Optional[int] = None
            else:
                concurrency_from_requests = int(max(1, allowed_req))
            if allowed_tok is None:
                concurrency_from_tokens: Optional[int] = None
            else:
                concurrency_from_tokens = int(max(1, allowed_tok // tokens_per_call))
            if concurrency_from_requests is None and concurrency_from_tokens is None:
                concurrency_possible: Optional[int] = None
            elif concurrency_from_requests is None:
                concurrency_possible = concurrency_from_tokens
            elif concurrency_from_tokens is None:
                concurrency_possible = concurrency_from_requests
            else:
                concurrency_possible = min(concurrency_from_requests, concurrency_from_tokens)
            # Determine final concurrency cap.  If concurrency_possible is None
            # (unknown), we leave n_parallels unchanged.  Otherwise we limit
            # to the lesser of the calculated parallelism and the user‑supplied
            # ceiling.
            if concurrency_possible is not None:
                concurrency_cap = max(1, min(n_parallels, concurrency_possible))
            else:
                concurrency_cap = max(1, n_parallels)
        except Exception:
            concurrency_cap = max(1, n_parallels)
        # Warn the user when concurrency is reduced due to rate limits.
        if concurrency_cap < n_parallels:
            logger.info(
                f"[parallel reduction] Limiting parallel workers from {n_parallels} to {concurrency_cap} based on your current rate limits. Consider upgrading your plan for faster processing."
            )
        n_parallels = concurrency_cap
        # Compute per‑minute budgets for gating.  When the API returns a
        # request or token limit, we use it; otherwise we derive a large
        # default based on the concurrency cap.  The defaults ensure that
        # limiters do not unnecessarily throttle when limits are unknown.
        if allowed_req is not None:
            allowed_req_pm = int(max(1, allowed_req))
        else:
            allowed_req_pm = max(1, n_parallels)
        if allowed_tok is not None:
            allowed_tok_pm = int(max(1, allowed_tok))
        else:
            allowed_tok_pm = int(max(1, n_parallels * tokens_per_call))
    else:
        # In batch mode we don't set concurrency or limiters here; they are
        # handled by the batch API submission.
        allowed_req_pm = 1
        allowed_tok_pm = 1

    # Batch submission path
    if use_batch:
        state_path = save_path + ".batch_state.json"

        # Helper to append batch rows
        def _append_results(rows: List[Dict[str, Any]]) -> None:
            nonlocal df
            if not rows:
                return
            batch_df = pd.DataFrame(rows)
            batch_df["Response"] = batch_df["Response"].apply(_ser)
            batch_df["Error Log"] = batch_df["Error Log"].apply(_ser)
            if os.path.exists(save_path):
                existing = pd.read_csv(save_path)
                existing = existing[~existing["Identifier"].isin(batch_df["Identifier"])]
                combined = pd.concat([existing, batch_df], ignore_index=True)
            else:
                combined = batch_df
            combined.to_csv(
                save_path,
                mode="w",
                header=True,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            combined["Response"] = combined["Response"].apply(_de)
            combined["Error Log"] = combined["Error Log"].apply(_de)
            df = combined

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
                body = _build_params(
                    model=get_response_kwargs.get("model", "gpt-5-mini"),
                    input_data=input_data,
                    max_output_tokens=cutoff,
                    system_instruction="Please provide a helpful response to this inquiry for purposes of academic research.",
                    temperature=get_response_kwargs.get("temperature", 0.9),
                    tools=get_response_kwargs.get("tools"),
                    tool_choice=get_response_kwargs.get("tool_choice"),
                    web_search=get_response_kwargs.get("web_search", False),
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
    req_lim = AsyncLimiter(allowed_req_pm, 60)
    tok_lim = AsyncLimiter(allowed_tok_pm, 60)
    success_times: List[float] = []
    inflight: Dict[str, Tuple[float, asyncio.Task, float]] = {}
    error_logs: Dict[str, List[str]] = defaultdict(list)
    call_count = 0
    samples_for_timeout = max(1, int(0.90 * min(len(todo_pairs), n_parallels)))
    queue: asyncio.Queue[Tuple[str, str, int]] = asyncio.Queue()
    for item in todo_pairs:
        queue.put_nowait((item[0], item[1], max_retries))
    results: List[Dict[str, Any]] = []
    processed = 0
    pbar = tqdm(total=len(todo_pairs), desc="Processing prompts", leave=True)
    cooldown_until = 0.0
    stop_event = asyncio.Event()
    # Counters used for the gentle concurrency adaptation below
    rate_limit_errors_since_adjust = 0
    successes_since_adjust = 0
    active_workers = 0
    concurrency_cap = n_parallels
    usage_samples: List[Tuple[int, int, int]] = []
    estimated_output_tokens = cutoff if cutoff is not None else ESTIMATED_OUTPUT_TOKENS_PER_PROMPT

    async def flush() -> None:
        nonlocal results, df, processed
        if results:
            batch_df = pd.DataFrame(results)
            batch_df["Response"] = batch_df["Response"].apply(_ser)
            batch_df["Error Log"] = batch_df["Error Log"].apply(_ser)
            if os.path.exists(save_path):
                existing = pd.read_csv(save_path)
                existing = existing[~existing["Identifier"].isin(batch_df["Identifier"])]
                combined = pd.concat([existing, batch_df], ignore_index=True)
            else:
                combined = batch_df
            combined.to_csv(
                save_path,
                mode="w",
                header=True,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            combined["Response"] = combined["Response"].apply(_de)
            combined["Error Log"] = combined["Error Log"].apply(_de)
            df = combined
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
        nonlocal nonlocal_timeout
        if not dynamic_timeout:
            return
        if len(success_times) < samples_for_timeout:
            return
        try:
            p90 = float(np.percentile(success_times, 90))
            new_timeout = min(max_timeout_val, timeout_factor * p90)
            if math.isinf(nonlocal_timeout) or new_timeout > nonlocal_timeout:
                logger.debug(
                    f"[dynamic timeout] Updating timeout to {new_timeout:.1f}s based on 90th percentile latency."
                )
                nonlocal_timeout = new_timeout
            if not math.isinf(nonlocal_timeout):
                now = time.time()
                for ident, (start, task, t_out) in list(inflight.items()):
                    limit = min(nonlocal_timeout, t_out) if dynamic_timeout else t_out
                    if now - start > limit and not task.done():
                        task.cancel()
        except Exception:
            pass

    # We removed the dynamic rate‑limit adjustment functionality.  If the API
    # returns a 429 (rate limit error), we simply retry after an exponential
    # backoff without modifying the per‑minute budgets.  This avoids
    # overreacting to a single error and keeps concurrency stable.  The
    # budgets and concurrency are determined once at the start of the job.
    async def rebuild_limiters() -> None:
        return None

    def maybe_adjust_concurrency() -> None:
        nonlocal concurrency_cap, rate_limit_errors_since_adjust, successes_since_adjust
        decrease_threshold = 20
        increase_threshold = 200
        if rate_limit_errors_since_adjust >= decrease_threshold:
            new_cap = max(1, int(math.floor(concurrency_cap * 0.75)))
            if new_cap != concurrency_cap:
                logger.warning(
                    f"[scale down] Reducing parallel workers from {concurrency_cap} to {new_cap} due to repeated rate limit errors."
                )
            concurrency_cap = new_cap
            rate_limit_errors_since_adjust = 0
            successes_since_adjust = 0
        elif rate_limit_errors_since_adjust == 0 and successes_since_adjust >= increase_threshold:
            new_cap = min(n_parallels, int(math.ceil(concurrency_cap * 1.25)))
            if new_cap != concurrency_cap:
                logger.warning(
                    f"[scale up] Increasing parallel workers from {concurrency_cap} to {new_cap} after sustained success."
                )
            concurrency_cap = new_cap
            successes_since_adjust = 0
            rate_limit_errors_since_adjust = 0

    async def worker() -> None:
        nonlocal processed, call_count, nonlocal_timeout, active_workers, concurrency_cap, cooldown_until, estimated_output_tokens, rate_limit_errors_since_adjust, successes_since_adjust, stop_event
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
                await req_lim.acquire()
                await tok_lim.acquire((input_tokens + gating_output) * n)
                call_count += 1
                error_logs.setdefault(ident, [])
                start = time.time()
                base_timeout = nonlocal_timeout
                multiplier = 1.5 ** (max_retries - attempts_left)
                call_timeout = None if math.isinf(base_timeout) else base_timeout * multiplier
                if call_timeout is not None and dynamic_timeout and not math.isinf(max_timeout_val):
                    call_timeout = min(call_timeout, max_timeout_val)
                task = asyncio.create_task(
                    get_response(
                        prompt,
                        n=n,
                        timeout=call_timeout,
                        use_dummy=use_dummy,
                        images=prompt_images.get(str(ident)) if prompt_images else None,
                        audio=prompt_audio.get(str(ident)) if prompt_audio else None,
                        return_raw=True,
                        **get_response_kwargs,
                    )
                )
                inflight[ident] = (start, task, call_timeout if call_timeout is not None else float("inf"))
                try:
                    resps, t, raw = await task
                except asyncio.CancelledError:
                    inflight.pop(ident, None)
                    raise asyncio.TimeoutError(
                        f"API call timed out after {call_timeout} s"
                    )
                inflight.pop(ident, None)
                success_times.append(t)
                await adjust_timeout()
                # collect usage
                total_input = sum(getattr(r.usage, "input_tokens", 0) for r in raw)
                total_output = sum(getattr(r.usage, "output_tokens", 0) for r in raw)
                total_reasoning = sum(
                    getattr(getattr(r.usage, "output_tokens_details", {}), "reasoning_tokens", 0)
                    for r in raw
                )
                summary_text = None
                try:
                    for r in raw:
                        out_items = getattr(r, "output", [])
                        if not out_items:
                            continue
                        for item in out_items:
                            if getattr(item, "type", None) == "reasoning":
                                summary_list = getattr(item, "summary", [])
                                if summary_list:
                                    first = summary_list[0]
                                    txt = getattr(first, "text", None)
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
                if len(usage_samples) >= token_sample_size:
                    avg_in = statistics.mean(u[0] for u in usage_samples)
                    avg_out = statistics.mean(u[1] for u in usage_samples)
                    avg_reason = statistics.mean(u[2] for u in usage_samples)
                    estimated_output_tokens = avg_out + avg_reason
                    tokens_per_call_est = (avg_in + estimated_output_tokens) * max(1, n)
                    new_cap = min(
                        n_parallels,
                        int(allowed_req_pm),
                        int(max(1, allowed_tok_pm // max(1, tokens_per_call_est))),
                    )
                    if new_cap < 1:
                        new_cap = 1
                    if new_cap != concurrency_cap:
                        logger.info(
                            f"[token-based adaptation] Updating parallel workers from {concurrency_cap} to {new_cap} based on observed token usage."
                        )
                    concurrency_cap = new_cap
                # Check for empty outputs
                if resps and all((isinstance(r, str) and not r.strip()) for r in resps):
                    if call_timeout is not None:
                        logger.warning(
                            f"Timeout for {ident} after {call_timeout:.1f}s. Consider increasing 'max_timeout'."
                        )
                else:
                    row = {
                        "Identifier": ident,
                        "Response": resps,
                        "Time Taken": t,
                        "Input Tokens": total_input,
                        "Reasoning Tokens": total_reasoning,
                        "Output Tokens": total_output,
                        "Reasoning Effort": get_response_kwargs.get(
                            "reasoning_effort", reasoning_effort
                        ),
                        "Successful": True,
                        "Error Log": error_logs.get(ident, []),
                    }
                    if reasoning_summary is not None:
                        row["Reasoning Summary"] = summary_text
                    results.append(row)
                    processed += 1
                    status.num_tasks_succeeded += 1
                    status.num_tasks_in_progress -= 1
                    pbar.update(1)
                    error_logs.pop(ident, None)
                    successes_since_adjust += 1
                    rate_limit_errors_since_adjust = 0
                    maybe_adjust_concurrency()
                    if processed % save_every_x_responses == 0:
                        await flush()
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as e:
                status.num_timeout_errors += 1
                logger.warning(f"Timeout error for {ident}: {e}")
                inflight.pop(ident, None)
                await adjust_timeout()
                error_logs[ident].append(str(e))
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
                logger.warning(f"Rate limit error for {ident}: {e}")
                error_logs[ident].append(str(e))
                rate_limit_errors_since_adjust += 1
                successes_since_adjust = 0
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
        while True:
            await asyncio.sleep(0.5)
            now = time.time()
            for ident, (start, task, t_out) in list(inflight.items()):
                if now - start > t_out and not task.done():
                    task.cancel()

    # Spawn workers and ensure they are cleaned up on exit or cancellation
    watcher = asyncio.create_task(timeout_watcher())
    workers = [asyncio.create_task(worker()) for _ in range(n_parallels)]
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
        worker_results = await asyncio.gather(*workers, return_exceptions=True)
        await asyncio.gather(watcher, return_exceptions=True)
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
