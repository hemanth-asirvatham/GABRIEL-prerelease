"""Tests for timeout selection helpers."""

import math
import time

from gabriel.utils.openai_utils import (
    _resolve_effective_timeout,
    _should_cancel_inflight_task,
)


def test_resolve_effective_timeout_uses_task_budget_when_available() -> None:
    """Retries keep their extended timeout budgets when dynamic timeouts run."""

    assert _resolve_effective_timeout(90.0, 135.0, True) == 135.0


def test_resolve_effective_timeout_falls_back_to_global_timeout() -> None:
    """Tasks dispatched before initialization should respect the global limit."""

    assert _resolve_effective_timeout(90.0, math.inf, True) == 90.0


def test_resolve_effective_timeout_respects_explicit_timeouts_when_static() -> None:
    """Static timeout configuration should always use the provided value."""

    assert _resolve_effective_timeout(math.inf, 40.0, False) == 40.0


def test_should_cancel_inflight_honors_dynamic_budget() -> None:
    """Tasks dispatched before initialization adopt the global timeout."""

    start = time.time() - 100.0
    now = time.time()
    assert _should_cancel_inflight_task(start, now, 90.0, math.inf, True)


def test_should_cancel_inflight_skips_infinite_budgets() -> None:
    """When no timeout applies the watcher should not cancel tasks."""

    start = time.time() - 10.0
    now = time.time()
    assert not _should_cancel_inflight_task(start, now, math.inf, math.inf, True)
