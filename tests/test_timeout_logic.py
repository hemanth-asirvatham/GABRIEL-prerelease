"""Tests for timeout selection helpers."""

import math

from gabriel.utils.openai_utils import _resolve_effective_timeout


def test_resolve_effective_timeout_uses_task_budget_when_available() -> None:
    """Retries keep their extended timeout budgets when dynamic timeouts run."""

    assert _resolve_effective_timeout(90.0, 135.0, True) == 135.0


def test_resolve_effective_timeout_falls_back_to_global_timeout() -> None:
    """Tasks dispatched before initialization should respect the global limit."""

    assert _resolve_effective_timeout(90.0, math.inf, True) == 90.0


def test_resolve_effective_timeout_respects_explicit_timeouts_when_static() -> None:
    """Static timeout configuration should always use the provided value."""

    assert _resolve_effective_timeout(math.inf, 40.0, False) == 40.0
