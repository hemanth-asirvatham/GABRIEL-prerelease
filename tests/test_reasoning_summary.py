import pytest

from gabriel.utils.openai_utils import _build_params


def _base_args():
    return dict(
        model="gpt-5",
        input_data=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        max_output_tokens=None,
        system_instruction="",
        temperature=0.0,
        tools=None,
        tool_choice=None,
        web_search=False,
        search_context_size="medium",
        json_mode=False,
        expected_schema=None,
        reasoning_effort="low",
    )


def test_summary_flag_in_reasoning():
    args = _base_args()
    params = _build_params(**args, reasoning_summary="auto")
    reasoning = params.get("reasoning", {})
    assert reasoning.get("summary") == "auto"


def test_summary_absent_when_none():
    args = _base_args()
    params = _build_params(**args, reasoning_summary=None)
    reasoning = params.get("reasoning", {})
    assert "summary" not in reasoning
