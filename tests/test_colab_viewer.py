import pandas as pd
import pytest

from gabriel.utils.passage_viewer import (
    _AttributeRequest,
    _build_highlighted_text,
    _build_note_html,
    _compute_numeric_intensity,
    _coerce_bool_value,
    _infer_attribute_kind,
    _expand_mapping_attribute_requests,
    _normalize_attribute_requests,
    _normalize_structured_dataframe,
    _passage_matches_filters,
)


def test_view_colab_runs():
    pytest.importorskip("IPython")
    from gabriel.utils.passage_viewer import view

    df = pd.DataFrame({"text": ["A snippet"], "cat": [["A snippet"]]})
    # Should not raise when using the lightweight HTML viewer
    view(df, "text", attributes=["cat"])


def test_view_colab_with_headers():
    pytest.importorskip("IPython")
    from gabriel.utils.passage_viewer import view

    df = pd.DataFrame(
        {
            "text": ["A snippet"],
            "cat": [["A snippet"]],
            "speaker": ["Test Speaker"],
        }
    )
    view(
        df,
        "text",
        attributes=["cat"],
        header_columns=["speaker"],
    )


def test_normalize_structured_dataframe_handles_strings():
    df = pd.DataFrame(
        {
            "text": [
                "A snippet of text used for testing.",
                "Another passage for validation.",
            ],
            "cat": [
                '["A snippet of text used for testing."]',
                "[]",
            ],
            "coded_passages": [
                '{"cat": ["A snippet of text used for testing."]}',
                '{"cat": []}',
            ],
        }
    )

    normalized_static = _normalize_structured_dataframe(df.copy(), ["cat"])
    assert normalized_static.at[0, "cat"] == [
        "A snippet of text used for testing."
    ]
    assert normalized_static.at[1, "cat"] == []

    normalized_dynamic = _normalize_structured_dataframe(df.copy(), "coded_passages")
    assert normalized_dynamic.at[0, "coded_passages"]["cat"] == [
        "A snippet of text used for testing."
    ]
    assert normalized_dynamic.at[1, "coded_passages"]["cat"] == []


def test_view_handles_stringified_inputs():
    pytest.importorskip("IPython")
    from gabriel.utils.passage_viewer import view

    df = pd.DataFrame(
        {
            "text": ["A snippet of text used for testing."],
            "cat": ['["A snippet of text used for testing."]'],
            "coded_passages": ['{"cat": ["A snippet of text used for testing."]}'],
        }
    )

    view(df, "text", attributes=["cat"])
    view(df, "text", attributes="coded_passages")


def test_view_handles_boolean_and_numeric_attributes():
    pytest.importorskip("IPython")
    from gabriel.utils.passage_viewer import view

    df = pd.DataFrame(
        {
            "text": ["Sample", "Another"],
            "flag": [True, False],
            "score": [4.5, 3.1],
        }
    )

    view(df, "text", attributes=["flag", "score"])


def test_view_handles_text_only_requests():
    pytest.importorskip("IPython")
    from gabriel.utils.passage_viewer import view

    df = pd.DataFrame({"text": ["Just text", "More text"]})
    view(df, "text")


def test_passage_matches_filters_helper():
    payload = {
        "snippets": {"topic": ["alpha"], "other": []},
        "bools": {"is_positive": True, "is_negative": False},
        "numeric": {"score": 4.2},
    }

    assert _passage_matches_filters(payload, required_snippets={"topic"})
    assert not _passage_matches_filters(payload, required_snippets={"missing"})
    assert _passage_matches_filters(payload, required_bools={"is_positive"})
    assert not _passage_matches_filters(payload, required_bools={"is_negative"})
    assert _passage_matches_filters(payload, numeric_filters={"score": (4.0, 5.0)})
    assert not _passage_matches_filters(payload, numeric_filters={"score": (4.3, 4.9)})


def test_numeric_intensity_defaults_to_neutral_for_uniform_values():
    assert _compute_numeric_intensity(0.0, (0.0, 0.0)) == 0.0


def test_highlight_builder_returns_plain_text_without_snippets():
    html_output = _build_highlighted_text(
        "Plain text only",
        {},
        {},
        {},
    )
    assert "Plain text only" in html_output


def test_note_builder_handles_multiple_messages():
    html_output = _build_note_html([" First message ", None, "Second"])
    assert html_output.count("gabriel-note") == 2
    assert "First message" in html_output
    assert "Second" in html_output


def test_normalize_attribute_requests_uses_mapping_keys_for_labels():
    requests = _normalize_attribute_requests({"score": "Measures intensity"})
    assert len(requests) == 1
    req = requests[0]
    assert req.column == "score"
    assert req.label == "Score"
    assert req.description == "Measures intensity"


def test_expand_mapping_attribute_requests_extracts_nested_values():
    df = pd.DataFrame(
        {
            "text": ["Alpha"],
            "attributes": [
                {"rating": 42, "flag": {"value": True}, "notes": {"answer": "ignored"}}
            ],
        }
    )
    df2, expanded = _expand_mapping_attribute_requests(
        df.copy(), [_AttributeRequest("attributes", "Attributes")]
    )
    columns = {req.column for req in expanded}
    assert "attributes::rating" in columns
    assert "attributes::flag" in columns
    assert df2.at[0, "attributes::rating"] == 42
    assert bool(df2.at[0, "attributes::flag"]) is True


def test_coerce_bool_value_does_not_treat_numeric_strings_as_bool():
    assert _coerce_bool_value("0") is None
    assert _coerce_bool_value("1") is None


def test_infer_attribute_kind_treats_empty_lists_as_snippets():
    series = pd.Series([[], [], []])
    assert _infer_attribute_kind(series) == "snippet"


def test_infer_attribute_kind_handles_stringified_empty_lists():
    series = pd.Series(["[]", "[]", "[]"])
    assert _infer_attribute_kind(series) == "snippet"


def test_top_level_view_colab_runs():
    pytest.importorskip("IPython")
    pytest.importorskip("aiolimiter")
    pytest.importorskip("openai")
    import gabriel

    df = pd.DataFrame(
        {
            "text": ["A snippet"],
            "cat": [["A snippet"]],
            "speaker": ["Top Level"],
        }
    )
    gabriel.view(
        df,
        "text",
        attributes=["cat"],
        header_columns=["speaker"],
    )
