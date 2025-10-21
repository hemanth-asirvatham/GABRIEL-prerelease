import pandas as pd
import pytest

from gabriel.utils.passage_viewer import _normalize_structured_dataframe


def test_view_coded_passages_colab_runs():
    pytest.importorskip("IPython")
    from gabriel.utils.passage_viewer import view_coded_passages

    df = pd.DataFrame({"text": ["A snippet"], "cat": [["A snippet"]]})
    # Should not raise when using the lightweight HTML viewer
    view_coded_passages(df, "text", ["cat"])


def test_view_coded_passages_colab_with_headers():
    pytest.importorskip("IPython")
    from gabriel.utils.passage_viewer import view_coded_passages

    df = pd.DataFrame(
        {
            "text": ["A snippet"],
            "cat": [["A snippet"]],
            "speaker": ["Test Speaker"],
        }
    )
    view_coded_passages(
        df,
        "text",
        ["cat"],
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


def test_view_coded_passages_handles_stringified_inputs():
    pytest.importorskip("IPython")
    from gabriel.utils.passage_viewer import view_coded_passages

    df = pd.DataFrame(
        {
            "text": ["A snippet of text used for testing."],
            "cat": ['["A snippet of text used for testing."]'],
            "coded_passages": ['{"cat": ["A snippet of text used for testing."]}'],
        }
    )

    view_coded_passages(df, "text", ["cat"])
    view_coded_passages(df, "text", "coded_passages")


def test_top_level_view_coded_passages_colab_runs():
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
    gabriel.view_coded_passages(
        df,
        "text",
        ["cat"],
        header_columns=["speaker"],
    )
