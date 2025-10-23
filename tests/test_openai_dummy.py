import asyncio

import pytest

from gabriel.utils import openai_utils
from gabriel.utils.openai_utils import DummyResponseSpec


@pytest.mark.parametrize("prompts", [["hello world"], ["alpha", "beta"]])
def test_dummy_mode_infers_usage_without_specs(prompts, tmp_path):
    identifiers = [f"row{i}" for i in range(len(prompts))]
    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=prompts,
            identifiers=identifiers,
            save_path=str(tmp_path / "auto.csv"),
            use_dummy=True,
            reset_files=True,
        )
    )
    assert set(df["Identifier"]) == set(identifiers)
    assert (df["Input Tokens"] > 0).all()
    assert (df["Output Tokens"] > 0).all()
    assert df["Response IDs"].apply(lambda ids: isinstance(ids, list) and ids).all()


def test_dummy_responses_override_usage_and_cost(tmp_path, capsys, caplog):
    specs = {
        "1": DummyResponseSpec(
            responses=["alpha"],
            duration=0.42,
            input_tokens=11,
            output_tokens=7,
            reasoning_tokens=3,
            reasoning_summary="chain",
            response_id="resp-1",
            warning="dummy warning triggered",
        ),
        "*": {
            "input_tokens": 5,
            "output_tokens": 4,
            "error_log": ["fallback issue"],
        },
    }
    with caplog.at_level("WARNING"):
        df = asyncio.run(
            openai_utils.get_all_responses(
                prompts=["first", "second"],
                identifiers=["1", "2"],
                save_path=str(tmp_path / "dummy.csv"),
                use_dummy=True,
                dummy_responses=specs,
                reasoning_summary="brief",
                model="gpt-4o-mini",
                reset_files=True,
            )
        )
    output = capsys.readouterr().out
    assert "Actual total cost" in output
    assert any("dummy warning triggered" in rec.message for rec in caplog.records)
    row1 = df.loc[df["Identifier"] == "1"].iloc[0]
    assert row1["Input Tokens"] == 11
    assert row1["Output Tokens"] == 7
    assert row1["Reasoning Tokens"] == 3
    assert row1["Reasoning Summary"] == "chain"
    assert row1["Time Taken"] == pytest.approx(0.42)
    assert row1["Response IDs"][0] == "resp-1"
    row2 = df.loc[df["Identifier"] == "2"].iloc[0]
    assert row2["Input Tokens"] == 5
    assert row2["Output Tokens"] == 4
    assert bool(row2["Successful"])
    assert row2["Error Log"] == ["fallback issue"]


def test_dummy_responses_can_mark_failures(tmp_path, caplog):
    specs = {
        "fail": DummyResponseSpec(
            responses=["bad"],
            input_tokens=3,
            output_tokens=2,
            successful=False,
            error_log=["boom"],
        )
    }
    with caplog.at_level("WARNING"):
        df = asyncio.run(
            openai_utils.get_all_responses(
                prompts=["ok", "bad"],
                identifiers=["ok", "fail"],
                save_path=str(tmp_path / "fail.csv"),
                use_dummy=True,
                dummy_responses=specs,
                reset_files=True,
            )
        )
    fail_row = df.loc[df["Identifier"] == "fail"].iloc[0]
    assert not bool(fail_row["Successful"])
    assert fail_row["Response"] == ["bad"]
    assert fail_row["Error Log"] == ["boom"]
    assert any("requests failed" in rec.message for rec in caplog.records)
    ok_row = df.loc[df["Identifier"] == "ok"].iloc[0]
    assert bool(ok_row["Successful"])
