import asyncio
from unittest.mock import AsyncMock, patch

import pandas as pd

import gabriel
from gabriel.tasks.filter import Filter, FilterConfig


async def _run_filter(tmp_path):
    cfg = FilterConfig(
        condition="contains a",
        save_dir=str(tmp_path),
        entities_per_call=2,
        shuffle=False,
    )
    task = Filter(cfg)
    df = pd.DataFrame({"tech": ["apple", "banana", "orange"]})
    return await task.run(df, "tech")


@patch("gabriel.tasks.filter.get_all_responses", new_callable=AsyncMock)
def test_filter_task(mock_resp, tmp_path):
    mock_resp.return_value = pd.DataFrame(
        {
            "Identifier": ["filter_000_00000", "filter_000_00001"],
            "Response": [
                '{"entities meeting condition": ["apple"]}',
                '{"entities meeting condition": ["orange"]}',
            ],
        }
    )
    result = asyncio.run(_run_filter(tmp_path))
    assert list(result["meets_condition"]) == [True, False, True]


@patch("gabriel.tasks.filter.get_all_responses", new_callable=AsyncMock)
def test_filter_api(mock_resp, tmp_path):
    mock_resp.return_value = pd.DataFrame(
        {
            "Identifier": ["filter_000_00000", "filter_000_00001"],
            "Response": [
                '{"entities meeting condition": ["apple"]}',
                '{"entities meeting condition": ["orange"]}',
            ],
        }
    )
    df = pd.DataFrame({"tech": ["apple", "banana", "orange"]})
    result = asyncio.run(
        gabriel.filter(
            df,
            "tech",
            condition="contains a",
            save_dir=str(tmp_path),
            entities_per_call=2,
            shuffle=False,
        )
    )
    assert list(result["meets_condition"]) == [True, False, True]


@patch("gabriel.tasks.filter.get_all_responses", new_callable=AsyncMock)
def test_filter_task_n_runs(mock_resp, tmp_path):
    mock_resp.return_value = pd.DataFrame(
        {
            "Identifier": [
                "filter_000_00000",
                "filter_000_00001",
                "filter_001_00000",
                "filter_001_00001",
                "filter_002_00000",
                "filter_002_00001",
            ],
            "Response": [
                '{"entities meeting condition": ["apple"]}',
                '{"entities meeting condition": ["orange"]}',
                '{"entities meeting condition": ["apple", "banana"]}',
                '{"entities meeting condition": []}',
                '{"entities meeting condition": []}',
                '{"entities meeting condition": ["orange"]}',
            ],
        }
    )
    cfg = FilterConfig(
        condition="contains a",
        save_dir=str(tmp_path),
        entities_per_call=2,
        shuffle=False,
        n_runs=3,
    )
    task = Filter(cfg)
    df = pd.DataFrame({"tech": ["apple", "banana", "orange"]})
    result = asyncio.run(task.run(df, "tech"))
    assert list(result["meets_condition_run_1"]) == [True, False, True]
    assert list(result["meets_condition_run_2"]) == [True, True, False]
    assert list(result["meets_condition_run_3"]) == [False, False, True]
    assert list(result["meets_condition"]) == [True, False, True]
