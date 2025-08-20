import asyncio
import pandas as pd
from unittest.mock import patch, AsyncMock

import gabriel
from gabriel.tasks.merge import Merge, MergeConfig


async def _run_merge(tmp_path):
    cfg = MergeConfig(save_dir=str(tmp_path), use_dummy=True, use_embeddings=False)
    task = Merge(cfg)
    df1 = pd.DataFrame({"id": [1, 2, 3], "term": ["apple", "banana", "pear"]})
    df2 = pd.DataFrame({"val": [10, 20, 30], "term": ["Apple", "Pear", "orange"]})
    merged = await task.run(df1, df2, on="term")
    return merged


def test_merge_dummy(tmp_path):
    merged = asyncio.run(_run_merge(tmp_path))
    assert "val" in merged.columns
    assert len(merged) == 3


def test_merge_api(tmp_path):
    df1 = pd.DataFrame({"id": [1, 2, 3], "term": ["apple", "banana", "pear"]})
    df2 = pd.DataFrame({"val": [10, 20, 30], "term": ["Apple", "Pear", "orange"]})
    merged = asyncio.run(
        gabriel.merge(
            df1,
            df2,
            on="term",
            save_dir=str(tmp_path),
            use_dummy=True,
            use_embeddings=False,
        )
    )
    assert "val" in merged.columns
    assert len(merged) == 3


@patch("gabriel.tasks.merge.get_all_responses", new_callable=AsyncMock)
def test_merge_trailing_space(mock_resp, tmp_path):
    mock_resp.return_value = pd.DataFrame(
        {"Identifier": ["merge_00000"], "Response": ['{"apple ": "Apple"}']}
    )
    cfg = MergeConfig(save_dir=str(tmp_path), use_embeddings=False)
    task = Merge(cfg)
    df1 = pd.DataFrame({"term": ["apple"]})
    df2 = pd.DataFrame({"val": [1], "term": ["Apple"]})
    merged = asyncio.run(task.run(df1, df2, on="term"))
    assert merged["val"].iloc[0] == 1
