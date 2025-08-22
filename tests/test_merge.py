import asyncio
import pandas as pd
from unittest.mock import patch, AsyncMock
from pathlib import Path

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
        {"Identifier": ["merge_00_00000"], "Response": ['{"apple ": "Apple"}']}
    )
    cfg = MergeConfig(save_dir=str(tmp_path), use_embeddings=False)
    task = Merge(cfg)
    df1 = pd.DataFrame({"term": ["apple"]})
    df2 = pd.DataFrame({"val": [1], "term": ["Apple"]})
    merged = asyncio.run(task.run(df1, df2, on="term"))
    assert merged["val"].iloc[0] == 1


@patch("gabriel.tasks.merge.get_all_responses", new_callable=AsyncMock)
def test_merge_html_entities(mock_resp, tmp_path):
    mock_resp.return_value = pd.DataFrame(
        {"Identifier": ["merge_00_00000"], "Response": ['{"B. Pôssas": "B. Pôssas"}']}
    )
    cfg = MergeConfig(save_dir=str(tmp_path), use_embeddings=False)
    task = Merge(cfg)
    df1 = pd.DataFrame({"term": ["B. Pôssas"]})
    df2 = pd.DataFrame({"val": [1], "term": ["B. P&#244;ssas"]})
    merged = asyncio.run(task.run(df1, df2, on="term"))
    assert merged["val"].iloc[0] == 1


@patch("gabriel.tasks.merge.get_all_responses", new_callable=AsyncMock)
def test_merge_no_certain_match(mock_resp, tmp_path):
    mock_resp.return_value = pd.DataFrame(
        {"Identifier": ["merge_00_00000"], "Response": ['{"apple": "no certain match"}']}
    )
    cfg = MergeConfig(save_dir=str(tmp_path), use_embeddings=False)
    task = Merge(cfg)
    df1 = pd.DataFrame({"term": ["apple"]})
    df2 = pd.DataFrame({"val": [1], "term": ["Apple"]})
    merged = asyncio.run(task.run(df1, df2, on="term"))
    assert pd.isna(merged["val"].iloc[0])


@patch("gabriel.tasks.merge.get_all_responses", new_callable=AsyncMock)
def test_merge_max_attempts(mock_resp, tmp_path):
    mock_resp.side_effect = [
        pd.DataFrame({"Identifier": ["merge_00_00000"], "Response": ['{"apple": "Apple"}']}),
        pd.DataFrame({"Identifier": ["merge_01_00000"], "Response": ['{"banana": "Banana"}']}),
    ]
    cfg = MergeConfig(
        save_dir=str(tmp_path),
        use_embeddings=False,
        short_list_len=1,
        max_attempts=2,
    )
    task = Merge(cfg)
    df1 = pd.DataFrame({"term": ["apple", "banana"]})
    df2 = pd.DataFrame({"val": [1, 2], "term": ["Apple", "Banana"]})
    merged = asyncio.run(task.run(df1, df2, on="term"))
    assert set(merged["val"].dropna()) == {1, 2}


@patch("gabriel.tasks.merge.get_all_responses", new_callable=AsyncMock)
def test_merge_deduplicates(mock_resp, tmp_path):
    mock_resp.return_value = pd.DataFrame(
        {"Identifier": ["merge_00_00000"], "Response": ['{"apple": "Apple"}']}
    )
    cfg = MergeConfig(save_dir=str(tmp_path), use_embeddings=False)
    task = Merge(cfg)
    df1 = pd.DataFrame({"term": ["apple"]})
    df2 = pd.DataFrame({"val": [1, 2], "term": ["Apple", "Apple"]})
    merged = asyncio.run(task.run(df1, df2, on="term"))
    assert len(merged) == 1


@patch("gabriel.tasks.merge.get_all_responses", new_callable=AsyncMock)
def test_merge_progress_tracking(mock_resp, tmp_path):
    mock_resp.side_effect = [
        pd.DataFrame({"Identifier": ["merge_00_00000"], "Response": ['{"apple": "Apple"}']}),
        pd.DataFrame({"Identifier": ["merge_01_00000"], "Response": ['{"banana": "Banana"}']}),
    ]
    cfg = MergeConfig(
        save_dir=str(tmp_path),
        use_embeddings=False,
        short_list_len=1,
        max_attempts=2,
    )
    task = Merge(cfg)
    df1 = pd.DataFrame({"term": ["apple", "banana"]})
    df2 = pd.DataFrame({"val": [1, 2], "term": ["Apple", "Banana"]})
    merged = asyncio.run(task.run(df1, df2, on="term"))
    progress_path = Path(tmp_path) / "merge_progress.csv"
    assert progress_path.exists()
    progress = pd.read_csv(progress_path)
    assert list(progress["matches_this_round"]) == [1, 1]
    assert list(progress["total_matches"]) == [1, 2]
    assert list(progress["remaining"]) == [1, 0]


@patch("gabriel.tasks.merge.get_all_embeddings", new_callable=AsyncMock)
@patch("gabriel.tasks.merge.get_all_responses", new_callable=AsyncMock)
def test_merge_auto_match(mock_resp, mock_emb, tmp_path):
    async def dummy_get_all_embeddings(texts, identifiers=None, **kwargs):
        mapping = {
            "apple": [1.0, 0.0],
            "banana": [0.0, 1.0],
            "Apple": [1.0, 0.0],
            "Banana": [0.0, 1.0],
            "Pear": [-1.0, 0.0],
        }
        return {t: mapping[t] for t in texts}

    mock_emb.side_effect = dummy_get_all_embeddings
    cfg = MergeConfig(
        save_dir=str(tmp_path), use_embeddings=True, use_dummy=False, long_list_len=1
    )
    task = Merge(cfg)
    df1 = pd.DataFrame({"term": ["apple", "banana"]})
    df2 = pd.DataFrame({"val": [1, 2, 3], "term": ["Apple", "Banana", "Pear"]})
    merged = asyncio.run(task.run(df1, df2, on="term"))
    assert set(merged["val"].dropna()) == {1, 2}
    assert mock_resp.await_count == 0
