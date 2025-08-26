import asyncio
import pandas as pd

from gabriel.tasks.deduplicate import Deduplicate, DeduplicateConfig


async def _run_dedup(tmp_path, n_runs=1):
    cfg = DeduplicateConfig(save_dir=str(tmp_path), use_dummy=True, use_embeddings=False, n_runs=n_runs)
    task = Deduplicate(cfg)
    df = pd.DataFrame({"term": ["apple", "Apple", "banana", "BANANA", "pear"]})
    return await task.run(df, on="term")


def test_deduplicate_dummy(tmp_path):
    result = asyncio.run(_run_dedup(tmp_path, n_runs=1))
    assert "mapped_term" in result.columns
    assert result["mapped_term"].tolist() == ["apple", "apple", "banana", "banana", "pear"]


def test_deduplicate_multiple_runs(tmp_path):
    result = asyncio.run(_run_dedup(tmp_path, n_runs=2))
    assert "mapped_term_run1" in result.columns
    assert "mapped_term_final" in result.columns
    assert "mapped_term" in result.columns
    assert result["mapped_term"].tolist() == ["apple", "apple", "banana", "banana", "pear"]
