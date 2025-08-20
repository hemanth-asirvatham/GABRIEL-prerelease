import asyncio
import pandas as pd

from gabriel.tasks.deduplicate import Deduplicate, DeduplicateConfig


async def _run_dedup(tmp_path):
    cfg = DeduplicateConfig(save_dir=str(tmp_path), use_dummy=True, use_embeddings=False)
    task = Deduplicate(cfg)
    df = pd.DataFrame({"term": ["apple", "Apple", "banana", "BANANA", "pear"]})
    return await task.run(df, on="term")


def test_deduplicate_dummy(tmp_path):
    result = asyncio.run(_run_dedup(tmp_path))
    assert "mapped_term" in result.columns
    assert result["mapped_term"].tolist() == ["apple", "apple", "banana", "banana", "pear"]
