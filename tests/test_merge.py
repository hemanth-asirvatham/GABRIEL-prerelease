import asyncio
import pandas as pd
from unittest.mock import patch

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
