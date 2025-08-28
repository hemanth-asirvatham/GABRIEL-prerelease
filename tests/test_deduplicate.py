import asyncio
import pandas as pd

import gabriel.tasks.deduplicate as deduplicate
from gabriel.tasks.deduplicate import Deduplicate, DeduplicateConfig


async def _run_dedup(tmp_path, n_runs=1):
    cfg = DeduplicateConfig(save_dir=str(tmp_path), use_dummy=True, use_embeddings=False, n_runs=n_runs)
    task = Deduplicate(cfg)
    df = pd.DataFrame({"term": ["apple", "Apple", "banana", "BANANA", "pear"]})
    return await task.run(df, column_name="term")


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


def test_prompt_contains_terms_with_embeddings(monkeypatch, tmp_path):
    captured = {}

    async def fake_get_all_responses(*, prompts, identifiers, **kwargs):
        captured["prompts"] = prompts
        return pd.DataFrame({"Identifier": identifiers, "Response": ["{}"] * len(identifiers)})

    monkeypatch.setattr(deduplicate, "get_all_responses", fake_get_all_responses)

    cfg = DeduplicateConfig(
        save_dir=str(tmp_path),
        use_dummy=True,
        use_embeddings=True,
        group_size=2,
        n_runs=1,
    )
    task = Deduplicate(cfg)
    df = pd.DataFrame({"term": ["apple", "Apple", "banana", "BANANA", "pear"]})
    asyncio.run(task.run(df, column_name="term"))

    assert "prompts" in captured
    prompt = captured["prompts"][0]
    assert "BEGIN RAW TERMS" in prompt
    assert "END RAW TERMS" in prompt
    body = prompt.split("BEGIN RAW TERMS", 1)[1].split("END RAW TERMS", 1)[0].strip()
    assert body != ""
