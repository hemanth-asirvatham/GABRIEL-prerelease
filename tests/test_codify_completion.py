import asyncio

from gabriel.tasks.codify import Codify, CodifyConfig


def test_completion_loop_respects_round_limit(monkeypatch, tmp_path):
    cfg = CodifyConfig(save_dir=str(tmp_path), completion_max_rounds=2)
    codify = Codify(cfg)

    aggregated = {0: {"cat": []}}
    original_texts = ["Example text"]
    raw_values = ["Example text"]
    categories = {"cat": "Example"}

    classify_calls = []
    gather_calls = []

    async def fake_classify(
        self,
        aggregated,
        original_texts,
        categories,
        additional_instructions,
        iteration,
        reset_files,
        **kwargs,
    ):
        classify_calls.append(iteration)
        if iteration == 0:
            return {0: {"cat"}}
        return {}

    async def fake_gather(self, row_texts, **kwargs):
        gather_calls.append(list(row_texts.keys()))
        return {idx: {"cat": [f"snippet-{idx}"]} for idx in row_texts}

    def fake_merge(self, destination, source):
        for row_idx, cat_map in source.items():
            dest_row = destination.setdefault(row_idx, {})
            for category, snippets in cat_map.items():
                dest_row.setdefault(category, []).extend(snippets)
        return bool(source)

    monkeypatch.setattr(Codify, "_classify_remaining", fake_classify)
    monkeypatch.setattr(Codify, "_gather_iteration", fake_gather)
    monkeypatch.setattr(Codify, "_merge_snippet_results", fake_merge)

    result = asyncio.run(
        codify._completion_loop(
            aggregated,
            original_texts,
            raw_values,
            categories,
            additional_instructions=None,
            reset_files=False,
        )
    )

    assert classify_calls == [0]
    assert gather_calls == [[0]]
    assert result[0]["cat"] == ["snippet-0"]


def test_completion_loop_skips_when_single_round(monkeypatch, tmp_path):
    cfg = CodifyConfig(save_dir=str(tmp_path), completion_max_rounds=1)
    codify = Codify(cfg)

    aggregated = {0: {"cat": []}}
    original_texts = ["Example text"]
    raw_values = ["Example text"]
    categories = {"cat": "Example"}

    async def fail_classify(*args, **kwargs):
        raise AssertionError("classification should not run when only one round is requested")

    monkeypatch.setattr(Codify, "_classify_remaining", fail_classify)

    result = asyncio.run(
        codify._completion_loop(
            aggregated,
            original_texts,
            raw_values,
            categories,
            additional_instructions=None,
            reset_files=False,
        )
    )

    assert result == aggregated
