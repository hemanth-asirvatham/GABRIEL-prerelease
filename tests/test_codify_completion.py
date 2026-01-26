import asyncio

from gabriel.tasks.codify import ChunkResult, Codify, CodifyConfig


def test_completion_loop_respects_round_limit(monkeypatch, tmp_path):
    cfg = CodifyConfig(save_dir=str(tmp_path), n_rounds=2)
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
    cfg = CodifyConfig(save_dir=str(tmp_path), n_rounds=1)
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


def test_consolidate_snippets_handles_non_string_excerpts(tmp_path):
    cfg = CodifyConfig(save_dir=str(tmp_path), n_rounds=1)
    codify = Codify(cfg)

    chunk_results = [
        ChunkResult(
            identifier="row-0-chunk-0",
            chunk_text="No numeric snippets here.",
            data={"cat": [{"beginning excerpt": 123, "ending excerpt": None}]},
        )
    ]

    snippets = codify.consolidate_snippets("Plain text", chunk_results, "cat")

    assert snippets == []


def test_completion_loop_accumulates_snippets_across_rounds(monkeypatch, tmp_path):
    cfg = CodifyConfig(save_dir=str(tmp_path), n_rounds=3)
    codify = Codify(cfg)

    aggregated = {0: {"cat": []}}
    original_texts = ["Example text"]
    raw_values = ["Example text"]
    categories = {"cat": "Example"}

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
        # Flag the same category on both completion iterations so that
        # we can verify snippets from each round are preserved.
        return {0: {"cat"}} if iteration in {0, 1} else {}

    async def fake_gather(self, row_texts, **kwargs):
        iteration = int(kwargs.get("iteration", -1))
        return {idx: {"cat": [f"snippet-round-{iteration}"]} for idx in row_texts}

    monkeypatch.setattr(Codify, "_classify_remaining", fake_classify)
    monkeypatch.setattr(Codify, "_gather_iteration", fake_gather)

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

    assert result[0]["cat"] == ["snippet-round-1", "snippet-round-2"]
