import asyncio

import pytest

from gabriel.utils.openai_utils import get_all_embeddings


def test_get_all_embeddings_uses_cache(tmp_path, capsys):
    save_path = tmp_path / "emb.pkl"
    texts = ["a", "b"]

    asyncio.run(
        get_all_embeddings(
            texts=texts,
            identifiers=texts,
            save_path=str(save_path),
            use_dummy=True,
            verbose=False,
        )
    )

    capsys.readouterr()

    asyncio.run(
        get_all_embeddings(
            texts=texts,
            identifiers=texts,
            save_path=str(save_path),
            use_dummy=True,
            verbose=False,
        )
    )

    captured = capsys.readouterr()
    out = captured.out
    assert "Loaded 2 existing embeddings" in out
    assert "Using cached embeddings" in out


def test_get_all_embeddings_dummy_default(tmp_path):
    save_path = tmp_path / "emb.pkl"
    texts = ["hello", "world"]
    result = asyncio.run(
        get_all_embeddings(
            texts=texts,
            identifiers=["1", "2"],
            save_path=str(save_path),
            use_dummy=True,
            verbose=False,
        )
    )
    assert set(result.keys()) == {"1", "2"}
    assert all(isinstance(vec, list) and vec for vec in result.values())


def test_get_all_embeddings_dummy_override(tmp_path):
    save_path = tmp_path / "emb.pkl"
    overrides = {
        "first": [0.1, 0.2, 0.3],
        "*": [9.0, 8.0, 7.0],
    }
    result = asyncio.run(
        get_all_embeddings(
            texts=["a", "b"],
            identifiers=["first", "second"],
            save_path=str(save_path),
            use_dummy=True,
            dummy_embeddings=overrides,
            verbose=False,
        )
    )
    assert result["first"] == [0.1, 0.2, 0.3]
    assert result["second"] == [9.0, 8.0, 7.0]
