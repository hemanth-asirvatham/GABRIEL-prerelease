import asyncio
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

