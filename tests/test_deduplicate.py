import asyncio
import pandas as pd

import gabriel


def test_deduplicate_basic(tmp_path):
    df = pd.DataFrame(
        {
            "name": ["Bob", "bob", "Bob!", "Bób", "Alice", "ALICE", "Eve"],
        }
    )

    out = asyncio.run(
        gabriel.deduplicate(
            df,
            "name",
            save_dir=str(tmp_path),
            use_dummy=True,
            use_embeddings=False,
        )
    )

    assert out["mapped_name"].tolist() == [
        "Bob",
        "Bob",
        "Bob",
        "Bob",
        "Alice",
        "Alice",
        "Eve",
    ]

