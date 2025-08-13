import asyncio
from gabriel.utils import openai_utils


def test_get_all_responses_reset_files(tmp_path):
    save_path = tmp_path / "out.csv"
    asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a", "b"],
            identifiers=["1", "2"],
            save_path=str(save_path),
            use_dummy=True,
        )
    )
    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["b"],
            identifiers=["2"],
            save_path=str(save_path),
            use_dummy=True,
            reset_files=True,
        )
    )
    assert set(df["Identifier"]) == {"2"}
