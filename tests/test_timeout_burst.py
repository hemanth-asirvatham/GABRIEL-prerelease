import asyncio
from pathlib import Path
from collections import defaultdict

import pandas as pd

from gabriel.utils import openai_utils


def test_timeout_burst_restarts_and_resumes(tmp_path: Path) -> None:
    attempts = defaultdict(int)
    first_timeout = {"fired": False}

    async def flaky_responder(prompt: str, **_: object):
        attempts[prompt] += 1
        # Only the very first overall request times out; all subsequent calls succeed.
        if not first_timeout["fired"]:
            first_timeout["fired"] = True
            await asyncio.sleep(0)
            raise asyncio.TimeoutError("simulated timeout")
        return [f"ok-{prompt}"], 0.01, []

    save_path = tmp_path / "responses.csv"
    df: pd.DataFrame = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["p1", "p2"],
            identifiers=["p1", "p2"],
            response_fn=flaky_responder,
            use_dummy=False,
            save_path=str(save_path),
            reset_files=True,
            timeout_burst_threshold=1,  # trigger on the very first timeout
            timeout_burst_window=10.0,
            timeout_burst_cooldown=0.01,
            timeout_burst_max_restarts=2,
            dynamic_timeout=False,  # keep deterministic for the test
            max_retries=1,
            n_parallels=1,
            logging_level="error",
        )
    )

    assert set(df["Identifier"]) == {"p1", "p2"}
    assert len(df) == 2
    # At least one timeout should have fired, triggering the burst handler, and the run should return without hanging.
    assert first_timeout["fired"]
