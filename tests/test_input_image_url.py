import asyncio
import openai
from gabriel.utils import openai_utils

class DummyClient:
    """Minimal stand‑in for the OpenAI client used in tests.

    The object captures the parameters passed to ``create`` so the test can
    inspect them without making real API calls.  The method is ``async`` to
    mirror the behaviour of the real client.
    """

    def __init__(self):
        self.responses = self

    async def create(self, **kwargs):
        DummyClient.captured = kwargs

        class Resp:
            output_text = "ok"

        return Resp()


def test_get_response_encodes_image(monkeypatch):
    """Verify that image strings are converted to data URLs before sending."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    openai_utils.client_async = None
    dummy = DummyClient()
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_: dummy)
    asyncio.run(openai_utils.get_response("Describe", images=["abc"], use_dummy=False))
    assert (
        DummyClient.captured["input"][0]["content"][1]["image_url"]
        == "data:image/jpeg;base64,abc"
    )
