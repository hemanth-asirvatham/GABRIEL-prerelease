import pytest
import openai
from gabriel.utils import openai_utils

class DummyClient:
    def __init__(self):
        self.responses = self
    async def create(self, **kwargs):
        DummyClient.captured = kwargs
        class Resp:
            output_text = "ok"
        return Resp()

@pytest.mark.asyncio
async def test_get_response_encodes_image(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    openai_utils.client_async = None
    dummy = DummyClient()
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda: dummy)
    await openai_utils.get_response("Describe", images=["abc"], use_dummy=False)
    assert DummyClient.captured["input"][0]["content"][1]["image_url"] == "data:image/jpeg;base64,abc"
