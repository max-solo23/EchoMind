from core.llm.providers.gemini import GeminiProvider


class AsyncResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}


class AsyncClientSpy:
    def __init__(self):
        self.called = False

    async def post(self, *args, **kwargs):
        self.called = True
        return AsyncResponse()


async def test_gemini_complete_uses_async_client():
    provider = GeminiProvider(api_key="test-key")
    async_client = AsyncClientSpy()
    provider._async_client = async_client

    response = await provider.complete(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert response.message.content == "Hello"
    assert async_client.called is True
