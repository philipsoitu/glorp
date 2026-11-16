"""Official SDK, one pooled client per connection, exactly one attempt per call."""

import inspect

from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from ..config import Provider


def serialize(value) -> dict:
    # exclude_unset retains unknown SDK extra fields but does not invent null defaults.
    return value.model_dump(mode="json", exclude_unset=True)


class OpenAIAdapter:
    def __init__(self, provider: Provider, *, http_client=None):
        if http_client is not None:
            http_client.follow_redirects = False
        self.client = AsyncOpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            timeout=300.0,
            max_retries=0,
            http_client=http_client if http_client is not None else DefaultAsyncHttpxClient(follow_redirects=False),
        )

    async def create(self, endpoint: str, body: dict):
        method = (
            self.client.chat.completions.create
            if endpoint.endswith("chat/completions")
            else self.client.responses.create
        )
        # Never interpret body fields as SDK transport options (headers, query, timeout).
        known = set(inspect.signature(method).parameters) - {"extra_headers", "extra_query", "extra_body", "timeout"}
        arguments = {k: v for k, v in body.items() if k in known}
        extra = {k: v for k, v in body.items() if k not in known}
        return await method(**arguments, extra_body=extra or None)

    async def close(self):
        await self.client.close()
