import asyncio
import json
import socket
from contextlib import asynccontextmanager
from copy import deepcopy

import httpx
import httpx2
import pytest
import uvicorn

from glorp.app import create_app
from glorp.config import Config, Model, Provider
from glorp.providers.openai import OpenAIAdapter

CHAT = {
    "id": "chat-one",
    "object": "chat.completion",
    "created": 123,
    "model": "model-version-1",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    "vendor_field": {"opaque": True},
}
RESPONSE = {
    "id": "resp-one",
    "object": "response",
    "created_at": 123,
    "model": "responses-version-1",
    "status": "completed",
    "output": [
        {
            "id": "rs-1",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "A summary"}],
            "encrypted_content": "opaque",
        },
        {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}],
        },
        {
            "type": "function_call",
            "id": "fc-1",
            "call_id": "call-1",
            "name": "search",
            "arguments": '{"q":"x"}',
            "status": "completed",
        },
    ],
    "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    "vendor_field": [1, 2],
}


def config_for(tmp_path, token=None):
    return Config(
        (
            Provider(
                "remote",
                "https://upstream.invalid/v1",
                "openai-completions",
                "UPSTREAM_SECRET",
                (Model("chat-id", "Chat Name"),),
            ),
            Provider(
                "local",
                "http://localhost:12345/v1",
                "openai-completions",
                "LOCAL_SECRET",
                (Model("local-id", "Local Name"),),
            ),
            Provider(
                "responses",
                "https://responses.invalid/v1",
                "openai-responses",
                "RESPONSES_SECRET",
                (Model("responses-id", "Responses Name"),),
            ),
        ),
        tmp_path / "transcripts.sqlite3",
        api_key=token,
    )


class Harness:
    def __init__(self, config, handler=None, store_factory=None):
        self.calls = []
        self.config = config
        self.handler = handler
        kwargs = {"store_factory": store_factory} if store_factory else {}
        self.app = create_app(config, adapter_factory=self.adapter, **kwargs)

    def adapter(self, provider):
        async def handle(request):
            self.calls.append(request)
            if self.handler:
                return await self.handler(request)
            data = RESPONSE if request.url.path.endswith("responses") else CHAT
            return httpx2.Response(200, json=deepcopy(data), headers={"x-request-id": "upstream-request"})

        return OpenAIAdapter(provider, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)))

    @asynccontextmanager
    async def client(self):
        async with self.app.router.lifespan_context(self.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app), base_url="http://proxy"
            ) as client:
                yield client

    def body(self, index=-1):
        return json.loads(self.calls[index].content)


@pytest.fixture
def harness(tmp_path):
    return Harness(config_for(tmp_path))


@asynccontextmanager
async def live_server(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False, timeout_graceful_shutdown=2))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        sock.close()
