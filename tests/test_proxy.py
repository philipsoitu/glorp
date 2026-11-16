import asyncio
import json
import sqlite3
from copy import deepcopy

import httpx2
import pytest
from conftest import CHAT, RESPONSE, Harness, config_for
from openai import AsyncOpenAI

from glorp.config import load_config
from glorp.storage import Store


async def test_routes_headers_models_and_health(harness):
    async with harness.client() as client:
        assert (await client.get("/health")).json() == {"status": "ok"}
        models = (await client.get("/v1/models")).json()["data"]
        assert [(m["id"], m["name"], m["owned_by"]) for m in models] == [
            ("chat-id", "Chat Name", "remote"),
            ("local-id", "Local Name", "local"),
            ("responses-id", "Responses Name", "responses"),
        ]
        for body, endpoint in [
            ({"model": "Chat Name"}, "chat/completions"),
            ({"model": "responses-id", "input": "hello"}, "chat/completions"),
            ({"model": "chat-id", "input": "hello"}, "responses"),
            ({"model": "chat-id", "messages": [], "stream": 1}, "chat/completions"),
        ]:
            response = await client.post("/v1/" + endpoint, json=body)
            assert response.status_code == 400 and "error" in response.json()
        for content in ("[", "[]", '{"model":"chat-id","stream":NaN}'):
            assert (await client.post("/v1/chat/completions", content=content)).status_code == 400
        assert not harness.calls
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "local-id", "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "Authorization": "Bearer INBOUND_SECRET",
                "Cookie": "secret-cookie",
                "X-Arbitrary": "no",
                "X-Glorp-Session-Id": "session",
                "X-Glorp-Run-Id": "run",
            },
        )
        assert response.status_code == 200 and response.json() == CHAT
        assert response.headers["x-glorp-request-id"] and response.headers["x-glorp-conversation-id"]
        call = harness.calls[0]
        assert call.url.host == "localhost" and harness.body()["model"] == "local-id"
        assert call.headers["authorization"] == "Bearer LOCAL_SECRET"
        assert not any(h in call.headers for h in ("cookie", "x-arbitrary", "x-glorp-session-id", "x-glorp-run-id"))
        row = (await harness.app.state.store.rows())[0]
        assert row["session_id"] == "session" and row["model"] == "local-id"
        assert row["turns"][-1]["upstream_request_id"] == "upstream-request"
        assert row["turns"][-1]["returned_model"] == CHAT["model"]
        assert "SECRET" not in json.dumps(row)


async def test_agent_fields_and_unknown_fields_are_not_dropped(harness):
    body = {
        "model": "chat-id",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "developer", "content": "developer"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "https://image.invalid/x"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c", "content": "tool result"},
        ],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "response_format": {"type": "json_schema", "json_schema": {"name": "data", "schema": {"type": "object"}}},
        "reasoning_effort": "high",
        "vendor_extension": {"opaque": [1, 2]},
        "extra_headers": {"X-Do-Not-Forward": "body-only"},
        "timeout": 1,
    }
    async with harness.client() as client:
        response = await client.post("/v1/chat/completions", json=body)
        assert response.json() == CHAT
        assert harness.body() == body
        assert "x-do-not-forward" not in harness.calls[0].headers
        row = (await harness.app.state.store.rows())[0]
        assert row["context"] == body["messages"] + [CHAT["choices"][0]["message"]]
        assert row["turns"][0]["options"] == {k: v for k, v in body.items() if k != "messages"}
        assert "messages" not in row["turns"][0]["options"]


async def test_responses_passthrough_and_previous_reference(harness):
    body = {
        "model": "responses-id",
        "input": "hello",
        "instructions": "do things",
        "reasoning": {"effort": "high"},
        "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}, "strict": False}],
        "text": {"format": {"type": "json_object"}},
        "vendor": ["extra"],
    }
    async with harness.client() as client:
        response = await client.post("/v1/responses", json=body)
        assert response.json() == RESPONSE
        assert harness.body() == body
        row = (await harness.app.state.store.rows())[0]
        assert row["context"] == [{"role": "user", "content": "hello"}] + RESPONSE["output"]
        result = await client.post(
            "/v1/responses",
            json={
                "model": "responses-id",
                "previous_response_id": "resp-one",
                "instructions": "do things",
                "input": [{"type": "function_call_output", "call_id": "call-1", "output": "result"}],
            },
        )
        assert result.headers["x-glorp-conversation-id"] == response.headers["x-glorp-conversation-id"]
        assert len(await harness.app.state.store.rows()) == 1
        assert harness.body()["input"] == [{"type": "function_call_output", "call_id": "call-1", "output": "result"}]
        assert (
            harness.body()["previous_response_id"] == "resp-one"
        )  # Local reconstruction does not rewrite upstream input.


async def test_two_turns_model_switch_and_unknown_conversation(harness):
    async with harness.client() as client:
        initial = [{"role": "user", "content": "hello"}]
        first = await client.post("/v1/chat/completions", json={"model": "chat-id", "messages": initial})
        history = initial + [CHAT["choices"][0]["message"], {"role": "user", "content": "next"}]
        second = await client.post("/v1/chat/completions", json={"model": "local-id", "messages": history})
        assert first.headers["x-glorp-conversation-id"] == second.headers["x-glorp-conversation-id"]
        row = (await harness.app.state.store.rows())[0]
        assert len(row["context"]) == 4 and [t["model"] for t in row["turns"]] == ["chat-id", "local-id"]
        before = len(harness.calls)
        invalid = await client.post(
            "/v1/chat/completions",
            json={"model": "chat-id", "messages": initial},
            headers={"X-Glorp-Conversation-Id": "unknown"},
        )
        assert invalid.status_code == 400 and len(harness.calls) == before


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_upstream_status_no_retries_and_safe_errors(tmp_path, status, caplog):
    async def handler(request):
        return httpx2.Response(
            status,
            json={
                "error": {
                    "message": "bad UPSTREAM_SECRET INBOUND_SECRET",
                    "type": "rate_limit_error",
                    "authorization": "Bearer UPSTREAM_SECRET",
                    "code": "limit",
                }
            },
            headers={"x-request-id": "failed-upstream"},
        )

    harness = Harness(config_for(tmp_path, "INBOUND_SECRET"), handler)
    async with harness.client() as client:
        assert (await client.get("/health")).status_code == 401
        assert (await client.get("/health", headers={"Authorization": "Bearer wrong"})).status_code == 401
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "chat-id", "messages": []},
            headers={"Authorization": "Bearer INBOUND_SECRET"},
        )
        assert response.status_code == status and len(harness.calls) == 1
        assert response.json()["error"]["code"] == "limit"
        assert "SECRET" not in response.text + json.dumps(await harness.app.state.store.rows()) + caplog.text
        assert response.headers["x-glorp-conversation-id"]


@pytest.mark.parametrize("exception,status", [(httpx2.ReadTimeout, 504), (httpx2.ConnectError, 502)])
async def test_transport_errors_no_retries(tmp_path, exception, status):
    async def handler(request):
        raise exception("UPSTREAM_SECRET must not leak", request=request)

    harness = Harness(config_for(tmp_path), handler)
    async with harness.client() as client:
        response = await client.post("/v1/chat/completions", json={"model": "chat-id", "messages": []})
        assert response.status_code == status and len(harness.calls) == 1
        assert "SECRET" not in response.text + json.dumps(await harness.app.state.store.rows())


async def test_database_failures(harness, monkeypatch, caplog):
    async def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk failure")

    async with harness.client() as client:
        store = harness.app.state.store
        original = store.reserve
        monkeypatch.setattr(store, "reserve", fail)
        response = await client.post("/v1/chat/completions", json={"model": "chat-id", "messages": []})
        assert response.status_code == 503 and not harness.calls
        monkeypatch.setattr(store, "reserve", original)
        monkeypatch.setattr(store, "finalize", fail)
        response = await client.post("/v1/chat/completions", json={"model": "chat-id", "messages": []})
        assert response.status_code == 503 and len(harness.calls) == 1
        assert "persistence failed" in caplog.text


async def test_startup_unwritable_location(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    harness = Harness(config_for(tmp_path), store_factory=lambda _: Store(blocker / "db"))
    with pytest.raises(OSError):
        async with harness.client():
            pytest.fail("startup should fail")
    assert not harness.calls


async def test_official_client_public_surface(harness):
    async with harness.app.router.lifespan_context(harness.app):
        async with AsyncOpenAI(
            base_url="http://proxy/v1",
            api_key="unused",
            max_retries=0,
            http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=harness.app)),
        ) as client:
            chat = await client.chat.completions.create(model="chat-id", messages=[{"role": "user", "content": "hi"}])
            assert chat.choices[0].message.content == "hi"
            response = await client.responses.create(model="responses-id", input="hello")
            assert response.output[0].type == "reasoning"


async def test_multiple_choices_and_concurrent_calls(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def handler(request):
        nonlocal count
        count += 1
        if count == 2:
            entered.set()
        await release.wait()
        result = deepcopy(CHAT)
        result["choices"].append(
            {"index": 1, "message": {"role": "assistant", "content": "alternative"}, "finish_reason": "stop"}
        )
        return httpx2.Response(200, json=result)

    harness = Harness(config_for(tmp_path), handler)
    async with harness.client() as client:

        async def send():
            return await client.post(
                "/v1/chat/completions",
                json={"model": "chat-id", "messages": [{"role": "user", "content": "hi"}], "n": 2},
            )

        tasks = [asyncio.create_task(send()), asyncio.create_task(send())]
        await asyncio.wait_for(entered.wait(), 2)
        release.set()
        responses = await asyncio.gather(*tasks)
        rows = await harness.app.state.store.rows()
        assert len(rows) == 4 and all(len(r["context"]) == 2 for r in rows)
        assert len({r.headers["x-glorp-conversation-id"] for r in responses}) == 2


async def test_environment_credentials_not_saved(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "ENV_SECRET")
    path = tmp_path / "config.yaml"
    path.write_text(
        'provider_list:\n  - name: p\n    base_url: https://example.invalid/v1\n    protocol: openai-completions\n    api_key: "${TEST_PROVIDER_KEY}"\n    models:\n      - id: chat-id\n        name: Chat\n'
    )
    config = load_config(path)

    async def handler(request):
        assert request.headers["authorization"] == "Bearer ENV_SECRET"
        return httpx2.Response(400, json={"error": {"message": "ENV_SECRET"}})

    harness = Harness(config, handler)
    async with harness.client() as client:
        response = await client.post("/v1/chat/completions", json={"model": "chat-id", "messages": []})
        assert response.status_code == 400
        assert "ENV_SECRET" not in response.text + json.dumps(await harness.app.state.store.rows())
