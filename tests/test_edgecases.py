import asyncio
import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from uuid import uuid4

import httpx
import httpx2
import pytest
from conftest import RESPONSE, Harness, config_for, live_server
from openai import AsyncOpenAI
from test_streaming import UpstreamStream, chat_events, response_events, streaming_harness, wire

from glorp.conversations import shared_prefix
from glorp.providers.openai import OpenAIAdapter
from glorp.storage import Store


async def reserve(store, transcript, **kwargs):
    args = dict(
        request_id=str(uuid4()),
        transcript=transcript,
        endpoint="/v1/responses",
        protocol="openai-responses",
        provider="p",
        model="m",
        options={},
    )
    args.update(kwargs)
    return await store.reserve(**args)


async def test_structural_json_types_and_optional_responses_status():
    assert shared_prefix([{"unknown": True}], [{"unknown": 1}], "openai-completions") == 0
    assert shared_prefix([{"role": []}], [{"role": []}], "openai-completions") == 1
    item = {
        "id": "fc1",
        "type": "function_call",
        "call_id": "call1",
        "name": "f",
        "arguments": "{}",
        "status": "completed",
    }
    echoed = {k: v for k, v in item.items() if k != "status"}
    assert shared_prefix([item], [echoed], "openai-responses") == 1
    assert shared_prefix([item], [dict(echoed, call_id="call2")], "openai-responses") == 0


async def test_item_references_and_ambiguous_previous_ids(tmp_path):
    store = Store(tmp_path / "items.sqlite3")
    await store.open()
    try:
        user = {"role": "user", "content": "hello"}
        item = {"id": "m1", "role": "assistant", "type": "message", "content": "hi", "status": "completed"}
        first = await reserve(store, [user])
        await store.finalize(first, [(0, [item], {})], status="completed", response_id="r1")
        known = await reserve(store, [{"type": "item_reference", "id": "m1"}])
        row = next(r for r in await store.rows() if r["id"] == known.conversation_id)
        assert row["context"] == [item] and row["context_complete"] == 1
        assert row["turns"][-1]["proxy_options"]["resolved_item_references"] == [
            {"input_index": 0, "reference": {"type": "item_reference", "id": "m1"}}
        ]
        unknown = await reserve(store, [{"type": "item_reference", "id": "unseen"}])
        row = next(r for r in await store.rows() if r["id"] == unknown.conversation_id)
        assert row["context_complete"] == 0 and row["context"][0]["id"] == "unseen"
        other = await reserve(store, [{"role": "user", "content": "unrelated"}])
        await store.finalize(other, [(0, [dict(item, content="different")], {})], status="completed", response_id="r1")
        ambiguous = await reserve(
            store, [{"role": "user", "content": "follow-up"}], options={"previous_response_id": "r1"}
        )
        row = next(r for r in await store.rows() if r["id"] == ambiguous.conversation_id)
        assert row["context_complete"] == 0 and row["parent_id"] is None
    finally:
        await store.close()


async def test_atomic_alternative_output_rollback(tmp_path, monkeypatch):
    store = Store(tmp_path / "atomic.sqlite3")
    await store.open()
    try:
        reservation = await reserve(store, [{"role": "user", "content": "hello"}])
        before = await store.rows()

        async def fail_insert(row):
            raise sqlite3.OperationalError("disk failure after primary update")

        monkeypatch.setattr(store, "_insert", fail_insert)
        with pytest.raises(sqlite3.OperationalError):
            await store.finalize(
                reservation,
                [
                    (0, [{"role": "assistant", "content": "one"}], {}),
                    (1, [{"role": "assistant", "content": "two"}], {}),
                ],
                status="completed",
            )
        assert await store.rows() == before
    finally:
        await store.close()


@pytest.mark.parametrize("stream", [False, True])
async def test_responses_failure_redacts_errors_and_preserves_status(tmp_path, stream):
    result = dict(
        deepcopy(RESPONSE),
        status="failed",
        error={
            "code": "failed",
            "message": "UPSTREAM_SECRET RESPONSES_SECRET",
            "headers": {"Authorization": "secret"},
            "stack_trace": "private stack",
        },
    )
    upstream = None

    async def handler(request):
        nonlocal upstream
        if not stream:
            return httpx2.Response(200, json=result)
        events = response_events("failed")[:-1] + [
            wire({"type": "response.failed", "response": result}, "response.failed")
        ]
        upstream = UpstreamStream(events)
        return httpx2.Response(200, stream=upstream, headers={"content-type": "text/event-stream"})

    harness = Harness(config_for(tmp_path), handler)
    async with harness.client() as client:
        response = await client.post(
            "/v1/responses", json={"model": "responses/responses-id", "input": "hello", "stream": stream}
        )
        row = (await harness.app.state.store.rows())[0]
        assert response.status_code == 200 and row["status"] == "failed"
        assert row["turns"][-1]["response_status"] == "failed"
        assert "SECRET" not in response.text + json.dumps(row)
        assert "private stack" not in response.text + json.dumps(row)
        assert "Authorization" not in json.dumps(row)


async def test_non_json_error_retained_locally_not_returned(tmp_path):
    async def handler(request):
        return httpx2.Response(500, text="<html>UPSTREAM_SECRET private debug details</html>")

    harness = Harness(config_for(tmp_path), handler)
    async with harness.client() as client:
        result = await client.post("/v1/chat/completions", json={"model": "remote/chat-id", "messages": []})
        assert result.status_code == 500 and "private debug details" not in result.text
        error = (await harness.app.state.store.rows())[0]["turns"][-1]["error"]
        assert error["upstream_body"] == "<html>[REDACTED] private debug details</html>"


async def test_unknown_opposite_dialect_history_field_preserved_in_options(harness):
    async with harness.client() as client:
        for endpoint, body, unknown in [
            ("chat/completions", {"model": "remote/chat-id", "messages": [], "input": {"unknown": True}}, "input"),
            (
                "responses",
                {"model": "responses/responses-id", "input": "hello", "messages": {"unknown": True}},
                "messages",
            ),
        ]:
            result = await client.post("/v1/" + endpoint, json=body)
            assert result.status_code == 200 and harness.body() == dict(body, model=body["model"].split("/", 1)[1])
            row = next(
                r for r in await harness.app.state.store.rows() if r["id"] == result.headers["x-glorp-conversation-id"]
            )
            assert row["turns"][-1]["options"][unknown] == body[unknown]


async def test_empty_conversation_id_not_silently_ignored(harness):
    async with harness.client() as client:
        result = await client.post(
            "/v1/chat/completions",
            json={"model": "remote/chat-id", "messages": []},
            headers={"X-Glorp-Conversation-Id": ""},
        )
        assert result.status_code == 400 and not harness.calls


async def test_streaming_tools_and_refusal_equal_nonstream(tmp_path):
    message = {
        "role": "assistant",
        "content": "hi",
        "refusal": "no",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}},
            {"id": "c2", "type": "function", "function": {"name": "g", "arguments": "{}"}},
        ],
        "unknown_message_field": {"opaque": True},
    }
    chunks = []
    for delta, finish in [
        (
            {
                "role": "assistant",
                "content": "h",
                "refusal": "n",
                "unknown_message_field": {"opaque": True},
                "tool_calls": [
                    {"index": 1, "id": "c2", "type": "function", "function": {"name": "g", "arguments": "{"}},
                    {"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":'}},
                ],
            },
            None,
        ),
        (
            {
                "content": "i",
                "refusal": "o",
                "tool_calls": [
                    {"index": 0, "function": {"arguments": "1}"}},
                    {"index": 1, "function": {"arguments": "}"}},
                ],
            },
            "tool_calls",
        ),
    ]:
        chunks.append(
            wire(
                {
                    "id": "chat-one",
                    "model": "model-version-1",
                    "object": "chat.completion.chunk",
                    "created": 123,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
            )
        )
    upstream = UpstreamStream(chunks + [wire("[DONE]")])
    harness = streaming_harness(tmp_path, upstream)
    async with harness.client() as client:
        result = await client.post(
            "/v1/chat/completions", json={"model": "remote/chat-id", "messages": [], "stream": True}
        )
        assert "[DONE]" in result.text
        assert (await harness.app.state.store.rows())[0]["context"] == [message]


async def test_official_client_actual_tcp_stream(tmp_path):
    upstream = UpstreamStream(chat_events())
    harness = streaming_harness(tmp_path, upstream)
    async with live_server(harness.app) as base:
        async with AsyncOpenAI(base_url=base + "/v1", api_key="unused", max_retries=0) as client:
            stream = await client.chat.completions.create(model="remote/chat-id", messages=[], stream=True)
            text = ""
            async for chunk in stream:
                if chunk.choices:
                    text += chunk.choices[0].delta.content or ""
            assert text == "hi"
            assert (await harness.app.state.store.rows())[0]["status"] == "completed"


async def test_disconnect_while_waiting_for_headers(tmp_path):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def handler(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    harness = Harness(config_for(tmp_path), handler)
    async with live_server(harness.app) as base:
        async with httpx.AsyncClient(base_url=base, timeout=None) as client:
            call = asyncio.create_task(
                client.post("/v1/chat/completions", json={"model": "remote/chat-id", "messages": [], "stream": True})
            )
            await asyncio.wait_for(entered.wait(), 2)
            call.cancel()
            await asyncio.gather(call, return_exceptions=True)
        await asyncio.wait_for(cancelled.wait(), 2)
        async with asyncio.timeout(2):
            while (row := (await harness.app.state.store.rows())[0])["status"] == "in_progress":
                await asyncio.sleep(0.01)
        assert row["status"] == "cancelled" and row["context"] == []


async def test_chat_unknown_state_fields_do_not_reconstruct_responses_history(harness):
    async with harness.client() as client:
        first = await client.post(
            "/v1/chat/completions", json={"model": "remote/chat-id", "messages": [{"role": "user", "content": "one"}]}
        )
        result = await client.post(
            "/v1/chat/completions",
            json={
                "model": "remote/chat-id",
                "messages": [{"role": "user", "content": "two"}],
                "previous_response_id": "chat-one",
                "conversation": "vendor-field",
            },
        )
        assert result.status_code == 200
        assert result.headers["x-glorp-conversation-id"] != first.headers["x-glorp-conversation-id"]
        rows = await harness.app.state.store.rows()
        assert len(rows[-1]["context"]) == 2 and rows[-1]["context"][0]["content"] == "two"
        assert rows[-1]["context_complete"] == 1


async def test_responses_failed_terminal_empty_output_keeps_observed_partial(tmp_path):
    events = response_events("failed")
    failure = dict(deepcopy(RESPONSE), status="failed", output=[], error={"message": "failed", "code": "failure"})
    upstream = UpstreamStream(events[:-1] + [wire({"type": "response.failed", "response": failure}, "response.failed")])
    harness = streaming_harness(tmp_path, upstream)
    async with harness.client() as client:
        result = await client.post(
            "/v1/responses", json={"model": "responses/responses-id", "input": "hello", "stream": True}
        )
        assert "event: response.failed" in result.text
        row = (await harness.app.state.store.rows())[0]
        assert row["status"] == "failed" and row["turns"][-1]["partial_output"]
        assert row["context"][-1]["content"][0]["text"] == "hi"


async def test_responses_write_failure_does_not_emit_success_terminal(tmp_path, monkeypatch):
    upstream = UpstreamStream(response_events())
    harness = streaming_harness(tmp_path, upstream)
    async with harness.client() as client:

        async def fail(*args, **kwargs):
            raise sqlite3.OperationalError("disk full")

        monkeypatch.setattr(harness.app.state.store, "finalize", fail)
        result = await client.post(
            "/v1/responses", json={"model": "responses/responses-id", "input": "hello", "stream": True}
        )
        assert "response.output_text.delta" in result.text and "event: error" in result.text
        assert "event: response.completed" not in result.text
        assert upstream.closed.is_set()


async def test_future_schema_rejected(tmp_path):
    path = tmp_path / "future.sqlite3"
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA user_version=2")
    store = Store(path)
    with pytest.raises(RuntimeError, match="schema version"):
        await store.open()
    assert store.db is None


async def test_shutdown_closes_stalled_stream_and_saves_partial(tmp_path):
    upstream = UpstreamStream(chat_events(), gate=asyncio.Event())
    harness = streaming_harness(tmp_path, upstream)
    async with httpx.AsyncClient(timeout=3) as client:
        async with live_server(harness.app) as base:
            context = client.stream(
                "POST", base + "/v1/chat/completions", json={"model": "remote/chat-id", "messages": [], "stream": True}
            )
            response = await context.__aenter__()
            lines = response.aiter_lines()
            async for line in lines:
                if '"content":"h"' in line:
                    break
            await asyncio.sleep(0.02)
            assert not upstream.closed.is_set()
            assert (await harness.app.state.store.rows())[0]["status"] == "in_progress"
        await lines.aclose()
        await context.__aexit__(None, None, None)
    assert upstream.closed.is_set()
    store = Store(harness.config.database)
    await store.open()
    try:
        row = (await store.rows())[0]
        assert row["status"] == "cancelled" and row["context"][-1]["content"] == "h"
    finally:
        await store.close()


async def test_sdk_defaults_fixed_timeout_no_redirects_no_retries(tmp_path):
    provider = config_for(tmp_path).providers[0]
    adapter = OpenAIAdapter(provider)
    try:
        assert adapter.client.max_retries == 0
        assert adapter.client.timeout == 300.0
        assert adapter.client._client.follow_redirects is False
    finally:
        await adapter.close()


def test_role_specific_unknown_fields_remain_significant():
    user = {"role": "user", "content": "hi"}
    assert shared_prefix([user], [dict(user, refusal=None)], "openai-completions") == 0
    annotated_input = {"role": "user", "content": [{"type": "input_text", "text": "hi", "annotations": []}]}
    assert shared_prefix([user], [annotated_input], "openai-responses") == 0
