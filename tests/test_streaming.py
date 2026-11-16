import asyncio
import json
import sqlite3
from copy import deepcopy

import httpx
import httpx2
import pytest
from conftest import CHAT, RESPONSE, Harness, config_for, live_server

from glorp.conversations import comparison


def wire(value, event=None):
    return (
        (f"event: {event}\n" if event else "")
        + "data: "
        + (value if isinstance(value, str) else json.dumps(value))
        + "\n\n"
    ).encode()


def chat_events():
    def chunk(delta, finish=None):
        return {
            "id": "chat-one",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": CHAT["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    return [
        wire(chunk({"role": "assistant", "content": "h"})),
        wire(chunk({"content": "i"})),
        wire(chunk({}, "stop")),
        wire(
            {
                "id": "chat-one",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": CHAT["model"],
                "choices": [],
                "usage": CHAT["usage"],
            }
        ),
        wire("[DONE]"),
    ]


def response_events(status="completed"):
    partial = dict(deepcopy(RESPONSE), status="in_progress", output=[], usage=None)
    final = dict(deepcopy(RESPONSE), status=status)
    if status != "completed":
        final["error"] = {"code": "upstream_failure", "message": "failed"} if status == "failed" else None
        final["incomplete_details"] = {"reason": "max_output_tokens"}
    return [
        wire({"type": "response.created", "sequence_number": 0, "response": partial}, "response.created"),
        wire(
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {"type": "message", "id": "msg-1", "role": "assistant", "content": [], "status": "in_progress"},
            },
            "response.output_item.added",
        ),
        wire(
            {
                "type": "response.content_part.added",
                "sequence_number": 2,
                "output_index": 0,
                "item_id": "msg-1",
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            "response.content_part.added",
        ),
        wire(
            {
                "type": "response.output_text.delta",
                "sequence_number": 3,
                "output_index": 0,
                "item_id": "msg-1",
                "content_index": 0,
                "delta": "hi",
            },
            "response.output_text.delta",
        ),
        wire({"type": "response." + status, "sequence_number": 4, "response": final}, "response." + status),
    ]


class UpstreamStream(httpx2.AsyncByteStream):
    def __init__(self, events, gate=None, fail=False, pause_after=1):
        self.events, self.gate, self.fail, self.pause_after = events, gate, fail, pause_after
        self.closed = asyncio.Event()

    async def __aiter__(self):
        for i, event in enumerate(self.events):
            yield event
            if i + 1 == self.pause_after:
                if self.gate:
                    await self.gate.wait()
                if self.fail:
                    raise httpx2.ReadError("must not leak UPSTREAM_SECRET")

    async def aclose(self):
        self.closed.set()


def streaming_harness(tmp_path, upstream):
    async def handler(request):
        return httpx2.Response(
            200, stream=upstream, headers={"content-type": "text/event-stream", "x-request-id": "stream-request"}
        )

    return Harness(config_for(tmp_path), handler)


@pytest.mark.parametrize("responses", [False, True])
async def test_stream_equivalent_to_nonstream(tmp_path, responses):
    upstream = UpstreamStream(response_events() if responses else chat_events())
    harness = streaming_harness(tmp_path, upstream)
    body = {
        "model": "responses/responses-id" if responses else "remote/chat-id",
        "stream": True,
        **({"input": "hello"} if responses else {"messages": [{"role": "user", "content": "hello"}]}),
    }
    async with harness.client() as client:
        response = await client.post("/v1/responses" if responses else "/v1/chat/completions", json=body)
        assert response.status_code == 200
        assert upstream.closed.is_set()
        if responses:
            assert "event: response.output_text.delta" in response.text
            assert "event: response.completed" in response.text and "[DONE]" not in response.text
        else:
            assert response.text.endswith("data: [DONE]\n\n")
            assert harness.body()["stream_options"] == {"include_usage": True}
        row = (await harness.app.state.store.rows())[0]
        expected = RESPONSE["output"] if responses else [CHAT["choices"][0]["message"]]
        assert [comparison(m, row["protocol"]) for m in row["context"][1:]] == [
            comparison(m, row["protocol"]) for m in expected
        ]
        turn = row["turns"][-1]
        assert harness.body()["model"] == ("responses-id" if responses else "chat-id")
        assert turn["model"] == row["model"] == harness.body()["model"]
        assert turn["proxy_model"] == body["model"]
        assert turn["options"]["model"] == body["model"]
        assert turn["usage"] == (RESPONSE if responses else CHAT)["usage"]
        assert turn["upstream_request_id"] == "stream-request" and row["status"] == "completed"
        assert turn["proxy_options"] == ({} if responses else {"stream_options": {"include_usage": True}})
        assert "delta" not in json.dumps(row)


async def test_explicit_false_usage_and_missing_usage(tmp_path):
    upstream = UpstreamStream(chat_events()[:3] + [wire("[DONE]")])
    harness = streaming_harness(tmp_path, upstream)
    async with harness.client() as client:
        result = await client.post(
            "/v1/chat/completions",
            json={
                "model": "remote/chat-id",
                "messages": [],
                "stream": True,
                "stream_options": {"include_usage": False},
            },
        )
        assert result.status_code == 200 and "[DONE]" in result.text
        assert harness.body()["stream_options"]["include_usage"] is False
        row = (await harness.app.state.store.rows())[0]
        assert row["turns"][-1]["usage"] is None and row["turns"][-1]["proxy_options"] == {}


@pytest.mark.parametrize("responses", [False, True])
async def test_incremental_and_success_waits_for_commit(tmp_path, responses, monkeypatch):
    upstream_gate, disk_gate, disk_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    upstream = UpstreamStream(
        response_events() if responses else chat_events(), gate=upstream_gate, pause_after=4 if responses else 1
    )
    harness = streaming_harness(tmp_path, upstream)
    async with live_server(harness.app) as base:
        original = harness.app.state.store.finalize

        async def delayed(*args, **kwargs):
            disk_entered.set()
            await disk_gate.wait()
            await original(*args, **kwargs)

        monkeypatch.setattr(harness.app.state.store, "finalize", delayed)
        async with httpx.AsyncClient(base_url=base, timeout=3) as client:
            async with client.stream(
                "POST",
                "/v1/responses" if responses else "/v1/chat/completions",
                json={
                    "model": "responses/responses-id" if responses else "remote/chat-id",
                    "stream": True,
                    **({"input": "hello"} if responses else {"messages": [{"role": "user", "content": "hello"}]}),
                },
            ) as response:
                lines = response.aiter_lines()
                received = []
                async with asyncio.timeout(2):
                    while True:
                        line = await anext(lines)
                        received.append(line)
                        if '"delta":"hi"' in line if responses else '"content":"h"' in line:
                            break
                assert not upstream_gate.is_set()  # Deltas arrive before upstream finishes.
                assert (await harness.app.state.store.rows())[0]["status"] == "in_progress"
                upstream_gate.set()

                async def read_rest():
                    async for line in lines:
                        received.append(line)
                        if "[DONE]" in line or line == "event: response.completed":
                            assert (await harness.app.state.store.rows())[0]["status"] == "completed"

                reader = asyncio.create_task(read_rest())
                await asyncio.wait_for(disk_entered.wait(), 2)
                await asyncio.sleep(0.03)
                assert not reader.done()
                assert not any("[DONE]" in line or line == "event: response.completed" for line in received)
                disk_gate.set()
                await asyncio.wait_for(reader, 2)


@pytest.mark.parametrize("responses", [False, True])
async def test_real_downstream_disconnect_saves_partial_once(tmp_path, responses):
    gate = asyncio.Event()
    upstream = UpstreamStream(
        response_events() if responses else chat_events(), gate=gate, pause_after=4 if responses else 1
    )
    harness = streaming_harness(tmp_path, upstream)
    async with live_server(harness.app) as base:
        async with httpx.AsyncClient(base_url=base, timeout=3) as client:
            async with client.stream(
                "POST",
                "/v1/responses" if responses else "/v1/chat/completions",
                json={
                    "model": "responses/responses-id" if responses else "remote/chat-id",
                    "stream": True,
                    **({"input": "hello"} if responses else {"messages": [{"role": "user", "content": "hello"}]}),
                },
            ) as response:
                async for line in response.aiter_lines():
                    if ('"delta":"hi"' in line) if responses else ('"content":"h"' in line):
                        break
            await asyncio.wait_for(upstream.closed.wait(), 2)
            async with asyncio.timeout(2):
                while (row := (await harness.app.state.store.rows())[0])["status"] == "in_progress":
                    await asyncio.sleep(0.01)
            assert row["status"] == "cancelled" and row["turns"][-1]["partial_output"]
            assert len(row["context"]) == 2 and len(row["turns"]) == 1
            text = row["context"][-1]["content"]
            assert text[0]["text"] == "hi" if responses else text == "h"
            assert not gate.is_set()


@pytest.mark.parametrize("failure", ["transport", "truncated", "write"])
async def test_midstream_failures_no_success_marker(tmp_path, failure, monkeypatch, caplog):
    upstream = UpstreamStream(
        chat_events() if failure != "truncated" else chat_events()[:1], fail=failure == "transport"
    )
    harness = streaming_harness(tmp_path, upstream)
    async with harness.client() as client:
        if failure == "write":

            async def fail(*args, **kwargs):
                raise sqlite3.OperationalError("disk full")

            monkeypatch.setattr(harness.app.state.store, "finalize", fail)
        response = await client.post(
            "/v1/chat/completions", json={"model": "remote/chat-id", "messages": [], "stream": True}
        )
        assert response.status_code == 200
        assert "event: error" in response.text and "[DONE]" not in response.text
        assert upstream.closed.is_set() and "SECRET" not in response.text
        row = (await harness.app.state.store.rows())[0]
        if failure == "write":
            assert "persistence failed" in caplog.text
            assert row["status"] == "in_progress"  # Disk failure can also prevent saving the failure.
        else:
            assert row["status"] == ("failed" if failure == "transport" else "incomplete")
            assert row["context"][-1]["content"] == "h" and row["turns"][-1]["partial_output"]


@pytest.mark.parametrize("status", ["failed", "incomplete"])
async def test_responses_terminal_failure_http_200(tmp_path, status):
    upstream = UpstreamStream(response_events(status))
    harness = streaming_harness(tmp_path, upstream)
    async with harness.client() as client:
        response = await client.post(
            "/v1/responses", json={"model": "responses/responses-id", "input": "hi", "stream": True}
        )
        assert response.status_code == 200 and f"event: response.{status}" in response.text
        assert "event: response.completed" not in response.text and "[DONE]" not in response.text
        row = (await harness.app.state.store.rows())[0]
        assert row["status"] == status and row["turns"][-1]["partial_output"]


async def test_pre_stream_error_status(tmp_path):
    async def handler(request):
        return httpx2.Response(429, json={"error": {"message": "limited", "type": "rate_limit_error"}})

    harness = Harness(config_for(tmp_path), handler)
    async with harness.client() as client:
        result = await client.post(
            "/v1/chat/completions", json={"model": "remote/chat-id", "messages": [], "stream": True}
        )
        assert result.status_code == 429 and result.headers["content-type"] == "application/json"
        assert result.headers["x-glorp-conversation-id"] and len(harness.calls) == 1
