import json

import httpx2
import pytest
from conftest import CHAT, RESPONSE, Harness
from test_streaming import UpstreamStream, chat_events, response_events

from glorp.config import Config, Model, Provider


def namespaced_config(tmp_path):
    return Config(
        (
            Provider(
                "custom_provider",
                "https://custom.invalid/v1",
                "openai-completions",
                "key-a",
                (Model("gpt-6-luna", "GPT 6 Luna"), Model("org/model", "Org Model")),
            ),
            Provider(
                "other_provider",
                "http://localhost:11434/v1",
                "openai-completions",
                "key-b",
                (Model("gpt-6-luna", "Another GPT 6 Luna"),),
            ),
            Provider(
                "responses",
                "https://responses.invalid/v1",
                "openai-responses",
                "key-c",
                (Model("gpt-6-luna", "Responses GPT 6 Luna"),),
            ),
        ),
        tmp_path / "namespaced.sqlite3",
    )


async def test_model_list_uses_provider_namespaces(tmp_path):
    harness = Harness(namespaced_config(tmp_path))
    async with harness.client() as client:
        models = (await client.get("/v1/models")).json()["data"]
        assert [(m["id"], m["owned_by"]) for m in models] == [
            ("custom_provider/gpt-6-luna", "custom_provider"),
            ("custom_provider/org/model", "custom_provider"),
            ("other_provider/gpt-6-luna", "other_provider"),
            ("responses/gpt-6-luna", "responses"),
        ]
        assert models[0]["name"] == "GPT 6 Luna"
        assert not harness.calls


@pytest.mark.parametrize(
    "proxy_model,upstream_model,host,endpoint",
    [
        ("custom_provider/gpt-6-luna", "gpt-6-luna", "custom.invalid", "chat/completions"),
        ("other_provider/gpt-6-luna", "gpt-6-luna", "localhost", "chat/completions"),
        ("custom_provider/org/model", "org/model", "custom.invalid", "chat/completions"),
        ("responses/gpt-6-luna", "gpt-6-luna", "responses.invalid", "responses"),
    ],
)
async def test_namespaced_routing_preserves_upstream_and_proxy_ids(
    tmp_path, proxy_model, upstream_model, host, endpoint
):
    harness = Harness(namespaced_config(tmp_path))
    body = {
        "model": proxy_model,
        "vendor": {"model": "leave-this-unchanged"},
        **({"input": "hello"} if endpoint == "responses" else {"messages": [{"role": "user", "content": "hello"}]}),
    }
    async with harness.client() as client:
        response = await client.post("/v1/" + endpoint, json=body)
        expected = RESPONSE if endpoint == "responses" else CHAT
        assert response.status_code == 200 and response.json() == expected
        assert harness.calls[0].url.host == host
        assert harness.body() == dict(body, model=upstream_model)
        row = (await harness.app.state.store.rows())[0]
        turn = row["turns"][-1]
        assert row["model"] == turn["model"] == upstream_model
        assert row["provider"] == turn["provider"] == proxy_model.split("/", 1)[0]
        assert turn["proxy_model"] == turn["options"]["model"] == proxy_model
        assert turn["returned_model"] == expected["model"]


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6-luna",
        "custom_provider",
        "custom_provider/",
        "unknown/gpt-6-luna",
        "custom_provider/GPT 6 Luna",
        "custom_provider/GPT-6-LUNA",
        "org/model",
        "custom_provider/custom_provider/gpt-6-luna",
        "/custom_provider/gpt-6-luna",
    ],
)
async def test_bare_unknown_and_display_ids_do_not_route(tmp_path, model):
    harness = Harness(namespaced_config(tmp_path))
    async with harness.client() as client:
        response = await client.post("/v1/chat/completions", json={"model": model, "messages": []})
        assert response.status_code == 400 and response.json()["error"]["code"] == "model_not_found"
        assert not harness.calls and not await harness.app.state.store.rows()


@pytest.mark.parametrize(
    "model,endpoint",
    [
        ("custom_provider/gpt-6-luna", "responses"),
        ("responses/gpt-6-luna", "chat/completions"),
    ],
)
async def test_namespaced_id_still_requires_matching_protocol(tmp_path, model, endpoint):
    harness = Harness(namespaced_config(tmp_path))
    async with harness.client() as client:
        response = await client.post("/v1/" + endpoint, json={"model": model, "messages": [], "input": "hello"})
        assert response.status_code == 400 and not harness.calls


@pytest.mark.parametrize("responses", [False, True])
async def test_streams_strip_only_request_namespace(tmp_path, responses):
    upstream = UpstreamStream(response_events() if responses else chat_events())

    async def handler(request):
        return httpx2.Response(200, stream=upstream, headers={"content-type": "text/event-stream"})

    harness = Harness(namespaced_config(tmp_path), handler)
    model = "responses/gpt-6-luna" if responses else "custom_provider/org/model"
    async with harness.client() as client:
        response = await client.post(
            "/v1/responses" if responses else "/v1/chat/completions",
            json={"model": model, "stream": True, **({"input": "hello"} if responses else {"messages": []})},
        )
        assert response.status_code == 200
        assert harness.body()["model"] == ("gpt-6-luna" if responses else "org/model")
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
        if responses:
            assert events[-1]["response"]["model"] == RESPONSE["model"]
        else:
            assert all(event["model"] == CHAT["model"] for event in events)
        row = (await harness.app.state.store.rows())[0]
        assert row["status"] == "completed" and row["turns"][-1]["proxy_model"] == model
        assert row["model"] == row["turns"][-1]["model"] == harness.body()["model"]


async def test_switching_providers_with_same_upstream_id_continues_branch(tmp_path):
    harness = Harness(namespaced_config(tmp_path))
    async with harness.client() as client:
        history = [{"role": "user", "content": "hello"}]
        first = await client.post(
            "/v1/chat/completions", json={"model": "custom_provider/gpt-6-luna", "messages": history}
        )
        # Simulate a pre-namespace turn in the existing database: no metadata rewrite needed.
        store = harness.app.state.store
        row = (await store.rows())[0]
        row["turns"][0].pop("proxy_model")
        row["turns"][0]["options"]["model"] = "gpt-6-luna"
        await store.db.execute(
            "UPDATE conversations SET turns_json=? WHERE id=?", (json.dumps(row["turns"]), row["id"])
        )
        history += [CHAT["choices"][0]["message"], {"role": "user", "content": "next"}]
        second = await client.post(
            "/v1/chat/completions", json={"model": "other_provider/gpt-6-luna", "messages": history}
        )
        assert first.headers["x-glorp-conversation-id"] == second.headers["x-glorp-conversation-id"]
        rows = await store.rows()
        assert len(rows) == 1 and len(rows[0]["context"]) == 4
        turns = rows[0]["turns"]
        assert [turn["model"] for turn in turns] == ["gpt-6-luna", "gpt-6-luna"]
        assert [turn["provider"] for turn in turns] == ["custom_provider", "other_provider"]
        assert "proxy_model" not in turns[0] and turns[1]["proxy_model"] == "other_provider/gpt-6-luna"
