import asyncio
import json
from uuid import uuid4

import pytest

from glorp.storage import ConversationError, Store

U = {"role": "user", "content": "hello"}
A = {"role": "assistant", "content": "hi"}


async def reserve(store, history=None, **kwargs):
    values = dict(
        request_id=str(uuid4()),
        transcript=[U] if history is None else history,
        endpoint="/v1/chat/completions",
        protocol="openai-completions",
        provider="p",
        model="actual-model",
        options={},
    )
    values.update(kwargs)
    return await store.reserve(**values)


async def finish(store, reservation, text="hi", **kwargs):
    await store.finalize(reservation, [(0, [{"role": "assistant", "content": text}], {})], status="completed", **kwargs)


@pytest.fixture
async def store(tmp_path):
    store = Store(tmp_path / "data/transcripts.sqlite3")
    await store.open()
    yield store
    await store.close()


async def test_continuations_forks_and_model_switch(store):
    first = await reserve(store)
    assert (await store.rows())[0]["context"] == [U]
    await finish(store, first, returned_model="versioned")
    second = await reserve(store, [U, A, {"role": "user", "content": "wife?"}], model="other")
    assert second.conversation_id == first.conversation_id
    await finish(store, second, "Michelle")
    original = (await store.rows())[0]
    assert len(original["context"]) == 4
    assert [t["model"] for t in original["turns"]] == ["actual-model", "other"]
    fork = await reserve(store, [U, A, {"role": "user", "content": "birthplace?"}])
    await finish(store, fork, "Honolulu")
    rows = await store.rows()
    assert len(rows) == 2
    assert rows[0] == original
    assert rows[1]["parent_id"] == first.conversation_id
    assert rows[1]["fork_index"] == 2
    assert rows[1]["context"][:2] == [U, A]
    assert rows[1]["turns"][0] == original["turns"][0]
    assert rows[1]["turns"][-1]["input_start"] == 2


async def test_regeneration_and_explicit_divergence(store):
    first = await reserve(store)
    await finish(store, first)
    fork = await reserve(store, conversation_id=first.conversation_id)
    assert fork.conversation_id != first.conversation_id
    await finish(store, fork, "alternate")
    edited = await reserve(store, [{"role": "user", "content": "edited"}], conversation_id=first.conversation_id)
    row = next(r for r in await store.rows() if r["id"] == edited.conversation_id)
    assert row["parent_id"] == first.conversation_id and row["fork_index"] == 0
    assert row["turns"][0]["model"] == "actual-model"
    assert len(row["turns"]) == 1  # No invented provenance from the abandoned branch.


async def test_system_only_not_meaningful_and_ambiguous_matches(store):
    system = {"role": "system", "content": "same system"}
    first = await reserve(store, [system, U])
    await finish(store, first)
    second = await reserve(store, [system, {"role": "user", "content": "unrelated"}])
    assert (await store.rows())[1]["parent_id"] is None
    await finish(store, second)
    third = await reserve(store, [system, U], session_id="independent")
    await finish(store, third)
    # Force an identical branch in the unscoped pool, then avoid guessing between them.
    duplicate = await reserve(store, [system, U], conversation_id=first.conversation_id)
    await finish(store, duplicate)
    ambiguous = await reserve(store, [system, U, A, {"role": "user", "content": "next"}])
    row = next(r for r in await store.rows() if r["id"] == ambiguous.conversation_id)
    assert row["parent_id"] is None
    precise = await reserve(store, [system, U, A], conversation_id=first.conversation_id)
    assert precise.conversation_id == first.conversation_id


async def test_run_and_session_scope_and_unknown_id(store):
    one = await reserve(store, run_id="run-a")
    await finish(store, one)
    two = await reserve(store, [U, A], run_id="run-b")
    assert (await store.rows())[1]["parent_id"] is None
    await finish(store, two)
    with pytest.raises(ConversationError):
        await reserve(store, conversation_id="unknown")
    with pytest.raises(ConversationError):
        await reserve(store, conversation_id=one.conversation_id, session_id="wrong")
    assert len(await store.rows()) == 2


async def test_concurrent_reservations_stale_writes_and_alternative_choices(store):
    one, two = await asyncio.gather(reserve(store), reserve(store))
    assert one.conversation_id != two.conversation_id
    assert (await store.rows())[1]["parent_id"] == one.conversation_id
    await store.finalize(one, [(1, [{"role": "assistant", "content": "second"}], {}), (0, [A], {})], status="completed")
    await finish(store, two, "concurrent")
    rows = await store.rows()
    assert len(rows) == 3
    assert all(len(r["context"]) == 2 for r in rows)
    assert rows[0]["context"][-1] == A
    with pytest.raises(RuntimeError, match="Stale"):
        await finish(store, one)


async def test_failed_retry_and_restart(tmp_path):
    path = tmp_path / "db.sqlite3"
    store = Store(path)
    await store.open()
    first = await reserve(store)
    await store.finalize(first, [], status="failed", error={"error": {"message": "429"}})
    retry = await reserve(store)
    assert retry.conversation_id == first.conversation_id
    assert len((await store.rows())[0]["context"]) == 1
    await store.close()
    await store.open()
    row = (await store.rows())[0]
    assert row["context"] == [U] and row["status"] == "interrupted"
    assert row["turns"][-1]["status"] == "interrupted"
    tables = await (await store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
    assert [r[0] for r in tables] == ["conversations"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert "messages" not in json.dumps(row["turns"])
    await store.close()


async def test_previous_responses_and_instructions(store):
    kwargs = dict(protocol="openai-responses", endpoint="/v1/responses")
    one = await reserve(store, **kwargs, options={"instructions": "first"})
    await finish(store, one, response_id="r1")
    two = await reserve(
        store,
        [{"role": "user", "content": "next"}],
        **kwargs,
        options={"previous_response_id": "r1", "instructions": "first"},
    )
    assert one.conversation_id == two.conversation_id
    await finish(store, two, "next reply", response_id="r2")
    fork = await reserve(
        store, [{"role": "user", "content": "different"}], **kwargs, options={"previous_response_id": "r1"}
    )
    row = next(r for r in await store.rows() if r["id"] == fork.conversation_id)
    assert row["parent_id"] == one.conversation_id and row["fork_index"] == 2
    assert len(row["context"]) == 3
    assert row["context_complete"] == 1
    await finish(store, fork)
    changed = await reserve(
        store, [U, A], **kwargs, conversation_id=one.conversation_id, options={"instructions": "changed"}
    )
    assert changed.conversation_id != one.conversation_id
    unknown = await reserve(store, **kwargs, options={"previous_response_id": "unseen"})
    await finish(store, unknown)
    descendant = await reserve(store, [U, A], **kwargs, conversation_id=unknown.conversation_id)
    row = next(r for r in await store.rows() if r["id"] == descendant.conversation_id)
    assert row["context_complete"] == 0
    unrelated = await reserve(store, [U, A], **kwargs, options={"conversation": "server-state"})
    row = next(r for r in await store.rows() if r["id"] == unrelated.conversation_id)
    assert row["context_complete"] == 0 and row["parent_id"] is None
