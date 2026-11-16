"""One SQLite row per branch, short serialized transactions, no request/event tables."""

import asyncio
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiosqlite

from .conversations import canonical, inherited_turns, meaningful, shared_prefix

SCHEMA = """
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES conversations(id), fork_index INTEGER,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, session_id TEXT,
    endpoint TEXT NOT NULL, protocol TEXT NOT NULL,
    provider TEXT NOT NULL, model TEXT NOT NULL,
    context_json TEXT NOT NULL, turns_json TEXT NOT NULL,
    context_complete INTEGER NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL
);
CREATE INDEX conversations_updated ON conversations(updated_at);
CREATE INDEX conversations_model ON conversations(model);
CREATE INDEX conversations_session ON conversations(session_id);
CREATE INDEX conversations_parent ON conversations(parent_id);
PRAGMA user_version = 1;
"""


def now() -> str:
    return datetime.now(UTC).isoformat()


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def decode(row) -> dict:
    row = dict(row)
    row["context"] = json.loads(row.pop("context_json"))
    row["turns"] = json.loads(row.pop("turns_json"))
    return row


@dataclass(frozen=True)
class Reservation:
    conversation_id: str
    request_id: str
    revision: int
    started: float


class ConversationError(ValueError):
    pass


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.lock = asyncio.Lock()
        self.db = None

    async def open(self):
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)
            os.chmod(self.path, 0o600)
            self.db = await aiosqlite.connect(self.path, isolation_level=None)
            self.db.row_factory = aiosqlite.Row
            await self.db.execute("PRAGMA foreign_keys = ON")
            await self.db.execute("PRAGMA busy_timeout = 5000")
            await self.db.execute("PRAGMA journal_mode = WAL")
            version = (await (await self.db.execute("PRAGMA user_version")).fetchone())[0]
            if version == 0:
                await self.db.executescript("BEGIN IMMEDIATE;\n" + SCHEMA + "\nCOMMIT;")
            elif version != 1:
                raise RuntimeError("Unsupported database schema version")
            # WAL/SHM use the database's restrictive permissions, including when reopened.
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(self.path) + suffix)
                if sidecar.exists():
                    sidecar.chmod(0o600)
            async with self.lock:
                try:
                    await self.db.execute("BEGIN IMMEDIATE")
                    for row in await self._rows():
                        if row["status"] == "in_progress":
                            for turn in row["turns"]:
                                if turn["status"] == "in_progress":
                                    ended = now()
                                    turn.update(
                                        status="interrupted",
                                        finished_at=ended,
                                        partial_output=False,
                                        duration=max(
                                            0,
                                            (
                                                datetime.fromisoformat(ended)
                                                - datetime.fromisoformat(turn["started_at"])
                                            ).total_seconds(),
                                        ),
                                    )
                            row.update(status="interrupted", updated_at=now(), revision=row["revision"] + 1)
                            await self._update(row)
                    await self.db.commit()
                except BaseException:
                    await self.db.rollback()
                    raise
        except BaseException:
            await self.close()
            raise

    async def close(self):
        if self.db:
            await self.db.close()
            self.db = None

    async def health(self):
        async with self.lock:
            await (await self.db.execute("SELECT count(*) FROM conversations")).fetchone()

    async def _rows(self):
        return [decode(r) for r in await (await self.db.execute("SELECT * FROM conversations")).fetchall()]

    async def rows(self):
        async with self.lock:
            return await self._rows()

    async def _insert(self, row):
        columns = [
            "id",
            "parent_id",
            "fork_index",
            "created_at",
            "updated_at",
            "session_id",
            "endpoint",
            "protocol",
            "provider",
            "model",
            "context_json",
            "turns_json",
            "context_complete",
            "status",
            "revision",
        ]
        values = dict(row, context_json=encode(row["context"]), turns_json=encode(row["turns"]))
        await self.db.execute(
            f"INSERT INTO conversations ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            [values[k] for k in columns],
        )

    async def _update(self, row, expected=None):
        cursor = await self.db.execute(
            "UPDATE conversations SET updated_at=?,provider=?,model=?,context_json=?,turns_json=?,"
            "context_complete=?,status=?,revision=? WHERE id=? AND revision=?",
            (
                row["updated_at"],
                row["provider"],
                row["model"],
                encode(row["context"]),
                encode(row["turns"]),
                row["context_complete"],
                row["status"],
                row["revision"],
                row["id"],
                row["revision"] - 1 if expected is None else expected,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Stale conversation reservation")

    @staticmethod
    def _scope(row, session, run):
        if row["session_id"] != session:
            return False
        # When sessions are supplied they take precedence over agent-run labels.
        return session is not None or (row["turns"][-1].get("run_id") if row["turns"] else None) == run

    async def reserve(
        self,
        *,
        request_id,
        transcript,
        endpoint,
        protocol,
        provider,
        model,
        options,
        proxy_model=None,
        conversation_id=None,
        session_id=None,
        run_id=None,
        proxy_options=None,
    ) -> Reservation:
        started = time.monotonic()
        async with self.lock:
            try:
                await self.db.execute("BEGIN IMMEDIATE")
                rows = [r for r in await self._rows() if r["protocol"] == protocol and r["endpoint"] == endpoint]
                explicit = next((r for r in rows if r["id"] == conversation_id), None)
                if conversation_id is not None:
                    if explicit is None:
                        raise ConversationError("Unknown conversation ID for this endpoint")
                    if session_id is not None and session_id != explicit["session_id"]:
                        raise ConversationError("Session ID does not match the selected conversation")
                    session_id = explicit["session_id"]
                pool = [r for r in rows if self._scope(r, session_id, run_id)]
                if explicit and explicit not in pool:
                    pool.append(explicit)
                transcript = deepcopy(transcript)
                responses = protocol == "openai-responses"
                complete = not responses or not bool(options.get("conversation") or options.get("previous_response_id"))
                reference_parent = None
                previous = options.get("previous_response_id") if responses else None
                if previous:
                    references = []
                    for row in pool:
                        for turn in row["turns"]:
                            if turn.get("response_id") == previous and turn["status"] != "in_progress":
                                references.append((row, turn["output_end"]))
                    prefixes = {
                        tuple(canonical(item, protocol) for item in row["context"][:boundary])
                        for row, boundary in references
                    }
                    if len(prefixes) == 1:
                        # Duplicated ancestral metadata has the same prefix; prefer the explicit branch.
                        references.sort(key=lambda pair: pair[0]["id"] != conversation_id)
                        reference_parent, boundary = references[0]
                        transcript = deepcopy(reference_parent["context"][:boundary]) + transcript
                        complete = bool(reference_parent["context_complete"]) and not options.get("conversation")
                resolved_references = []
                if protocol == "openai-responses":
                    for index, item in enumerate(transcript):
                        if item.get("type") != "item_reference":
                            continue
                        if set(item) != {"type", "id"}:
                            complete = False
                            continue
                        matches = {
                            canonical(known, protocol): known
                            for row in pool
                            for known in row["context"]
                            if known.get("id") == item.get("id") and known.get("type") != "item_reference"
                        }
                        if len(matches) == 1:
                            resolved_references.append({"input_index": index, "reference": deepcopy(item)})
                            transcript[index] = deepcopy(next(iter(matches.values())))
                        else:
                            complete = False
                parent = explicit or reference_parent
                shared = shared_prefix(parent["context"], transcript, protocol) if parent else 0
                if not parent and complete:
                    candidates = []
                    for row in pool:
                        if not row["context_complete"]:
                            continue
                        n = shared_prefix(row["context"], transcript, protocol)
                        if meaningful(transcript[:n]):
                            candidates.append((n, row))
                    if candidates:
                        longest = max(n for n, _ in candidates)
                        winners = [row for n, row in candidates if n == longest]
                        if len(winners) == 1:
                            parent, shared = winners[0], longest
                instructions_match = (
                    protocol != "openai-responses"
                    or not parent
                    or parent["turns"][-1]["options"].get("instructions") == options.get("instructions")
                )
                reuse = (
                    parent is not None
                    and parent["status"] != "in_progress"
                    and shared == len(parent["context"])
                    and instructions_match
                    and complete
                )
                timestamp = now()
                if reuse:
                    row = deepcopy(parent)
                    # Retain the recorded representation of the already-known prefix.
                    row["context"].extend(transcript[shared:])
                    row["revision"] += 1
                else:
                    row = dict(
                        id=str(uuid4()),
                        parent_id=parent["id"] if parent else None,
                        fork_index=shared if parent else None,
                        created_at=timestamp,
                        updated_at=timestamp,
                        session_id=session_id,
                        endpoint=endpoint,
                        protocol=protocol,
                        provider=provider,
                        model=model,
                        context=transcript,
                        turns=inherited_turns(parent["turns"], shared) if parent else [],
                        context_complete=int(complete and (not parent or parent["context_complete"])),
                        status="in_progress",
                        revision=1,
                    )
                turn = dict(
                    request_id=request_id,
                    run_id=run_id,
                    input_start=shared if parent else 0,
                    input_end=len(row["context"]),
                    output_start=len(row["context"]),
                    output_end=len(row["context"]),
                    started_at=timestamp,
                    finished_at=None,
                    duration=None,
                    provider=provider,
                    model=model,
                    proxy_model=proxy_model,
                    returned_model=None,
                    response_id=None,
                    upstream_request_id=None,
                    status="in_progress",
                    usage=None,
                    error=None,
                    partial_output=False,
                    options=deepcopy(options),
                    proxy_options=deepcopy(proxy_options or {}),
                )
                if resolved_references:
                    turn["proxy_options"]["resolved_item_references"] = resolved_references
                row["turns"].append(turn)
                row.update(updated_at=timestamp, provider=provider, model=model, status="in_progress")
                if reuse:
                    await self._update(row)
                else:
                    await self._insert(row)
                await self.db.commit()
                return Reservation(row["id"], request_id, row["revision"], started)
            except BaseException:
                await self.db.rollback()
                raise

    async def finalize(
        self,
        reservation: Reservation,
        outputs: list,
        *,
        status: str,
        returned_model=None,
        response_id=None,
        upstream_request_id=None,
        usage=None,
        error=None,
    ):
        async with self.lock:
            try:
                await self.db.execute("BEGIN IMMEDIATE")
                raw = await (
                    await self.db.execute("SELECT * FROM conversations WHERE id=?", (reservation.conversation_id,))
                ).fetchone()
                base = decode(raw)
                if (
                    base["revision"] != reservation.revision
                    or base["turns"][-1]["request_id"] != reservation.request_id
                    or base["status"] != "in_progress"
                ):
                    raise RuntimeError("Stale conversation reservation")
                outputs = sorted(outputs, key=lambda value: value[0]) or [(0, [], {})]
                for position, (choice_index, items, details) in enumerate(outputs):
                    row = deepcopy(base)
                    boundary = len(row["context"])
                    if position:
                        row.update(
                            id=str(uuid4()), parent_id=base["id"], fork_index=boundary, created_at=now(), revision=0
                        )
                    row["context"].extend(deepcopy(items))
                    turn_status = details.get("status", status) if status == "completed" else status
                    row.update(updated_at=now(), status=turn_status, revision=row["revision"] + 1)
                    row["turns"][-1].update(
                        output_end=len(row["context"]),
                        finished_at=row["updated_at"],
                        duration=time.monotonic() - reservation.started,
                        status=turn_status,
                        returned_model=returned_model,
                        response_id=response_id,
                        upstream_request_id=upstream_request_id,
                        usage=deepcopy(usage),
                        error=deepcopy(error or details.get("error")),
                        partial_output=turn_status != "completed" and bool(items),
                        choice_index=choice_index,
                        **{k: deepcopy(v) for k, v in details.items() if k not in {"status", "error"}},
                    )
                    if position:
                        await self._insert(row)
                    else:
                        await self._update(row, reservation.revision)
                await self.db.commit()
            except BaseException:
                await self.db.rollback()
                raise
