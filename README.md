# Glorp

A small personal OpenAI-compatible proxy that routes **provider/model** IDs and
records whole conversation branches in local SQLite. Python **3.14+**, `uv`, one worker.
No arbitrary model aliases, protocol translation, retries, inference hosting, or training pipeline.

## Run

```sh
cp config.example.yaml config.yaml
# Edit URLs, model IDs, protocols, and credentials to match your providers.
export CUSTOM_PROVIDER_API_KEY='...'
uv sync
uv run glorp --config config.yaml
```

The default configuration path is `config.yaml`. The default address is
`http://127.0.0.1:8000/v1`. Database paths are relative to the **configuration
file**, not your current directory. Schema creation and recovery happen before
traffic is accepted. Configuration is fixed until restart.

Each `provider_list` entry supplies a unique name (without `/`), fixed base URL,
protocol, API key, and models with `id` and display-only `name` fields. YAML model
IDs must match upstream IDs exactly and be unique **within that provider**;
different providers can expose the same upstream ID.

Use **`<provider name>/<model id>`** in requests to Glorp. For example, provider
`custom_provider` with YAML model `id: gpt-6-luna` is requested as
`custom_provider/gpt-6-luna`. Glorp sends only `gpt-6-luna` upstream. If the upstream
ID itself contains slashes, keep them: `custom_provider/org/model` forwards
`org/model`. Do not add the provider prefix to YAML model IDs.

Bare IDs such as `gpt-6-luna` are rejected, even with only one provider. Upstream
response model IDs (including versioned names) are returned unchanged.

- `openai-completions`: only `POST /v1/chat/completions`.
- `openai-responses`: only `POST /v1/responses`.
- Multiple connections use the same official async OpenAI SDK adapter. A new
  compatible endpoint needs only another configuration entry; a native vendor
  protocol requires an adapter and registry support, not just a URL change.
- Keys can be literals (`API_KEY` means that literal string), or whole-value
  `${ENV_VAR}` references. Missing/empty referenced variables fail startup.
  Use a dummy key for unauthenticated local endpoints.
- Scheme-less loopback URLs such as `localhost:11434/v1` gain `http://`.
  Remote URLs require explicit HTTPS. Base URLs cannot come from requests.
- The SDK has a fixed 300-second timeout and `max_retries=0`: one request means
  at most one upstream attempt. Redirects are not followed. There is no fallback
  or load balancing.

Active configuration, database files/sidecars, `.env`, and logs are Git-ignored.
`config.example.yaml` contains no real credentials; its model IDs are illustrative.

## Connect a harness

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-unused",  # Or your inbound proxy token, if configured.
    max_retries=0,  # Avoid harness-side retries too, if desired.
)
result = client.chat.completions.create(
    model="custom_provider/gpt-6-luna",  # Provider name + configured upstream ID.
    messages=[{"role": "user", "content": "Explain this repository."}],
    extra_headers={"X-Glorp-Session-Id": "example-session"},
)
print(result.choices[0].message.content)
```

Use `stream=True` on the same endpoint. For a model configured with
`openai-responses`, use `client.responses.create(model=..., input=...)` instead.
Tools, structured outputs, reasoning controls, multimodal references, and
additional JSON fields pass through without dialect translation. The proxy never
executes tools or fetches files/images. SDK objects are serialized without
inventing absent optional fields or removing unknown response fields.

`GET /v1/models` lists the provider-qualified IDs to use, provider ownership, and display names.
`GET /health` checks the service and SQLite connection. Neither calls upstream.

### Conversation identity

Send full accumulated history, including assistant messages and tool results,
on subsequent requests. An ordinary continuation extends the same row. An edited
history, earlier-prompt regeneration, concurrent request, or divergence creates a
self-contained fork. Changing models alone does not split a branch. Multiple Chat
Completions choices become separate branches, not sequential assistant messages.

Optional headers (local labels, never forwarded):

| Header | Meaning |
| --- | --- |
| `X-Glorp-Conversation-Id` | An ID previously returned by Glorp; selects the continuation or fork parent |
| `X-Glorp-Session-Id` | Separates otherwise identical, independent chats |
| `X-Glorp-Run-Id` | Agent-run label; also scopes matching when no session is supplied |

Accepted calls return `X-Glorp-Request-Id` and `X-Glorp-Conversation-Id`, including
streams and upstream errors. The latter identifies the primary choice's branch.
Unknown explicit conversation IDs fail with 400. A supplied session must match
the selected conversation.

To retrieve headers with the SDK, use its raw response interface:

```python
raw = client.chat.completions.with_raw_response.create(
    model="custom_provider/gpt-6-luna",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={"X-Glorp-Session-Id": "another-chat"},
)
conversation_id = raw.headers["X-Glorp-Conversation-Id"]
result = raw.parse()
# Round-trip conversation_id in the next request's extra_headers.
```

Without explicit IDs, Glorp selects only a unique longest meaningful structural
prefix in the same dialect and matching scope. Shared system/developer prompts
alone do not link chats; ambiguous matches create roots. Identical histories in
independent chats are inherently indistinguishable: use distinct session IDs.

Matching ignores JSON key order and narrowly documented equivalents: plain text
versus text-only blocks; known absent/null optional Chat message fields; and
Responses message `type`/output `status` transport metadata. IDs, argument strings,
annotations with content, and unknown fields remain significant. Stored originals
are not normalized. Responses string input is stored as a user message with that
string as `content`.

Locally known `previous_response_id` values reconstruct history through that
turn's output, even when resuming an older turn. Known Responses item references
can be resolved from locally observed items (the original reference is recorded
in turn metadata). Unseen/ambiguous references and server-side conversations
remain explicitly incomplete; visible suffixes do not automatically merge them.
Omitted Responses instructions are **not** inherited from a previous response.

## Authentication and privacy

By default Glorp binds only to loopback. To enable inbound bearer authentication:

```yaml
server:
  host: 127.0.0.1
  port: 8000
  database: ./data/glorp.sqlite3
  api_key_env: GLORP_API_KEY
```

Set `GLORP_API_KEY` before startup. All routes then require `Authorization: Bearer
<token>`, checked with constant-time comparison. This token is separate from
upstream credentials and never forwarded. Non-loopback binding requires a token;
remote use **also requires TLS or a trusted tunnel**, which Glorp does not provide.
Do not expose the unauthenticated loopback endpoint through an untrusted proxy.

SQLite stores **sensitive plaintext prompts, outputs, options, and tool data**.
Database/sidecar permissions are restricted to the current user, but this is not
encryption. Keep the containing directory private, protect backups, and avoid
putting credentials in prompts, request metadata, or tool results. Provider keys
and full configuration are not recorded; upstream errors are sanitized. No public
log-reading API or automatic retention/deletion exists.

## Storage and failure semantics

The only application table is `conversations`: one full native transcript and an
ordered array of turn metadata per branch, with parent/fork boundaries, model
IDs, options, usage (null if absent), response/request IDs, status, timing,
completeness, and revision. The row/turn `model` remains the exact upstream request
ID; each new turn also records the namespaced `proxy_model` and exact
`returned_model`. Request options retain the original proxy-facing `model`.
Existing databases remain usable without migration; old turns are not rewritten.
Each fork includes applicable ancestor metadata. Provenance is not invented for
harness-supplied messages. No per-request history snapshots, raw-response archives,
or stream-event rows.

SQLite uses WAL, foreign keys, a busy timeout, and short serialized transactions.
Input is reserved **before** an upstream attempt. Output is assembled in memory
and saved once; streaming deltas are forwarded incrementally but never logged.
Chat streaming defaults `stream_options.include_usage` to true unless explicitly
false, with this addition recorded in metadata. Responses usage comes from terminal
objects. Success terminators are withheld until the final write commits.

Validation errors use OpenAI-shaped JSON. Upstream HTTP statuses are preserved;
connection failures map to 502 and timeouts to 504. Failed initial saves return
503 without contacting upstream. A final write failure is surfaced as an error,
never silently served as an unlogged success; stderr receives a safe operational
error. After SSE headers, failures use error events (or Responses failure/incomplete
terminal events), not a changed HTTP status or normal success marker.

Disconnects/upstream failures close streams and save assembled partial output as
non-successful. Graceful shutdown attempts cleanup. A hard crash can lose in-memory
output; saved input remains and stale in-progress turns become `interrupted` on
restart. A disk failure may prevent recording the failure itself.

### Inspect and back up

With the SQLite CLI installed:

```sh
sqlite3 data/glorp.sqlite3 \
  'SELECT id,parent_id,model,status,context_complete,updated_at FROM conversations;'
sqlite3 data/glorp.sqlite3 \
  'SELECT json_pretty(context_json),json_pretty(turns_json) FROM conversations LIMIT 1;'
# SQLite's online backup includes committed WAL contents; protect this copy too.
sqlite3 data/glorp.sqlite3 ".backup '/private/path/glorp-backup.sqlite3'"
```

Do not copy only the main file while WAL writes are active. Use online backup or
stop the service before copying. Monitor disk growth manually (`du -h data/`).
Back up before manually deleting/editing data; foreign keys protect fork lineage.
Glorp does not select training examples or claim that a complete transcript is
suitable training data. Remote file/image contents, hidden reasoning, and harness
state are not automatically visible. Returned summaries/opaque items are retained.

## Development

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests use temporary databases and mocked **SDK HTTP transports**, never paid APIs,
credentials, or internet access. Real loopback Uvicorn tests cover incremental SSE,
commit-before-success, and downstream disconnects. The official client is also
smoke-tested against the public proxy surface.
