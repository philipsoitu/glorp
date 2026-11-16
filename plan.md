# Glorp: minimal personal LLM proxy

## Goal

Build a small, local Python service that a custom agent harness can use in place of OpenAI's API base URL. It routes requests through configured providers and records the contexts and outputs in SQLite for later analysis and fine-tuning dataset preparation.

Use `uv` for dependency management and execution, and the official `openai` Python library for upstream requests. Implement **only OpenAI-compatible protocols initially**, usable with OpenAI itself or compatible custom/local endpoints. Leave a small extension point for other protocols later.

Use actual model IDs throughout: the harness sends the upstream model ID, the proxy forwards it unchanged, and SQLite records it with the context. No model aliases, model tiers, or model-name rewriting.

This document is the implementation spec; do not build a training pipeline yet.

## Scope

### Initial release

- One OpenAI-compatible base URL: `http://127.0.0.1:8000/v1`.
- `POST /v1/chat/completions` and `POST /v1/responses`, including streaming.
- `GET /v1/models`, listing configured model IDs and display names without an upstream call.
- `GET /health`, checking that the service and SQLite connection are usable without calling upstream.
- YAML configuration with a `provider_list`, provider connection details, and each provider's available models.
- Multiple OpenAI-compatible connections are allowed, using the same SDK-backed adapter.
- One SQLite entry per conversation branch, containing the entire accumulated context and completed assistant outputs—not one entry per request or streaming chunk.
- Automatically append continuations to existing conversations; create a separate entry when history forks.
- Optional harness-supplied conversation, session, and run identifiers to make matching unambiguous.
- No UI, accounts, billing system, deployment platform, or external database.

### Explicit non-goals

- Implementing native Anthropic, Google, or Ollama APIs in this release. An Ollama or custom server exposing an OpenAI-compatible endpoint can already be configured.
- Translating between Chat Completions, Responses, or different vendors' protocols.
- Fine-tuning, dataset export, cross-conversation deduplication, evaluation, or automatic training-data selection.
- Hosting local inference, semantic caching, fallback routing, automatic retries, or load balancing.
- WebSockets, audio streaming, file uploads, batch APIs, image generation endpoints, or other OpenAI APIs.
- Managing upstream conversation state, executing tools, or fetching referenced files and remote images. Local transcript assembly and branch detection are in scope.

## Stack and existing project

Keep the existing `src/glorp` package, `glorp` console entry point, and `uv_build` build backend. The repository currently requires Python 3.14; retain this unless an actual dependency incompatibility requires changing it.

Runtime dependencies:

- `fastapi`: HTTP routes and request lifecycle.
- `uvicorn`: run the local server.
- `openai`: async upstream client and streaming APIs.
- `aiosqlite`: nonblocking access to the local SQLite database.
- `pyyaml`: load configuration with `yaml.safe_load`; do not allow arbitrary YAML object construction.

Development dependencies: `pytest`, `pytest-asyncio`, `httpx`, and `ruff`. Use standard-library `argparse` and `uuid`; do not introduce a configuration framework or ORM.

```sh
uv add fastapi uvicorn openai aiosqlite pyyaml
uv add --dev pytest pytest-asyncio httpx ruff
uv run glorp --config config.yaml
uv run pytest
uv run ruff check .
```

Commit `uv.lock`. Never commit local credentials, logs, or the active configuration.

## Configuration and routing

Ship `config.example.yaml`; load `config.yaml` by default, with `--config` to override it. Resolve relative database paths relative to the configuration file, not the shell's working directory.

```yaml
provider_list:
  - name: custom_provider_name
    base_url: https://ai.soitu.ca/v1
    protocol: openai-completions
    api_key: "${CUSTOM_PROVIDER_API_KEY}"
    models:
      - id: gpt-6-luna
        name: GPT 6 Luna
      - id: gpt-6.1-sol
        name: GPT 6.1 Sol

  - name: local_ollama
    base_url: localhost:11434/v1
    protocol: openai-completions
    api_key: local-unused
    models:
      - id: qwen3.8-27b
        name: Qwen 3.8 27B

# Optional; these are the defaults when omitted.
server:
  host: 127.0.0.1
  port: 8000
  database: ./data/glorp.sqlite3
  # Optional inbound bearer token, read from this environment variable:
  # api_key_env: GLORP_API_KEY
```

The model IDs above are illustrative. Configure IDs actually served by each upstream. `api_key` accepts a literal string, as in `api_key: API_KEY`, or a whole-value `${ENV_VAR}` reference. Resolve references at startup and fail if the variable is missing or empty. A literal such as `API_KEY` is not implicitly an environment variable name. Use a dummy key for a local endpoint that does not require authentication, since the SDK expects a key.

Rules:

- Each provider has `name`, `base_url`, `protocol`, `api_key`, and `models`. Provider names must be unique.
- Each model has `id` and `name`. `id` is the exact upstream identifier used in requests and logs; `name` is display-only metadata, never a routing key or substitute ID.
- Incoming `model` must match a configured model `id` exactly, such as `gpt-6-luna`. Route to the provider listing that ID and forward `model` unchanged. Unknown IDs return an OpenAI-shaped `400` error without an upstream request.
- Require model IDs to be unique across the entire configuration. Reject duplicates at startup rather than guessing a provider, introducing prefixes, or silently choosing the first match.
- Preserve upstream response model identifiers, including versioned names. Store the exact requested ID and, when supplied, the returned model ID in SQLite.
- `protocol: openai-completions` uses `client.chat.completions.create(...)` for `/v1/chat/completions`. Also support `openai-responses` through the same SDK-backed adapter for `/v1/responses`. A request to an endpoint inconsistent with the configured protocol returns `400` before contacting upstream; do not translate protocols or assume a compatible server supports both.
- `GET /v1/models` returns an OpenAI-shaped list using each configured `id`, with `owned_by` set to the provider name and `name` as extra display metadata. Never manufacture alternative model IDs.
- Validate configuration at startup: schema, supported protocols, unique names/IDs, required environment references, URLs, ports, and writable database location. Fail with a useful message before accepting traffic.
- Literal API keys may be kept in the user's untracked YAML file; environment references are recommended. Never persist keys or the full configuration in SQLite or print them in errors.
- A configured provider base URL must be fixed at startup, not accepted from a request. Normalize a scheme-less loopback URL such as `localhost:11434/v1` to `http://localhost:11434/v1`. Require explicit HTTPS for non-loopback URLs; do not infer insecure HTTP for remote hosts.
- If inbound authentication is configured, validate `Authorization: Bearer ...` with constant-time comparison. This credential is separate from upstream credentials and is never forwarded.
- Without inbound authentication, bind only to loopback. Refuse non-loopback binding unless a proxy token is configured; remote access still requires TLS or a trusted tunnel.
- No model rewriting, implicit default provider, arbitrary unconfigured-model passthrough, or runtime config reload. Use a fixed upstream timeout of 300 seconds initially.

Adding another OpenAI-compatible provider requires only another `provider_list` entry. Adding a different protocol later should require a new adapter and registry entry—not changes to context logging. Configuration alone does not make another vendor OpenAI-compatible.

## HTTP behavior

### Request path

1. Validate inbound authentication, if enabled.
2. Parse a JSON object, find its exact model ID in `provider_list`, check the endpoint matches the provider's protocol, and validate proxy-owned fields such as `stream`.
3. Assign a UUID request ID. Match or create a conversation branch, save its incoming context, and reserve its next turn in SQLite **before** contacting upstream. This updates one conversation row, not a separate request log.
4. Select the configured provider connection and forward using the corresponding `AsyncOpenAI` method, leaving `model` unchanged.
5. Assemble the assistant output, append it to that conversation, finalize turn metadata, and return it using the same API dialect.

Create one async SDK client per configured provider during application startup, reuse its connection pool, and close it on shutdown. Set SDK `max_retries=0` so one incoming request represents at most one upstream attempt. Do not silently retry failures that may already have incurred cost.

Keep request validation narrow: do not build a duplicate schema for every OpenAI parameter. Preserve messages/input, tools, tool choices, structured-output settings, reasoning controls, multimodal content, sampling options, and other request fields. Use SDK arguments for known parameters and `extra_body` for additional JSON fields; verify unknown fields are not silently dropped. Allow OpenAI to validate model-specific options.

Only proxy-owned metadata headers are consumed locally:

- `X-Glorp-Conversation-Id`: optional ID previously returned by the proxy; selects a conversation for continuation or as a fork's parent.
- `X-Glorp-Session-Id`: optional user-defined session identifier; separates otherwise identical histories from unrelated chats.
- `X-Glorp-Run-Id`: optional agent-run identifier.

These identifiers are untrusted labels, not authentication, and are not forwarded. Return `X-Glorp-Request-Id` and `X-Glorp-Conversation-Id` on accepted calls, including streamed responses and errors. If a request forks, return the new branch's conversation ID. Unknown explicit conversation IDs return `400`, not a guessed match. Do not forward inbound authorization, cookies, arbitrary headers, or hop-by-hop headers. Capture the upstream request ID when available.

### Non-streaming

Use `client.chat.completions.create(...)` or `client.responses.create(...)`. Serialize returned SDK objects as JSON without dropping unknown response fields or adding absent optional fields. Do not filter responses through a restrictive FastAPI response model.

Preserve tool-call IDs, JSON argument strings, output-item ordering, refusals, usage, and finish/status details. Extract the final assistant message or Responses output items and append them to the stored transcript before returning success. Store request options and turn metadata alongside the transcript without duplicating the entire input history. The proxy never executes tools.

### Streaming

Use the SDK's async streams and emit OpenAI-compatible server-sent events:

- Chat Completions: `data: <chunk-json>\n\n`, followed by `data: [DONE]\n\n` only on normal completion.
- Responses: preserve event type in `event: ...` and serialize each event in `data: ...`; use Responses terminal events rather than inventing a Chat Completions `[DONE]` marker.
- Preserve event order and flush incrementally. Never buffer the entire model response before sending it.
- Assemble output in memory while forwarding chunks immediately. **Do not persist chunks, deltas, or SSE events.** The stored result must have the same transcript shape as a non-streaming response.
- For Chat Completions, accumulate text, refusals, and tool-call deltas by choice/tool-call index, preserving call IDs and argument strings. For Responses, use terminal response objects for final output and accumulate deltas only as needed for partial-output recovery.
- For streaming Chat Completions, default `stream_options.include_usage` to `true` if omitted; honor an explicit `false`. Record this outbound addition in turn metadata. Responses usage comes from terminal events when supplied. Missing usage stays null, never zero.
- At completion, append the assembled output to the conversation in one transaction. Hold back the success terminator (`[DONE]` or Responses terminal event) until this write succeeds; earlier deltas are still forwarded incrementally.
- If the caller disconnects or upstream fails, close/cancel the upstream stream promptly and save any assembled partial output once, marked incomplete, in cancellation-safe cleanup. Do not represent it as a successful assistant completion.
- Once SSE headers are sent, an HTTP status cannot be changed. Record the failure and emit an error event if the client is still connected; never send a normal-success terminator for a failed stream.

There is deliberately no stream-event table or on-disk chunk log. A hard process crash may lose output accumulated since the turn started; its saved input remains, and the turn is marked interrupted on restart. Mark an upstream Responses failure/incomplete event as such even if its HTTP status was `200`.

### Errors and logging failures

- Return OpenAI-shaped JSON errors (`{"error": {...}}`) for local validation, routing, authentication, and pre-stream upstream failures.
- Preserve upstream HTTP error status and useful structured error details where safe. Map connection failures to `502` and timeouts to `504`. Never include secrets or stack traces in HTTP responses.
- Store upstream error bodies when available, but never authorization headers or arbitrary exception dumps that may contain credentials.
- If saving/reserving the incoming conversation context fails, return `503` without making an upstream request.
- If persistence fails after an upstream call has started, stop the call/stream where possible and report a failure; do not silently serve an unlogged successful response. Emit an obvious operational error to stderr. A disk failure can still prevent the failure itself from being persisted.
- Invalid JSON, failed authentication, and unknown-model requests need not be retained in the context database. Do not log their raw bodies or credentials to stderr.

## SQLite conversation storage

Store **one row per conversation branch**, updated as the conversation grows. Do not keep separate per-request rows, full-history snapshots for every turn, or streaming-event rows. A fork gets its own full, self-contained transcript, so reading an entry never requires rebuilding it from ancestors.

Use one local database, schema initialization at startup, foreign keys, WAL mode, and a busy timeout. Serialize matching/reservation and final writes through a small store-level async lock around short transactions; never hold a transaction open while waiting on upstream or a slow client. Run one Uvicorn worker. No ORM, external queue, or separate logging service.

### `conversations` table

| Column | Purpose |
| --- | --- |
| `id TEXT PRIMARY KEY` | Proxy-generated conversation/branch UUID |
| `parent_id TEXT` | Nullable foreign key to the conversation this branch forked from |
| `fork_index INTEGER` | Number of shared transcript items at the fork, nullable for roots |
| `created_at TEXT`, `updated_at TEXT` | UTC timestamps |
| `session_id TEXT` | Optional harness-supplied session label |
| `endpoint TEXT`, `protocol TEXT` | Native API dialect; do not merge different dialects |
| `provider TEXT`, `model TEXT` | Most recent provider and exact requested model ID; each turn also records its own |
| `context_json TEXT` | Entire ordered transcript, including prompts, tool calls/results, and assistant outputs |
| `turns_json TEXT` | Ordered per-turn metadata and request options, without repeated history or stream events |
| `context_complete INTEGER` | Whether all referenced prior context was observable locally |
| `status TEXT` | Latest turn: `in_progress`, `completed`, `incomplete`, `failed`, `cancelled`, or `interrupted` |
| `revision INTEGER` | Incremented for atomic updates and stale-write protection |

`context_json` contains native Chat Completions messages or native Responses input/output items, not a lossy universal message schema. A Responses string input has a defined equivalent user-message representation. Keep all content blocks, tool identifiers, argument strings, refusals, reasoning items, and opaque fields that are actually provided.

Each `turns_json` element records the request ID, run ID, input/output transcript boundaries, timestamps, duration, provider, exact requested/returned model IDs, upstream response/request IDs when available, status, usage, and sanitized error details. Preserve all non-history request fields here, including tools, instructions, generation options, state references, and proxy-added options. Mark partial outputs explicitly. This metadata is embedded in the conversation row; it is not a second copy of the context or a raw response/event archive.

Index updated time, model ID, session ID, and parent ID. Use SQLite `PRAGMA user_version` for minimal schema versioning. On startup, mark leftover in-progress turns/conversations as interrupted, retaining their saved input. Close streams and finalize assembled outputs on graceful shutdown where possible.

### Matching continuations and detecting forks

Use deterministic structural comparison, not an LLM, embeddings, or fuzzy text similarity:

1. Build a comparison view of the incoming transcript. JSON object key order is irrelevant. Normalize only documented equivalent representations, such as known absent/null optional message fields and plain text versus equivalent text-only content blocks. Ignore only known transport-only response fields when matching output echoed back as input; tool-call IDs and semantic item references must still match. Preserve original content in storage; normalization is for comparison only. Unknown differences are real differences, not guessed equivalences.
2. Match within the same API dialect and session. If no session label is provided, use the run label as a matching scope when present; otherwise use an unscoped pool. An explicit conversation ID selects the candidate/parent and resolves ambiguity, but never permits overwriting divergent history. Validate any supplied session label against that conversation.
3. If an idle conversation's **entire stored transcript is a prefix of the incoming transcript**, and its effective Responses instructions are unchanged where applicable, reuse its row and append only the new suffix. Keep the already recorded prefix, including assistant output; do not duplicate messages. Reserve the turn and save the extended input atomically before calling upstream. Changing the model/provider alone does not split a conversation; record the actual model/provider on each turn.
4. If the incoming history diverges from an existing conversation or resumes from an earlier point, create a new row containing the supplied full history. Set `parent_id` and `fork_index` to the shared prefix. Preserve the original branch unchanged. Regenerating a reply from an earlier prompt is a fork, not a reason to overwrite the old reply. Changed system/developer messages or Responses instructions likewise must not silently rewrite the original conversation.
5. If no candidate shares meaningful history, start a new root. Without an explicit conversation ID, shared system prompts alone are not evidence of a shared chat. Choose the unique longest qualifying prefix; if several candidates are equally plausible, start a separate root rather than attach to an arbitrary one. A session label scopes the search but does not override history checks.
6. Finalize output on the reserved branch only, checking its request ID/revision. A second concurrent request against the same branch creates a separate branch from its own input rather than interleaving two assistant replies. Failed attempts without output can reuse their existing input row on retry. Multiple Chat Completions choices are alternative branches, never consecutive assistant messages in one transcript; the response header identifies the primary choice's branch.

Store shared history again in a fork deliberately: the goal is one complete entry per branch, not a normalized conversation graph. Copy applicable ancestor turn metadata through the fork boundary as well, excluding abandoned output, so retained model provenance and request options remain self-contained. Do not invent provenance for messages only observed in harness-supplied history. No two branches may share mutable transcript data. Prefix lookup can scan candidate rows for this single-user MVP; do not add a vector database or prefix-index service.

Automatic matching cannot distinguish two independent chats with identical history. Use distinct session IDs for those chats, and round-trip `X-Glorp-Conversation-Id` when precise lineage matters. Conservative extra entries are better than merging unrelated contexts.

### Example

After the first call, conversation A contains:

```text
user: hello
assistant: hi, how can i help you
```

The next request includes that history plus the new question. Update **the same row A** to:

```text
user: hello
assistant: hi, how can i help you
user: who is barack obama's wife?
assistant: michelle obama
```

If a later request instead continues from the first exchange with a different question, create row B, leaving A unchanged:

```text
# B: parent_id = A, fork_index = 2
user: hello
assistant: hi, how can i help you
user: where was barack obama born?
assistant: Honolulu, Hawaii
```

Both rows contain their whole branch. Ordinary follow-up turns do not create extra rows.

### Responses references and limits

- Prefer harness-supplied full history. If `previous_response_id` refers to a response recorded locally, reconstruct the input prefix through that turn's saved output boundary and append the new input. A reference to an earlier turn can therefore create a fork even after its original branch has grown.
- If a reference or server-side conversation points to unseen context, save only what was observed with `context_complete = 0`. Do not invent earlier messages or merge incomplete contexts merely because their visible suffixes match. Preserve that incompleteness on descendants. Respect Responses instruction semantics rather than assuming omitted instructions are inherited.
- Tool results are visible only when the harness sends them in a later request. Local harness state, referenced file contents, remote images, and hidden reasoning are not automatically available. Preserve returned reasoning summaries or opaque fields, but do not try to recover private chain of thought.
- Usage and model IDs are per-turn facts, not properties of every message in the history. A conversation can contain outputs from different models. Do not treat missing usage as zero or automatically mark a complete transcript as good training data.
- SQLite contains sensitive plaintext context. Exclude database files and `-wal`/`-shm` sidecars from Git; use restrictive local file permissions and no public log-reading API. No automatic retention/deletion in the MVP: document manual backup and disk-growth monitoring.

## Suggested file layout

```text
config.example.yaml
plan.md
pyproject.toml
uv.lock
src/glorp/
  __init__.py          # Existing console entry point delegates to CLI.
  cli.py               # argparse and uvicorn startup.
  config.py            # Safe YAML parsing, validation, model-ID-to-provider lookup.
  app.py               # App factory, lifespan, auth, routes, SSE handling.
  storage.py           # Conversation rows, atomic reservations/finalization.
  conversations.py     # Transcript assembly, prefix matching, fork detection.
  providers/
    __init__.py        # Small adapter contract and provider registry.
    openai.py          # AsyncOpenAI integration for both API dialects.
tests/
  test_config.py
  test_proxy.py
  test_streaming.py
  test_storage.py
  test_conversations.py
```

Keep the adapter interface small: create a non-streaming result or async event stream for a supported endpoint, expose upstream metadata, and close resources. Streaming events are transport inputs to the in-memory assembler, never database records. Avoid a universal cross-provider message format until a second provider establishes concrete requirements.

## Implementation sequence

1. **Bootstrap:** add dependencies with `uv`, CLI, example configuration, Git ignores, and startup validation.
2. **Conversation storage:** initialize/version SQLite, implement full-transcript rows, deterministic continuation/fork matching, atomic turn reservations, and crash-recovery markers.
3. **OpenAI non-streaming:** app lifecycle, routing, adapter, both POST endpoints, transcript extraction, model listing, health, optional inbound auth, and error mapping.
4. **Streaming:** endpoint-specific SSE serialization, in-memory response assembly, one final transcript update, usage capture, upstream failures, downstream disconnects, and cancellation-safe cleanup.
5. **Tests and documentation:** complete mocked integration tests and a README showing setup, harness configuration, database inspection, and data/privacy limitations.

## Acceptance tests

All automated tests must run without real credentials, internet access, or paid API calls. Mock the SDK's HTTP transport against realistic JSON/SSE fixtures and use temporary SQLite files. Include incremental-stream and actual disconnect coverage, not only fully buffered test-client responses.

- Startup rejects malformed YAML/schema, unresolved credential references, unsupported protocols, duplicate provider names/model IDs, and unsafe unauthenticated binding.
- Optional inbound auth works and neither inbound nor upstream secrets appear in persisted records or error output, whether provider keys come from YAML literals or environment references.
- Exact model IDs route to the provider listing them, including across custom and local connections. Request bodies and SQLite records retain the original IDs without prefixes or rewriting; display names cannot be used as model IDs.
- Scheme-less loopback base URLs are normalized correctly. Remote HTTP URLs are rejected.
- Requests using an endpoint inconsistent with the configured protocol are rejected before any upstream call.
- Both POST endpoints preserve agent-oriented inputs and outputs: system/developer messages, tool definitions and results, parallel tool calls, structured output, reasoning options, multimodal references, and unknown JSON fields.
- The two-turn example above produces exactly one conversation row containing all four messages; repeated history is not duplicated.
- Further continuations extend that row. Forks, edited history, regeneration, and earlier-turn resumption create self-contained new rows with correct parent/fork boundaries; the old transcript is unchanged.
- Identical system prompts do not merge unrelated chats. Explicit conversation/session IDs resolve matching ambiguity, while ambiguous automatic matches do not select arbitrary parents.
- Known equivalent SDK/harness message representations match correctly; changed content, tool-call IDs, instructions, and unknown fields are not silently equated.
- Switching models within a conversation preserves one branch and records the exact model ID per turn.
- Non-streaming and streaming responses produce equivalent assembled transcripts, including text, refusals, tool-call arguments, and Responses output items. No chunk/event records or per-request context snapshots are persisted.
- Both streaming protocols still emit incremental ordered SSE; final success is signaled only after the assembled transcript is saved.
- Errors before streaming preserve appropriate HTTP statuses; midstream errors and disconnects save one marked partial output, retain the supplied context, and close upstream resources.
- Known `previous_response_id` references reconstruct the correct local prefix, including forks from earlier turns; unknown references remain explicitly incomplete.
- Multiple choices and simultaneous requests from a shared prefix become separate branches, never interleaved replies or lost updates. Retrying a failed turn with no output does not duplicate its history.
- Upstream `429`, upstream `5xx`, and timeouts cause exactly one attempt, not SDK retries.
- A failed initial context save prevents any upstream request. Later write failures are surfaced instead of silently losing output.
- Restarting marks stale in-progress turns interrupted while retaining saved context; no claim is made that in-memory deltas survive a hard crash.
- `GET /v1/models` returns configured model IDs with display names and provider ownership; health/model-list requests do not contact upstream.
- Smoke-test the actual public surface with the official `OpenAI` or `AsyncOpenAI` client pointed at the proxy.

## Done when

A user can run:

```sh
cp config.example.yaml config.yaml
# Edit provider URLs, protocols, model IDs, and credentials in config.yaml.
export CUSTOM_PROVIDER_API_KEY='...'
uv sync
uv run glorp --config config.yaml
```

Then configure their harness equivalently to:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-unused",  # Use the proxy token here if auth is enabled.
)
response = client.chat.completions.create(
    model="gpt-6-luna",
    messages=[{"role": "user", "content": "Explain this repository."}],
    extra_headers={"X-Glorp-Session-Id": "example-session"},
)
```

Streaming works through the same base URL. For a provider configured with `protocol: openai-responses`, use `client.responses.create(...)` with that provider's actual model ID. Calls route without changing model IDs. SQLite contains one full transcript per conversation branch, extended across turns, with per-turn model/provider/options metadata and no streaming-event records. Forks are separate entries with the shared history included.
