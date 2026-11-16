"""HTTP orchestration. Native payloads pass through; only transcripts reach SQLite."""

import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager
from copy import deepcopy
from uuid import uuid4

import anyio
from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, StreamingResponse

from .config import Config
from .conversations import ENDPOINTS, ChatAssembler, ResponsesAssembler, incoming, nonstream_output
from .errors import error_body, sanitize, stored_error, upstream_error
from .providers import create_adapter
from .providers.openai import serialize
from .storage import ConversationError, Store

log = logging.getLogger("glorp")


class PersistenceFailure(Exception):
    pass


class CallerDisconnected(Exception):
    pass


class StreamFailure(Exception):
    def __init__(self, body):
        self.body = body


def sse(data, event=None):
    if event is not None and (not isinstance(event, str) or "\n" in event or "\r" in event):
        raise ValueError("Invalid upstream event type")
    prefix = f"event: {event}\n" if event else ""
    payload = (
        data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    )
    return f"{prefix}data: {payload}\n\n".encode()


async def wait_disconnect(receive):
    while True:
        if (await receive())["type"] == "http.disconnect":
            return


class LiveStreamResponse(StreamingResponse):
    """Watch disconnects on every ASGI version, even while upstream is stalled."""

    def __init__(self, iterator, *, cleanup, active, headers):
        super().__init__(iterator, media_type="text/event-stream", headers=headers)
        self.cleanup, self.active = cleanup, active

    async def __call__(self, scope, receive, send):
        task = asyncio.current_task()
        self.active.add(task)
        try:
            async with anyio.create_task_group() as group:

                async def send_stream():
                    try:
                        await self.stream_response(send)
                    finally:
                        group.cancel_scope.cancel()

                async def disconnect():
                    await wait_disconnect(receive)
                    group.cancel_scope.cancel()

                group.start_soon(send_stream)
                group.start_soon(disconnect)
        finally:
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()
                await self.cleanup()
            self.active.discard(task)


async def call_until_disconnect(adapter, endpoint, body, request):
    call = asyncio.create_task(adapter.create(endpoint, body))
    disconnect = asyncio.create_task(wait_disconnect(request.receive))
    try:
        done, _ = await asyncio.wait({call, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        if call in done:
            return await call
        raise CallerDisconnected
    finally:
        disconnect.cancel()
        if not call.done():
            call.cancel()
        await asyncio.gather(call, disconnect, return_exceptions=True)


def create_app(config: Config, *, adapter_factory=create_adapter, store_factory=Store) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = store_factory(config.database)
        app.state.adapters = {}
        app.state.active = set()
        app.state.cleanups = set()
        try:
            await app.state.store.open()
            for provider in config.providers:
                app.state.adapters[provider.name] = adapter_factory(provider)
            yield
        finally:
            active = list(app.state.active)
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            if app.state.cleanups:
                await asyncio.gather(*list(app.state.cleanups), return_exceptions=True)
            for adapter in app.state.adapters.values():
                await adapter.close()
            await app.state.store.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def auth(request, call_next):
        if config.api_key:
            supplied = request.headers.get("authorization", "").encode()
            if not hmac.compare_digest(supplied, ("Bearer " + config.api_key).encode()):
                return JSONResponse(error_body("Invalid proxy API key", "authentication_error"), status_code=401)
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(error_body("Unknown route or unsupported HTTP method"), status_code=exc.status_code)

    @app.get("/health")
    async def health():
        try:
            await app.state.store.health()
        except Exception:
            return JSONResponse(error_body("SQLite is unavailable", "server_error"), status_code=503)
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {"id": m.id, "object": "model", "created": 0, "owned_by": p.name, "name": m.name}
                for p in config.providers
                for m in p.models
            ],
        }

    async def protected(coro):
        # Shield against both ASGI disconnect cancellation and server shutdown. The
        # lifespan also joins these tasks before closing SQLite.
        task = asyncio.create_task(coro)
        app.state.cleanups.add(task)
        task.add_done_callback(app.state.cleanups.discard)
        with anyio.CancelScope(shield=True):
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    if task.cancelled():
                        raise
                    # asyncio task cancellation (unlike an AnyIO cancel scope) is
                    # not stopped by CancelScope. Finish this short cleanup first.

    async def save(reservation, outputs, **metadata):
        try:
            await protected(app.state.store.finalize(reservation, outputs, **metadata))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.error(
                "Conversation persistence failed after upstream activity; response cannot be served as successful"
            )
            raise PersistenceFailure from None

    async def post(request: Request):
        try:
            # Reject nonstandard NaN/Infinity as well as non-object JSON.
            def invalid_constant(value):
                raise ValueError

            body = json.loads(await request.body(), parse_constant=invalid_constant)
            if not isinstance(body, dict):
                raise ValueError
        except ValueError, UnicodeError:
            return JSONResponse(error_body("Expected a JSON object"), status_code=400)
        model = body.get("model")
        provider = config.routes.get(model) if isinstance(model, str) else None
        if not provider:
            return JSONResponse(error_body("Unknown model ID", code="model_not_found"), status_code=400)
        endpoint = request.url.path
        if endpoint != ENDPOINTS[provider.protocol]:
            return JSONResponse(error_body("Endpoint does not match the model's configured protocol"), status_code=400)
        try:
            if "stream" in body and type(body["stream"]) is not bool:
                raise ValueError("stream must be a boolean")
            if "stream_options" in body:
                options = body["stream_options"]
                if options is not None and not isinstance(options, dict):
                    raise ValueError("stream_options must be an object or null")
                if (
                    provider.protocol == "openai-completions"
                    and isinstance(options, dict)
                    and "include_usage" in options
                    and type(options["include_usage"]) is not bool
                ):
                    raise ValueError("stream_options.include_usage must be a boolean")
            if provider.protocol == "openai-completions" and "messages" not in body:
                raise ValueError("messages is required")
            transcript = incoming(body, provider.protocol)
        except ValueError as exc:
            return JSONResponse(error_body(str(exc)), status_code=400)
        request_id = str(uuid4())
        headers = {"X-Glorp-Request-Id": request_id}
        outbound = deepcopy(body)
        additions = {}
        streaming = body.get("stream", False)
        if streaming and provider.protocol == "openai-completions":
            options = outbound.get("stream_options") or {}
            if "include_usage" not in options:
                options["include_usage"] = True
                additions = {"stream_options": {"include_usage": True}}
            outbound["stream_options"] = options
        history_key = "messages" if provider.protocol == "openai-completions" else "input"
        options = {k: v for k, v in body.items() if k != history_key}
        try:
            reservation = await app.state.store.reserve(
                request_id=request_id,
                transcript=transcript,
                endpoint=endpoint,
                protocol=provider.protocol,
                provider=provider.name,
                model=model,
                options=options,
                proxy_options=additions,
                conversation_id=request.headers.get("x-glorp-conversation-id"),
                session_id=request.headers.get("x-glorp-session-id"),
                run_id=request.headers.get("x-glorp-run-id"),
            )
        except ConversationError as exc:
            return JSONResponse(error_body(str(exc)), status_code=400, headers=headers)
        except Exception:
            log.error("Conversation reservation failed; no upstream request was made")
            return JSONResponse(
                error_body("Could not save incoming context", "server_error"), status_code=503, headers=headers
            )
        headers["X-Glorp-Conversation-Id"] = reservation.conversation_id
        adapter = app.state.adapters[provider.name]
        task = asyncio.current_task()
        app.state.active.add(task)
        try:
            result = await call_until_disconnect(adapter, endpoint, outbound, request)
        except CallerDisconnected:
            try:
                await save(reservation, [], status="cancelled")
            except PersistenceFailure:
                pass
            return JSONResponse(error_body("Caller disconnected", "cancelled_error"), status_code=499, headers=headers)
        except asyncio.CancelledError:
            await save(reservation, [], status="cancelled")
            raise
        except Exception as exc:
            status, error = upstream_error(exc, config.secrets)
            upstream_id = getattr(exc, "request_id", None)
            try:
                await save(
                    reservation,
                    [],
                    status="failed",
                    error=stored_error(exc, error, config.secrets),
                    upstream_request_id=upstream_id,
                )
            except PersistenceFailure:
                return JSONResponse(
                    error_body("Could not save failed turn", "server_error"), status_code=503, headers=headers
                )
            return JSONResponse(error, status_code=status, headers=headers)
        finally:
            app.state.active.discard(task)
        if streaming:
            return stream_response(result, reservation, provider.protocol, headers)
        try:
            data = serialize(result)
            for field in ("error", "incomplete_details"):
                if data.get(field) is not None:
                    data[field] = sanitize(data[field], config.secrets)
            response = JSONResponse(data, headers=headers)
            outputs = nonstream_output(data, provider.protocol)
            await save(
                reservation,
                outputs,
                status="completed",
                returned_model=data.get("model"),
                response_id=data.get("id"),
                upstream_request_id=getattr(result, "_request_id", None),
                usage=data.get("usage"),
            )
        except PersistenceFailure:
            return JSONResponse(
                error_body("Could not save upstream output", "server_error"), status_code=503, headers=headers
            )
        except Exception as exc:
            status, error = upstream_error(exc, config.secrets)
            try:
                await save(reservation, [], status="failed", error=error)
            except PersistenceFailure:
                status, error = 503, error_body("Could not save turn", "server_error")
            return JSONResponse(error, status_code=status, headers=headers)
        return response

    def stream_response(stream, reservation, protocol, headers):
        chat = protocol == "openai-completions"
        assembler = ChatAssembler() if chat else ResponsesAssembler()
        finalized, closed = False, False
        upstream_id = getattr(getattr(stream, "response", None), "headers", {}).get("x-request-id")

        async def close():
            nonlocal closed
            if not closed:
                closed = True
                try:
                    await protected(stream.close())
                except Exception:
                    log.error("Unable to close upstream stream cleanly")

        async def finish(status, error=None):
            nonlocal finalized
            if finalized:
                return
            finalized = True  # An unsuccessful disk write must not trigger a second write.
            await save(
                reservation,
                assembler.outputs(),
                status=status,
                returned_model=assembler.model,
                response_id=assembler.response_id,
                upstream_request_id=upstream_id,
                usage=assembler.usage,
                error=sanitize(error, config.secrets),
            )

        async def cleanup():
            await close()
            if not finalized:
                try:
                    await finish("cancelled")
                except PersistenceFailure:
                    pass  # Already emitted an operational error; disk may still be unavailable.

        async def events():
            try:
                async for sdk_event in stream:
                    event = serialize(sdk_event)
                    if isinstance(event.get("response"), dict):
                        for field in ("error", "incomplete_details"):
                            if event["response"].get(field) is not None:
                                event["response"][field] = sanitize(event["response"][field], config.secrets)
                    if "error" in event and chat:
                        raise StreamFailure(event)
                    assembler.feed(event)
                    if not chat:
                        if event.get("type") == "error":
                            raise StreamFailure(event)
                        if event.get("type") in ResponsesAssembler.TERMINAL:
                            break
                    yield sse(event, None if chat else event.get("type"))
                await close()
                if assembler.complete:
                    await finish("completed")
                    yield sse("[DONE]") if chat else sse(assembler.terminal, assembler.terminal["type"])
                else:
                    terminal = None if chat else assembler.terminal
                    status = "failed" if terminal and terminal["type"] == "response.failed" else "incomplete"
                    error = (None if chat else assembler.error) or error_body(
                        "Upstream stream did not complete", "upstream_error"
                    )
                    await finish(status, error)
                    yield sse(terminal, terminal["type"]) if terminal else sse(sanitize(error, config.secrets), "error")
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as exc:
                await close()
                error = (
                    error_body("Could not save upstream output", "server_error")
                    if isinstance(exc, PersistenceFailure)
                    else sanitize(exc.body, config.secrets)
                    if isinstance(exc, StreamFailure)
                    else upstream_error(exc, config.secrets)[1]
                )
                try:
                    await finish("failed", error)
                except PersistenceFailure:
                    error = error_body("Could not save upstream output", "server_error")
                yield sse(error, "error")
            finally:
                await cleanup()

        return LiveStreamResponse(
            events(),
            cleanup=cleanup,
            active=app.state.active,
            headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.add_api_route("/v1/chat/completions", post, methods=["POST"])
    app.add_api_route("/v1/responses", post, methods=["POST"])
    return app
