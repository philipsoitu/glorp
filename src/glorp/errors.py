"""Safe diagnostics: never stringify arbitrary upstream exceptions."""

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError


def sanitize(value, secrets: tuple[str, ...]):
    if isinstance(value, str):
        for secret in sorted(set(secrets), key=len, reverse=True):
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [sanitize(v, secrets) for v in value]
    if isinstance(value, dict):
        return {
            sanitize(str(k), secrets): sanitize(v, secrets)
            for k, v in value.items()
            if str(k).lower().replace("-", "_")
            not in {
                "authorization",
                "proxy_authorization",
                "cookie",
                "set_cookie",
                "api_key",
                "headers",
                "stack",
                "stacktrace",
                "stack_trace",
                "traceback",
            }
        }
    return value


def error_body(message, kind="invalid_request_error", code=None):
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}


def stored_error(exc, public_error, secrets):
    """Retain sanitized non-JSON error bodies, but don't expose HTML/debug pages over HTTP."""
    if isinstance(exc, APIStatusError) and exc.body is not None and not isinstance(exc.body, dict):
        return {**public_error, "upstream_body": sanitize(exc.body, secrets)}
    return public_error


def upstream_error(exc, secrets):
    if isinstance(exc, APITimeoutError):
        return 504, error_body("Upstream request timed out", "timeout_error")
    if isinstance(exc, APIConnectionError):
        return 502, error_body("Could not connect to upstream", "connection_error")
    if isinstance(exc, APIStatusError):
        body = sanitize(exc.body, secrets)
        if isinstance(body, dict):
            if "error" not in body:
                body = {"error": body}
            return exc.status_code, body
        return exc.status_code, error_body("Upstream HTTP request failed", "upstream_error")
    if isinstance(exc, APIError) and isinstance(exc.body, dict):
        body = sanitize(exc.body, secrets)
        return 502, body if "error" in body else {"error": body}
    return 502, error_body("Upstream response failed", "upstream_error")
