"""Startup-only configuration. Credentials never participate in repr or diagnostics."""

import ipaddress
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Model:
    id: str
    name: str


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    protocol: str
    api_key: str = field(repr=False)
    models: tuple[Model, ...]


@dataclass(frozen=True)
class Config:
    providers: tuple[Provider, ...]
    database: Path
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = field(default=None, repr=False)

    @property
    def routes(self) -> dict[str, Provider]:
        return {m.id: p for p in self.providers for m in p.models}

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(p.api_key for p in self.providers) + ((self.api_key,) if self.api_key else ())


def loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a nonempty string")
    return value


def mapping(value, required: set[str], optional: set[str], label: str) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise ConfigError(f"Invalid {label} schema (check required and unknown fields)")
    return value


def base_url(value) -> str:
    value = text(value, "provider base_url")
    try:
        if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError
        parsed = urlsplit(value if "://" in value else "//" + value)
        host = parsed.hostname
        port = parsed.port
        if not host or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
        try:
            ipaddress.ip_address(host)
        except ValueError:
            domain = host.encode("idna").decode("ascii").rstrip(".")
            if len(domain) > 253 or any(
                not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) for label in domain.split(".")
            ):
                raise ValueError from None
        scheme = parsed.scheme or ("http" if loopback(host) else "")
        if scheme not in {"http", "https"} or (scheme == "http" and not loopback(host)):
            raise ValueError
        return urlunsplit((scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    except ValueError, UnicodeError:
        raise ConfigError("Invalid provider base_url: use HTTPS, or HTTP for loopback only") from None


def credential(value) -> str:
    value = text(value, "provider api_key")
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
    if match:
        resolved = os.environ.get(match[1])
        if not resolved or not resolved.strip():
            raise ConfigError("A required credential environment variable is missing or empty")
        return resolved
    if value.startswith("${"):
        raise ConfigError("api_key environment references must use a whole-value ${ENV_VAR}")
    return value


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path).resolve()
    try:
        with path.open() as f:
            raw = yaml.safe_load(f)
    except OSError, yaml.YAMLError:
        raise ConfigError("Unable to read configuration: check the file and YAML syntax") from None
    raw = mapping(raw, {"provider_list"}, {"server"}, "configuration")
    if not isinstance(raw["provider_list"], list) or not raw["provider_list"]:
        raise ConfigError("provider_list must be a nonempty list")
    providers, names, ids = [], set(), set()
    for item in raw["provider_list"]:
        item = mapping(item, {"name", "base_url", "protocol", "api_key", "models"}, set(), "provider")
        name = text(item["name"], "provider name")
        if name in names:
            raise ConfigError("Duplicate provider name")
        names.add(name)
        protocol = item["protocol"]
        if protocol not in ("openai-completions", "openai-responses"):
            raise ConfigError("Unsupported provider protocol")
        if not isinstance(item["models"], list) or not item["models"]:
            raise ConfigError("Provider models must be a nonempty list")
        models = []
        for m in item["models"]:
            m = mapping(m, {"id", "name"}, set(), "model")
            mid = text(m["id"], "model id")
            if mid in ids:
                raise ConfigError("Duplicate model ID")
            ids.add(mid)
            models.append(Model(mid, text(m["name"], "model name")))
        providers.append(
            Provider(name, base_url(item["base_url"]), protocol, credential(item["api_key"]), tuple(models))
        )
    server = mapping(raw.get("server", {}), set(), {"host", "port", "database", "api_key_env"}, "server")
    host = text(server.get("host", "127.0.0.1"), "server host")
    # A bind address must be an IP address or localhost, not an arbitrary DNS name.
    try:
        if host != "localhost":
            ipaddress.ip_address(host)
    except ValueError:
        raise ConfigError("server host must be an IP address or localhost") from None
    port = server.get("port", 8000)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ConfigError("server port must be an integer from 1 to 65535")
    token = None
    if "api_key_env" in server:
        env = text(server["api_key_env"], "server api_key_env")
        token = os.environ.get(env)
        if not token or not token.strip():
            raise ConfigError("Inbound authentication environment variable is missing or empty")
    if not loopback(host) and not token:
        raise ConfigError("Non-loopback binding requires an inbound API token and TLS or a trusted tunnel")
    database = path.parent / text(server.get("database", "./data/glorp.sqlite3"), "server database")
    return Config(tuple(providers), database.resolve(), host, port, token)
