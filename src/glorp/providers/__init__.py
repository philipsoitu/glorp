"""Minimal native-dialect adapter contract and startup registry."""

from typing import Any, Protocol

from ..config import Provider


class Adapter(Protocol):
    async def create(self, endpoint: str, body: dict) -> Any: ...
    async def close(self) -> None: ...


def create_adapter(provider: Provider) -> Adapter:
    from .openai import OpenAIAdapter

    registry = {"openai-completions": OpenAIAdapter, "openai-responses": OpenAIAdapter}
    return registry[provider.protocol](provider)
