"""100% Offline Local HTTP Application Interface for FieldWise PS1."""

from typing import Any

__all__ = ["OfflineAssistantHTTPServer", "create_offline_server"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
