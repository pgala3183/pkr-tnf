"""Model serving utilities (ONNX export, inference APIs)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["app", "create_app"]


def __getattr__(name: str):
    if name in ("app", "create_app"):
        from poker_transformer.serving.api import app, create_app

        return app if name == "app" else create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
