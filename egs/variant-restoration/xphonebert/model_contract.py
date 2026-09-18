"""Resolve the text-only contract from Qwen3.5's multimodal config wrapper."""

from __future__ import annotations


def text_config(config):
    value = getattr(config, "text_config", None)
    return value if value is not None else config


def text_hidden_size(config) -> int | None:
    return getattr(text_config(config), "hidden_size", None)


def context_limit(config) -> int | None:
    return getattr(text_config(config), "max_position_embeddings", None) or getattr(config, "max_position_embeddings", None)
