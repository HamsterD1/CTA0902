"""Lossless windowing for frozen XPhoneBERT inputs longer than its position limit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IpaWindow:
    input_ids: list[int]
    content_positions: list[int]
    original_token_indices: list[int]


@dataclass(frozen=True)
class ChunkedIpa:
    content_token_count: int
    sequence_token_count: int
    windows: list[IpaWindow]


def window_ranges(content_token_count: int, max_content_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    """Cover every original token using overlapping, position-safe windows."""
    if content_token_count < 1:
        return []
    if max_content_tokens < 1:
        raise ValueError("XPhoneBERT max position limit leaves no content capacity")
    if not 0 <= overlap_tokens < max_content_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than max_content_tokens")
    if content_token_count <= max_content_tokens:
        return [(0, content_token_count)]
    stride = max_content_tokens - overlap_tokens
    last_start = content_token_count - max_content_tokens
    starts = [0]
    while starts[-1] < last_start:
        next_start = min(starts[-1] + stride, last_start)
        if next_start == starts[-1]:
            break
        starts.append(next_start)
    return [(start, min(start + max_content_tokens, content_token_count)) for start in starts]


def _ids(encoded) -> list[int]:
    value = encoded["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token, int) for token in value):
        raise TypeError("Expected one token-id list from the XPhoneBERT tokenizer")
    return value


def chunk_ipa(tokenizer, ipa: str, max_input_tokens: int = 512, overlap_tokens: int = 128) -> ChunkedIpa:
    """Tokenize once, then chunk existing IDs without changing IPA text or boundaries."""
    content_ids = _ids(tokenizer(ipa, add_special_tokens=False))
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("XPhoneBERT tokenizer must expose BOS and EOS token ids")
    special_count = 2
    max_content_tokens = max_input_tokens - special_count
    if not content_ids:
        return ChunkedIpa(0, 2, [IpaWindow([bos_token_id, eos_token_id], [0, 1], [0, 1])])
    windows = []
    for start, end in window_ranges(len(content_ids), max_content_tokens, overlap_tokens):
        window_content_ids = content_ids[start:end]
        # XPhoneBERT is RoBERTa: each single sequence is exactly BOS + content + EOS.
        input_ids = [bos_token_id, *window_content_ids, eos_token_id]
        special_mask = [1, *([0] * len(window_content_ids)), 1]
        content_positions = [index for index, is_special in enumerate(special_mask) if not is_special]
        if len(input_ids) > max_input_tokens:
            raise ValueError("XPhoneBERT window exceeds its position-safe input limit")
        if len(content_positions) != end - start:
            raise ValueError("XPhoneBERT special-token plan does not preserve window content")
        state_positions = list(content_positions)
        state_indices = [index + 1 for index in range(start, end)]
        if start == 0:
            state_positions.insert(0, 0)
            state_indices.insert(0, 0)
        if end == len(content_ids):
            state_positions.append(len(input_ids) - 1)
            state_indices.append(len(content_ids) + 1)
        windows.append(IpaWindow(input_ids, state_positions, state_indices))
    sequence_token_count = len(content_ids) + 2
    coverage = [0] * sequence_token_count
    for window in windows:
        for index in window.original_token_indices:
            coverage[index] += 1
    if any(count == 0 for count in coverage):
        raise ValueError("Chunk plan dropped at least one XPhoneBERT sequence token")
    return ChunkedIpa(len(content_ids), sequence_token_count, windows)
