"""Shared helpers for transcript-derived features.

Currently: word/token char-span alignment. Both semantic and syntactic features
join transcript words into a single string, tokenize it (HF or spaCy), then
attribute each token back to its source word by char-span overlap. The span math
and the overlap test live here so the two generators share one source of truth.
"""

from __future__ import annotations


def build_word_spans(words: list[str]) -> list[tuple[int, int]]:
    """Char span of each word in `" ".join(words)`.

    `spans[i]` is the `[start, end)` span of `words[i]` in the joined string —
    the string token offsets must index. Callers own dropping null words first;
    this assumes every entry is joinable.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, word in enumerate(words):
        if i:
            cursor += 1  # the single joining space
        span_start = cursor
        cursor += len(word)
        spans.append((span_start, cursor))
    return spans


def first_overlapping_word(
    token_start: int, token_end: int, spans: list[tuple[int, int]]
) -> int:
    """Index of the first word whose char span overlaps [token_start, token_end).

    Overlap, not char_start containment: a fast tokenizer folds the leading
    space into the offset (gpt-2 `Ġcat` -> (3,7) while `cat` spans [4,7)), so
    containment would match no word. A token always overlaps at least its start
    word, so this never fails to attribute (which would desync output rows from
    the model's per-token states).
    """
    for idx, (word_start, word_end) in enumerate(spans):
        if token_start < word_end and word_start < token_end:
            return idx
    raise ValueError(f"token span [{token_start}, {token_end}) overlaps no word span")
