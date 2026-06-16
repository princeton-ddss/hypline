"""Shared helpers for transcript-derived features.

Transcript loading plus word/token char-span alignment. All three word-level
features (semantic, phonemic, syntactic) load a transcript the same way — drop
null-word rows, retain untimed words, forward-fill `turn_sub` — then join the
kept words, tokenize, and attribute each token back to its source word by
char-span overlap. The loader, span math, and overlap test live here so the
generators share one source of truth.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from loguru import logger


def load_transcript_words(
    path: Path,
) -> tuple[list[float | None], list[str], list[str | None]]:
    """Return `(starts, words, turns)` for a transcript's usable words.

    Drops null-word rows (cannot be tokenized) but retains untimed (null
    `start_time`) words — they are real context for the LM/parser. `turn_sub` is
    forward-filled *before* the drop: stamp_turns leaves untimed words null, but
    they belong to the surrounding utterance, so a null would fragment a turn
    mid-sentence and corrupt its parse. Raises if no usable words remain.
    """
    name = path.name
    df = pl.read_csv(path, schema_overrides={"turn_sub": pl.Utf8}).with_columns(
        pl.col("turn_sub").fill_null(strategy="forward")
    )

    null_words = df.get_column("word").is_null().sum()
    if null_words:
        logger.warning("Skipped {} null-word row(s) in {}", null_words, name)

    df = df.filter(pl.col("word").is_not_null())
    if df.is_empty():
        raise ValueError(
            f"{name}: no usable words (all null); cannot generate features"
        )

    return (
        df.get_column("start_time").to_list(),
        df.get_column("word").cast(pl.Utf8).to_list(),
        df.get_column("turn_sub").to_list(),
    )


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
