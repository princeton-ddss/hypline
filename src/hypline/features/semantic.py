from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger
from pydantic import BaseModel, field_validator

from hypline.bids import BIDS_ENTITY_VALUE_RE, BIDSPath, normalize_bids_filters
from hypline.cache import hypline_cache_dir
from hypline.enums import Device
from hypline.io import skip_existing, write_feature
from hypline.layout import BIDSLayout

if TYPE_CHECKING:
    import numpy as np


def _first_overlapping_word(
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


class HFModelConfig(BaseModel):
    """Loader config for any Hugging Face causal LM.

    `name` is passed verbatim to `from_pretrained` — the hub is open and
    unbounded, so there is no name map or enum. `from_pretrained` validates the
    id better than we could pre-load.
    """

    name: str
    tokenizer_name: str | None = None
    model_dir: Path | None = None
    device: Device = Device.CPU
    layer: int | None = None

    @field_validator("model_dir", mode="after")
    @classmethod
    def _default_model_dir(cls, v: Path | None) -> Path:
        if v is None:
            return hypline_cache_dir("huggingface")
        if not v.is_dir():
            raise ValueError(f"model_dir does not exist: {v}")
        return v


class SemanticFeature:
    def __init__(
        self,
        config: HFModelConfig,
        *,
        bids_root: str | Path,
        bids_filters: list[str] | None = None,
        desc: str | None = None,
        force: bool = False,
    ):
        if desc is not None and not BIDS_ENTITY_VALUE_RE.match(desc):
            raise ValueError(f"Invalid desc: {desc!r}")

        self._config = config
        self._layout = BIDSLayout(bids_root)
        self._bids_filters = normalize_bids_filters(bids_filters, reserved={"dyad"})
        self._desc = desc
        self._force = force

        self._load()

        # Default to the middle layer once the loaded config's depth is known.
        # hidden_states has num_hidden_layers + 1 entries (embeddings + each
        # layer), so layer 0 is the static table and num_hidden_layers is valid.
        n_layers = self._model.config.num_hidden_layers
        if config.layer is None:
            self._layer = n_layers // 2
        elif not 0 <= config.layer <= n_layers:
            raise ValueError(
                f"layer {config.layer} out of range for {config.name} (0..{n_layers})"
            )
        else:
            self._layer = config.layer

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        )

        self._torch = torch

        cfg = self._config
        cache_dir = str(cfg.model_dir)
        tokenizer_name = cfg.tokenizer_name or cfg.name

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=cache_dir
        )

        # Fast (Rust) tokenizer is required for `return_offsets_mapping`, which
        # drives token->word alignment. It is the `AutoTokenizer` default.
        if not self._tokenizer.is_fast:
            raise ValueError(
                f"{tokenizer_name!r} has no fast tokenizer; semantic features "
                "need offset mapping (a Rust tokenizer)"
            )

        # `_forward` prepends `bos_token_id`, so it must exist. Every major decoder
        # family defines one; this rejects only the rare no-BOS tail, which would
        # otherwise crash deep in `_forward` with an opaque tensor error.
        if self._tokenizer.bos_token_id is None:
            raise ValueError(
                f"{tokenizer_name!r} has no bos_token_id; semantic features need "
                "one to give the first token left-context"
            )

        model = AutoModelForCausalLM.from_pretrained(cfg.name, cache_dir=cache_dir)

        # `AutoModelForCausalLM` silently loads non-causal checkpoints (BERT ->
        # `BertLMHeadModel`) and emits garbage with no downstream tripwire, so
        # check the checkpoint's native architecture against the causal mapping.
        causal_archs = set(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values())
        native = model.config.architectures or []
        if not any(arch in causal_archs for arch in native):
            raise ValueError(
                f"{cfg.name!r} is not a causal/decoder LM (architectures="
                f"{native}); semantic features need a causal LM"
            )

        # Token ids must index the right embedding rows. `<=` (not `==`) tolerates
        # models that pad the embedding table to a multiple of 64/128.
        if len(self._tokenizer) > model.config.vocab_size:
            raise ValueError(
                f"tokenizer vocab ({len(self._tokenizer)}) exceeds model vocab "
                f"({model.config.vocab_size}); incompatible tokenizer/model pair"
            )

        self._model = model.eval().to(cfg.device)

    def generate(self, dyad_id: str):
        transcripts = self._layout.find.stimuli(
            dyad=dyad_id,
            kind="transcript",
            ext=".csv",
            bids_filters=self._bids_filters,
        )

        for transcript in transcripts:
            out = self._layout.path.feature(
                source=transcript, kind="semantic", desc=self._desc
            )
            if skip_existing(out.path, force=self._force):
                continue
            logger.info("Generating semantic features for {}", transcript.path.name)
            self._generate_one(transcript, out.path)

    def _generate_one(self, source: BIDSPath, out_path: Path) -> None:
        df = pl.read_csv(source.path)
        start_times = df.get_column("start_time").to_list()
        raw_words = df.get_column("word").cast(pl.Utf8).to_list()

        # Join timed and untimed words into one string and record each word's
        # char span from that same construction (single source of truth — offsets
        # index this string, not the CSV). Null words cannot be tokenized; drop
        # them, but retain untimed (null start_time) words as real LM context.
        words, word_starts, spans = [], [], []
        cursor = 0
        null_words = 0
        for start, word in zip(start_times, raw_words):
            if word is None:
                null_words += 1
                continue
            if words:
                cursor += 1  # the single joining space
            span_start = cursor
            cursor += len(word)
            words.append(word)
            word_starts.append(start)
            spans.append((span_start, cursor))

        if null_words:
            logger.warning(
                "Skipped {} null-word row(s) in {}", null_words, source.path.name
            )

        if not words:
            raise ValueError(
                f"{source.path.name}: no usable words (all null); cannot "
                "generate semantic features"
            )

        text = " ".join(words)
        encoding = self._tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]

        # +1 for the bos prefix _forward prepends — the model sees that many
        # positions, so the guard must count it. `max_position_embeddings`
        # resolves correctly across families (n_positions on gpt-2, native
        # elsewhere); a larger model gets its larger budget automatically.
        max_pos = self._model.config.max_position_embeddings
        if len(token_ids) + 1 > max_pos:
            raise ValueError(
                f"{source.path.name}: {len(token_ids)} tokens (+bos) exceed "
                f"{self._config.name} context limit "
                f"max_position_embeddings={max_pos}"
            )

        tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
        rows_start, rows_word = [], []
        for token_start, token_end in offsets:
            idx = _first_overlapping_word(token_start, token_end, spans)
            rows_start.append(word_starts[idx])
            rows_word.append(words[idx])

        states, metrics = self._forward(token_ids)

        data = {"start_time": rows_start, "token": tokens, "word": rows_word}
        if metrics is not None:
            data.update(metrics)
        data["feature"] = pl.Series(
            "feature",
            states.tolist(),
            dtype=pl.Array(pl.Float64, states.shape[1]),
        )

        out_df = pl.DataFrame(data)
        metadata = {
            "hf_model": self._config.name,
            "hf_tokenizer": self._config.tokenizer_name or self._config.name,
            "layer": self._layer,
        }
        write_feature(out_df, out_path, metadata=metadata)
        logger.debug("Wrote semantic feature to {}", out_path)

    def _forward(
        self, token_ids: list[int]
    ) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
        """Run the model and extract per-token hidden states + LM metrics.

        Prepends `bos_token_id` at position 0 so the first real token has
        in-distribution left-context, then drops it from all output. Returns
        `(states, metrics)`; metrics is None for the layer-0 static path (no
        forward pass, so rank/true_prob/entropy are undefined).
        """
        torch = self._torch
        device = self._config.device
        # gpt-2 led training documents with <|endoftext|> (= bos id); a BOS-led
        # transcript is in-distribution. Read from the tokenizer, never hardcode.
        ids = [self._tokenizer.bos_token_id] + token_ids
        batch = torch.tensor([ids], dtype=torch.long, device=device)

        if self._layer == 0:
            with torch.no_grad():
                table = self._model.get_input_embeddings().weight
                states = table[batch][0, 1:].numpy(force=True)
            return states, None

        with torch.no_grad():
            output = self._model(batch, output_hidden_states=True)
            # Drop the prefix at position 0: output rows = real sub-word tokens
            states = output.hidden_states[self._layer][0, 1:].numpy(force=True)
            logits = output.logits[0]

            # Score each real token by the distribution that predicted it:
            # position i predicts token i+1, so logits[:-1] aligns with targets
            # = ids[1:] (the real tokens). Gather the per-position diagonal — each
            # target's prob from its own predicting row, not a single fixed row.
            targets = batch[0, 1:]
            pred_logits = logits[:-1]
            probs = pred_logits.softmax(-1)

            order = pred_logits.argsort(descending=True, dim=-1)
            rank = torch.eq(order, targets[:, None]).nonzero()[:, 1]
            true_prob = probs[torch.arange(probs.shape[0]), targets]
            entropy = torch.distributions.Categorical(probs=probs).entropy()

        metrics = {
            "rank": rank.numpy(force=True),
            "true_prob": true_prob.numpy(force=True),
            "entropy": entropy.numpy(force=True),
        }
        return states, metrics
