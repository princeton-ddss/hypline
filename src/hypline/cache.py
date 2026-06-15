"""Default cache locations for downloaded model artifacts.

One root (`~/.cache/hypline`, overridable via `HYPLINE_CACHE`) with per-tool
subdirs. Configs that load HF transformers artifacts share the `huggingface`
subdir so transformers dedupes them; whisperx weights are a different format
(CTranslate2) and live under their own subdir.
"""

import os
from pathlib import Path


def hypline_cache_dir(subdir: str) -> Path:
    root = os.environ.get("HYPLINE_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "hypline"
    d = base / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d
