"""Pure-Python/NumPy vocabulary: tokenisation -> integer ids -> padding (explicit, easy to learn from)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from banking_ai.nlp.preprocessing import tokenize

PAD, UNK = "<pad>", "<unk>"


class Vocabulary:
    """Maps tokens to ids. Id 0 is reserved for padding (so the model can mask it), id 1 for unknown."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id

    @classmethod
    def build(cls, texts: Iterable[str], max_size: int = 5000, min_freq: int = 1) -> "Vocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        mapping = {PAD: 0, UNK: 1}
        for token, freq in counts.most_common(max_size - 2):
            if freq >= min_freq:
                mapping[token] = len(mapping)
        return cls(mapping)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, max_len: int) -> np.ndarray:
        """Tokenise, map to ids, truncate and right-pad with zeros to ``max_len``."""
        ids = [self.token_to_id.get(tok, 1) for tok in tokenize(text)][:max_len]
        ids += [0] * (max_len - len(ids))
        return np.asarray(ids, dtype=np.int32)

    def encode_batch(self, texts: Iterable[str], max_len: int) -> np.ndarray:
        return np.stack([self.encode(t, max_len) for t in texts]) if texts else np.zeros((0, max_len), dtype=np.int32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text(encoding="utf-8")))
