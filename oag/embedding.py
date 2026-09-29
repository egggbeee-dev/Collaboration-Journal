"""Text embedders for the Auction (proposal) stage. Scores are cosine similarities, so no LLM is involved.

- SBertEmbedder : sentence-transformers model (use this in Colab / real runs)
- HashingEmbedder: dependency-free bag-of-words vectors (offline tests / --mock runs)
"""
from __future__ import annotations

import re
import zlib

import numpy as np


class HashingEmbedder:
    _STOP = {"a", "an", "the", "to", "and", "of", "in", "on", "it", "over", "up", "have", "has", "be", "is", "are", "for", "with", "go", "receive", "hand", "bring", "take", "out"}

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @staticmethod
    def _stem(w: str) -> str:
        for suf in ("ing", "ed", "es", "s", "e"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)]
        return w

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w in self._STOP:
                    continue
                out[i, zlib.crc32(self._stem(w).encode()) % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class SBertEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def get_embedder(name: str = "sbert"):
    if name == "hash":
        return HashingEmbedder()
    if name == "sbert":
        return SBertEmbedder()
    raise ValueError(f"unknown embedder {name!r} (use 'sbert' or 'hash')")
