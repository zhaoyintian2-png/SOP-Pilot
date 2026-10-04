# -*- coding: utf-8 -*-
"""
================================================================================
  retrieval.py — Vector Retrieval
================================================================================
Encapsulates the embedding + retrieval pipeline in a :class:`Retriever`:
  - Tries the dense model ``sentence-transformers/all-MiniLM-L6-v2`` first.
  - Falls back to TF-IDF (sklearn) if the dense model cannot be loaded.
  - Embeds all corpus chunks once, then answers queries via cosine similarity
    over the normalized embedding matrix.

The retriever is intentionally stateless beyond the corpus it is built from,
so it can be constructed fresh in tests or swapped for another implementation.
================================================================================
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME


class Retriever:
    """Top-k retriever over a list of chunk dicts."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self._embedder: Any = None
        self._embed_backend: str | None = None
        self._tfidf_vectorizer: Any = None
        self._tfidf_matrix: Any = None
        self._chunk_emb: np.ndarray | None = None

        self._load_embedder()
        self._build_chunk_embeddings()

    # ------------------------------------------------------------------
    # Embedder setup
    # ------------------------------------------------------------------

    def _load_embedder(self) -> str:
        """Load the embedding backend.

        Returns the active backend name (``"dense"`` or ``"tfidf"``).
        """
        texts = [c["text"] for c in self.chunks]
        try:
            from sentence_transformers import SentenceTransformer

            print(f"[Embed] Loading {EMBEDDING_MODEL_NAME} ...")
            self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
            self._embed_backend = "dense"
            print("[Embed] Dense embedding model loaded successfully.")
            return "dense"
        except Exception as e:  # noqa: BLE001
            print(f"[Embed] Dense model load failed, falling back to TF-IDF: {e}")

        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf_vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
        self._embed_backend = "tfidf"
        print("[Embed] TF-IDF fallback ready.")
        return "tfidf"

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts using the active backend (L2-normalized)."""
        from sklearn.preprocessing import normalize

        if self._embed_backend == "dense":
            vecs = self._embedder.encode(
                texts, convert_to_numpy=True, show_progress_bar=False
            )
            return normalize(vecs)
        # tfidf
        vecs = self._tfidf_vectorizer.transform(texts)
        return normalize(vecs).toarray()

    def _build_chunk_embeddings(self) -> None:
        """Pre-compute and cache embeddings for the full corpus."""
        if not self.chunks:
            self._chunk_emb = np.zeros((0, 1))
            return
        texts = [c["text"] for c in self.chunks]
        self._chunk_emb = self._embed(texts)
        print(f"[Embed] Chunk embedding matrix shape: {self._chunk_emb.shape}")

    @property
    def backend(self) -> str | None:
        return self._embed_backend

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve(self, question: str, k: int = DEFAULT_TOP_K) -> list[dict]:
        """Return the top-``k`` chunks most similar to ``question``.

        Each returned chunk is a copy of the original chunk dict with an
        extra ``"score"`` key (cosine similarity).
        """
        if not self.chunks or self._chunk_emb is None:
            return []
        q_vec = self._embed([question])
        scores = (self._chunk_emb @ q_vec.T).flatten()
        top_idx = np.argsort(scores)[::-1][:k]
        results: list[dict] = []
        for idx in top_idx:
            item = dict(self.chunks[idx])
            item["score"] = float(scores[idx])
            results.append(item)
        return results
