"""Embedding models. All expose the LangChain ``Embeddings`` interface (embed_documents / embed_query).

* ``tfidf``       - default. Offline, deterministic, no download. Lexical (word-overlap) similarity.
* ``huggingface`` - recommended upgrade: semantic sentence-transformer embeddings (downloads a model once).
* ``openai``      - OpenAI embeddings (needs LLM_API_KEY).
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from banking_ai.config.settings import Settings

_TOKEN = re.compile(r"[a-z0-9]+")
_SYNONYMS = {"min": ["minimum"], "bal": ["balance"], "roi": ["interest", "rate"], "fd": ["fixed", "deposit"],
             "rd": ["recurring", "deposit"], "txn": ["transaction"], "fees": ["charge", "fee"], "fee": ["fee", "charge"],
             "penalty": ["penalty", "charge"], "amb": ["average", "monthly", "balance"]}
_KEEP = {"no", "not", "zero"}


def analyze(text: str) -> list[str]:
    """Lowercase, tokenise, drop stop-words, apply light plural stemming + banking synonyms, add bigrams."""
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in ENGLISH_STOP_WORDS and tok not in _KEEP:
            continue
        if tok in _SYNONYMS:
            tokens.extend(_SYNONYMS[tok])
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        if len(tok) > 6 and tok.endswith("ing"):
            tok = tok[:-3]
        tokens.append(tok)
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


class TfidfEmbeddings:
    """Dependency-light embeddings: L2-normalised TF-IDF vectors (cosine-ready)."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(analyzer=analyze, sublinear_tf=True, norm="l2")
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "TfidfEmbeddings":
        self._vectorizer.fit(list(texts))
        self._fitted = True
        return self

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._require_fit()
        return self._vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        self._require_fit()
        return self._vectorizer.transform([text]).toarray()[0].tolist()

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddings must be fitted on the knowledge-base chunks first.")


def build_embeddings(settings: Settings, corpus: Sequence[str]) -> Any:
    """Create the embedding model chosen by ``EMBEDDING_PROVIDER`` (fitting it when it is corpus-based)."""
    provider = settings.embedding_provider
    if provider == "tfidf":
        return TfidfEmbeddings().fit(corpus)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model, encode_kwargs={"normalize_embeddings": True})
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model or "text-embedding-3-small", api_key=settings.llm_api_key)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER '{provider}'. Use tfidf, huggingface or openai.")
