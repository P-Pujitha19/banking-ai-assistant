"""Central, environment-driven configuration.

All tunables come from environment variables (optionally loaded from ``.env``).
No secrets are hardcoded anywhere in the code base.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Load ``.env`` from the project root if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _get_bool(name: str, default: bool) -> bool:
    return _get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError:
        return default


def _get_path(name: str, default_relative: str) -> Path:
    path = Path(_get(name, default_relative))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # --- general / logging -------------------------------------------------
    app_env: str = "development"
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    log_json: bool = True
    demo_mode: bool = True

    # --- LLM ---------------------------------------------------------------
    llm_provider: str = "mock"  # mock | openai | gemini
    llm_model: str = ""
    llm_api_key: str = field(default="", repr=False)
    llm_base_url: str = ""
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 30.0
    llm_max_tool_steps: int = 4
    use_llm_entity_extraction: bool = False

    # --- mock bank ----------------------------------------------------------
    bank_api_mode: str = "http"  # http | inprocess
    bank_api_url: str = "http://127.0.0.1:8001"
    bank_api_key: str = field(default="", repr=False)
    bank_api_timeout_seconds: float = 5.0
    bank_data_dir: Path = PROJECT_ROOT / "data" / "bank"
    bank_as_of_date: str = ""  # ISO date; empty -> latest transaction date

    # --- assistant API / UI ------------------------------------------------
    assistant_api_url: str = "http://127.0.0.1:8000"

    # --- ML ----------------------------------------------------------------
    intent_model_dir: Path = PROJECT_ROOT / "models" / "intent"
    intent_dataset_path: Path = PROJECT_ROOT / "data" / "ml" / "intent_dataset.csv"
    intent_backend: str = "auto"  # auto | tensorflow | sklearn
    intent_confidence_threshold: float = 0.60

    # --- RAG ---------------------------------------------------------------
    kb_dir: Path = PROJECT_ROOT / "knowledge_base"
    vector_store_dir: Path = PROJECT_ROOT / "models" / "vector_store"
    vector_backend: str = "auto"  # auto | chroma | numpy
    embedding_provider: str = "tfidf"  # tfidf | huggingface | openai
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 4
    rag_min_score: float = 0.12
    chunk_size: int = 600
    chunk_overlap: int = 80

    # --- safety / sessions ---------------------------------------------------
    max_message_chars: int = 500
    session_ttl_minutes: int = 30
    max_history_turns: int = 10
    prompts_dir: Path = PROJECT_ROOT / "banking_ai" / "prompts"

    @property
    def llm_enabled(self) -> bool:
        """True when a real LLM provider with credentials is configured."""
        return self.llm_provider.lower() not in {"", "mock", "none"} and bool(self.llm_api_key)

    def with_overrides(self, **changes: Any) -> "Settings":
        """Return a copy with selected fields replaced (handy for tests)."""
        return replace(self, **changes)

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from process environment variables."""
        _load_dotenv()
        log_file = _get("LOG_FILE", "logs/assistant.log")
        return cls(
            app_env=_get("APP_ENV", "development"),
            log_level=_get("LOG_LEVEL", "INFO").upper(),
            log_file=_get_path("LOG_FILE", log_file) if log_file.lower() != "none" else None,
            log_json=_get_bool("LOG_JSON", True),
            demo_mode=_get_bool("DEMO_MODE", True),
            llm_provider=_get("LLM_PROVIDER", "mock").lower(),
            llm_model=_get("LLM_MODEL", ""),
            llm_api_key=_get("LLM_API_KEY", ""),
            llm_base_url=_get("LLM_BASE_URL", ""),
            llm_temperature=_get_float("LLM_TEMPERATURE", 0.0),
            llm_timeout_seconds=_get_float("LLM_TIMEOUT_SECONDS", 30.0),
            llm_max_tool_steps=_get_int("LLM_MAX_TOOL_STEPS", 4),
            use_llm_entity_extraction=_get_bool("USE_LLM_ENTITY_EXTRACTION", False),
            bank_api_mode=_get("BANK_API_MODE", "http").lower(),
            bank_api_url=_get("BANK_API_URL", "http://127.0.0.1:8001"),
            bank_api_key=_get("BANK_API_KEY", ""),
            bank_api_timeout_seconds=_get_float("BANK_API_TIMEOUT_SECONDS", 5.0),
            bank_data_dir=_get_path("BANK_DATA_DIR", "data/bank"),
            bank_as_of_date=_get("BANK_AS_OF_DATE", ""),
            assistant_api_url=_get("ASSISTANT_API_URL", "http://127.0.0.1:8000"),
            intent_model_dir=_get_path("INTENT_MODEL_DIR", "models/intent"),
            intent_dataset_path=_get_path("INTENT_DATASET_PATH", "data/ml/intent_dataset.csv"),
            intent_backend=_get("INTENT_BACKEND", "auto").lower(),
            intent_confidence_threshold=_get_float("INTENT_CONFIDENCE_THRESHOLD", 0.60),
            kb_dir=_get_path("KB_DIR", "knowledge_base"),
            vector_store_dir=_get_path("VECTOR_STORE_DIR", "models/vector_store"),
            vector_backend=_get("VECTOR_BACKEND", "auto").lower(),
            embedding_provider=_get("EMBEDDING_PROVIDER", "tfidf").lower(),
            embedding_model=_get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            rag_top_k=_get_int("RAG_TOP_K", 4),
            rag_min_score=_get_float("RAG_MIN_SCORE", 0.12),
            chunk_size=_get_int("CHUNK_SIZE", 600),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 80),
            max_message_chars=_get_int("MAX_MESSAGE_CHARS", 500),
            session_ttl_minutes=_get_int("SESSION_TTL_MINUTES", 30),
            max_history_turns=_get_int("MAX_HISTORY_TURNS", 10),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings object."""
    return Settings.from_env()
