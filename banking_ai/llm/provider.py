"""Configurable chat-model factory (OpenAI, Gemini or none/mock). No keys in code - everything from settings."""
from __future__ import annotations

from typing import Any, Optional

from banking_ai.config.logging_config import get_logger
from banking_ai.config.settings import Settings

logger = get_logger("llm")
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "gemini": "gemini-2.0-flash"}


def build_chat_model(settings: Settings) -> Optional[Any]:
    """Return a LangChain chat model, or ``None`` for mock/demo mode (or if the provider cannot be created)."""
    if not settings.llm_enabled:
        logger.info("LLM disabled (provider=%s, key set=%s): running in deterministic demo mode",
                    settings.llm_provider, bool(settings.llm_api_key))
        return None
    provider = settings.llm_provider
    model = settings.llm_model or DEFAULT_MODELS.get(provider, "")
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            kwargs: dict[str, Any] = {"model": model, "api_key": settings.llm_api_key, "temperature": settings.llm_temperature,
                                      "timeout": settings.llm_timeout_seconds, "max_retries": 1}
            if settings.llm_base_url:
                kwargs["base_url"] = settings.llm_base_url
            return ChatOpenAI(**kwargs)
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=model, google_api_key=settings.llm_api_key,
                                          temperature=settings.llm_temperature, timeout=settings.llm_timeout_seconds)
        logger.error("Unsupported LLM_PROVIDER '%s'; falling back to demo mode", provider)
    except ImportError as exc:
        logger.error("LLM provider package missing (%s); falling back to demo mode", exc)
    return None
