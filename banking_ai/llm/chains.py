"""LangChain chains: grounded response generation and structured (Pydantic) entity extraction."""
from __future__ import annotations

import json
from typing import Any

from banking_ai.llm.prompts import PromptLibrary
from banking_ai.nlp.entities import ExtractedEntities


class LlmResponder:
    """``ChatPromptTemplate | chat_model | StrOutputParser`` - writes the final answer from verified facts only."""

    def __init__(self, chat_model: Any, prompts: PromptLibrary) -> None:
        self.chat_model, self.prompts = chat_model, prompts

    def respond(self, *, question: str, intent: str, facts: dict[str, Any], kb_context: str, notes: list[str],
                history: list[tuple[str, str]]) -> str:
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts.response_system(with_kb=bool(kb_context))),
            MessagesPlaceholder("history"),
            ("human", self.prompts.get("response_generation")),
        ])
        chain = prompt | self.chat_model | StrOutputParser()
        past = [HumanMessage(c) if r == "user" else AIMessage(c) for r, c in history]
        return chain.invoke({
            "history": past, "question": question, "intent": intent,
            "notes": "; ".join(notes) or "none",
            "facts": json.dumps(facts, default=str, ensure_ascii=False),
            "kb_context": kb_context or "(none)",
        }).strip()


class LlmEntityExtractor:
    """Structured extraction: ``prompt | chat_model.with_structured_output(ExtractedEntities)``."""

    def __init__(self, chat_model: Any, prompts: PromptLibrary) -> None:
        self.chat_model, self.prompts = chat_model, prompts

    def extract(self, text: str) -> ExtractedEntities:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([("system", self.prompts.get("entity_extraction")), ("human", "{message}")])
        chain = prompt | self.chat_model.with_structured_output(ExtractedEntities)
        result = chain.invoke({"message": text})
        return result if isinstance(result, ExtractedEntities) else ExtractedEntities(**dict(result))
