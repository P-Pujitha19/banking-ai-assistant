"""Loads prompt templates from ``banking_ai/prompts`` (prompts live outside the Python logic)."""
from __future__ import annotations

from pathlib import Path


class PromptLibrary:
    """Reads ``*.txt`` prompt files once and composes the system prompts."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._cache: dict[str, str] = {}

    def get(self, name: str) -> str:
        if name not in self._cache:
            path = self.directory / f"{name}.txt"
            if not path.exists():
                raise FileNotFoundError(f"Prompt template not found: {path}")
            self._cache[name] = path.read_text(encoding="utf-8").strip()
        return self._cache[name]

    def agent_system(self) -> str:
        """System prompt for the tool-selecting agent: role + safety + tool policy."""
        return "\n\n".join(self.get(n) for n in ("banking_assistant", "safety_grounding", "tool_usage"))

    def response_system(self, with_kb: bool) -> str:
        """System prompt for the final answer: role + safety (+ RAG answering rules when passages exist)."""
        names = ["banking_assistant", "safety_grounding"] + (["rag_answer"] if with_kb else [])
        return "\n\n".join(self.get(n) for n in names)
