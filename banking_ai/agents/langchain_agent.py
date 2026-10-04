"""LLM-driven tool-calling agent built on LangChain (``bind_tools`` + an explicit, bounded agent loop).

Loop: LLM sees question + NLU hints -> returns tool calls -> we execute them (validated, customer-bound) ->
results go back as ToolMessages -> repeat until the LLM stops calling tools (max N steps).
The final natural-language answer is written afterwards by ``LlmResponder`` from the verified results.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from banking_ai.config.logging_config import get_logger
from banking_ai.llm.prompts import PromptLibrary
from banking_ai.tools.langchain_adapter import to_langchain_tools
from banking_ai.tools.toolkit import ToolExecutor

logger = get_logger("agent")


@dataclass
class AgentTrace:
    steps: int
    tool_calls: int
    final_text: str = ""


class LangChainToolAgent:
    def __init__(self, chat_model: Any, prompts: PromptLibrary, max_steps: int = 4) -> None:
        self.chat_model, self.prompts, self.max_steps = chat_model, prompts, max_steps

    def run(self, *, message: str, history: list[tuple[str, str]], hints: Optional[str], executor: ToolExecutor) -> AgentTrace:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        tools = to_langchain_tools(executor)
        tool_map = {t.name: t for t in tools}
        llm = self.chat_model.bind_tools(tools)
        messages: list[Any] = [SystemMessage(content=self.prompts.agent_system())]
        if hints:
            messages.append(SystemMessage(content=hints))
        messages += [HumanMessage(content=c) if r == "user" else AIMessage(content=c) for r, c in history]
        messages.append(HumanMessage(content=message))

        total_calls, steps, final = 0, 0, ""
        for steps in range(1, self.max_steps + 1):
            ai = llm.invoke(messages)
            messages.append(ai)
            calls = getattr(ai, "tool_calls", None) or []
            if not calls:
                final = ai.content if isinstance(ai.content, str) else json.dumps(ai.content)
                break
            for n, call in enumerate(calls):
                total_calls += 1
                tool = tool_map.get(call["name"])
                result = tool.invoke(call.get("args") or {}) if tool else json.dumps({"ok": False, "error": "unknown_tool"})
                messages.append(ToolMessage(content=result, tool_call_id=call.get("id") or f"call_{steps}_{n}", name=call["name"]))
        return AgentTrace(steps=steps, tool_calls=total_calls, final_text=final)


def build_hints(intent: str, confidence: float, entities: dict[str, Any], low_confidence: bool) -> str:
    """Upstream-NLU hints for the agent (TensorFlow intent + extracted entities)."""
    if low_confidence:
        return "NLU hint: the intent classifier was not confident; decide the tools yourself from the customer's words."
    return (f"NLU hints (from the TensorFlow intent classifier and entity extractor; verify against the customer's words): "
            f"intent={intent} (confidence {confidence:.2f}); entities={json.dumps(entities, default=str)}")
