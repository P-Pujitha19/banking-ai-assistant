"""Wraps the toolkit as LangChain ``StructuredTool`` objects so a chat model can tool-call them."""
from __future__ import annotations

from typing import Any

from banking_ai.tools.toolkit import ToolExecutor


def to_langchain_tools(executor: ToolExecutor) -> list[Any]:
    """Create one ``StructuredTool`` per banking tool; every call goes through the :class:`ToolExecutor`."""
    from langchain_core.tools import StructuredTool

    tools = []
    for name, spec in executor.specs.items():
        def _run(_name: str = name, **kwargs: Any) -> str:
            return executor.execute_json(_name, **kwargs)

        tools.append(StructuredTool.from_function(func=_run, name=name, description=spec.description, args_schema=spec.args_schema))
    return tools
