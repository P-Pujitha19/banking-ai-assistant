"""Tool selection (planner) and the LangChain tool-calling loop (with a scripted fake chat model)."""
import json

import pytest

from banking_ai.agents.planner import plan_for
from banking_ai.nlp.entities import RuleEntityExtractor

rules = RuleEntityExtractor()


def tools_for(intent, text):
    return [c.name for c in plan_for(intent, rules.extract(text), text).calls]


@pytest.mark.parametrize("intent,text,expected", [
    ("balance_inquiry", "What is my balance?", ["get_account_balance"]),
    ("transaction_history", "Show my last 5 transactions", ["get_transaction_history"]),
    ("recent_transaction", "What was my last transaction?", ["get_recent_transaction"]),
    ("interest_rate", "What is the interest rate for a personal loan?", ["get_interest_rates"]),
    ("loan_eligibility", "Can I get a personal loan of 5 lakh?", ["check_loan_eligibility"]),
    ("emi_calculation", "Calculate EMI for 5 lakh for 5 years at 10%", ["calculate_emi"]),
    ("account_charges", "What are the charges for my account?", ["get_account_charges"]),
    ("minimum_balance", "minimum balance for a savings account", ["search_knowledge_base"]),
    ("transaction_explanation", "why was money deducted", ["explain_transaction"]),
    ("loan_comparison", "compare personal loan and home loan", ["compare_loan_products"]),
    ("greeting", "hi", []),
])
def test_planner_selects_expected_tools(intent, text, expected):
    assert tools_for(intent, text) == expected


def test_planner_arguments_come_from_entities():
    call = plan_for("emi_calculation", rules.extract("EMI for 5 lakh for 5 years at 10%"), "").calls[0]
    assert call.args == {"principal": 500000.0, "tenure_months": 60, "annual_rate": 10.0}
    history = plan_for("transaction_history", rules.extract("show my last 7 transactions"), "").calls[0]
    assert history.args["limit"] == 7


def test_emi_without_amount_asks_a_clarifying_question():
    plan = plan_for("emi_calculation", rules.extract("calculate my emi"), "calculate my emi")
    assert plan.calls == [] and "loan amount" in plan.clarification


class FakeChatModel:
    """Scripted stand-in for a tool-calling chat model: step 1 requests a tool, step 2 answers."""

    def __init__(self):
        self.step = 0

    def bind_tools(self, tools):
        self.tool_names = [t.name for t in tools]
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.step += 1
        if self.step == 1:
            return AIMessage(content="", tool_calls=[{"name": "get_account_balance", "args": {"customer_id": "CUST1002"}, "id": "call_1"}])
        return AIMessage(content="done")


def test_langchain_agent_loop_executes_tools_through_the_executor(settings, bank_service):
    pytest.importorskip("langchain_core")
    from banking_ai.agents.langchain_agent import LangChainToolAgent
    from banking_ai.bank.client import InProcessBankClient
    from banking_ai.llm.prompts import PromptLibrary
    from banking_ai.rag.service import RagService
    from banking_ai.tools.toolkit import BankingToolkit, ToolExecutor

    executor = ToolExecutor(BankingToolkit("CUST1001", InProcessBankClient(bank_service), RagService(settings).ensure_index()).specs())
    model = FakeChatModel()
    trace = LangChainToolAgent(model, PromptLibrary(settings.prompts_dir), max_steps=3).run(
        message="what is my balance", history=[], hints="hint", executor=executor)
    assert trace.tool_calls == 1 and trace.final_text == "done"
    assert "get_account_balance" in model.tool_names and len(model.tool_names) >= 10
    assert executor.outcomes[0].ok and executor.outcomes[0].data["customer_id"] == "CUST1001"  # the model's CUST1002 was ignored
