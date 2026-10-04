import pytest

from banking_ai.bank.client import InProcessBankClient
from banking_ai.rag.service import RagService
from banking_ai.tools.toolkit import BankingToolkit, ToolExecutor


@pytest.fixture(scope="module")
def rag(settings):
    return RagService(settings).ensure_index()


def make_executor(customer_id, bank_service, rag):
    return ToolExecutor(BankingToolkit(customer_id, InProcessBankClient(bank_service), rag).specs())


def test_all_required_tools_are_registered(bank_service, rag):
    required = {"get_account_balance", "get_account_details", "get_transaction_history", "get_recent_transaction", "get_loan_products",
                "check_loan_eligibility", "get_interest_rates", "calculate_emi", "get_account_charges", "explain_transaction"}
    assert required <= set(make_executor("CUST1001", bank_service, rag).specs)


def test_balance_tool_returns_real_bank_data(bank_service, rag):
    outcome = make_executor("CUST1001", bank_service, rag).execute("get_account_balance")
    assert outcome.ok and outcome.source_type == "customer_data"
    assert outcome.data["accounts"][0]["balance"] == bank_service.get_balance("CUST1001")["accounts"][0]["balance"]


def test_llm_supplied_customer_id_is_ignored(bank_service, rag):
    """A manipulated model cannot redirect a tool to another customer."""
    mine = make_executor("CUST1001", bank_service, rag).execute("get_account_balance", {"customer_id": "CUST1002"})
    assert mine.ok and mine.data["customer_id"] == "CUST1001"


def test_calculator_tool_and_default_rate(bank_service, rag):
    ex = make_executor("CUST1001", bank_service, rag)
    explicit = ex.execute("calculate_emi", {"principal": 500000, "tenure_months": 60, "annual_rate": 10})
    assert explicit.data["emi"] == 10623.52 and explicit.source_type == "calculation"
    defaulted = ex.execute("calculate_emi", {"principal": 500000, "tenure_months": 60, "loan_type": "home_loan"})
    assert defaulted.data["rate_source"] == "product_indicative_rate" and defaulted.data["annual_rate_pct"] == 8.65


def test_errors_become_safe_outcomes(bank_service, rag):
    ex = make_executor("CUST1001", bank_service, rag)
    assert not ex.execute("does_not_exist").ok
    bad = ex.execute("check_loan_eligibility", {"loan_type": "yacht_loan"})
    assert not bad.ok and "Traceback" not in (bad.error or "")
    assert not make_executor("CUST9999", bank_service, rag).execute("get_account_balance").ok


def test_knowledge_tool_is_labelled_as_rag(bank_service, rag):
    outcome = make_executor("CUST1001", bank_service, rag).execute("search_knowledge_base", {"query": "minimum balance savings account"})
    assert outcome.source_type == "knowledge_base" and outcome.data["grounded"] and outcome.data["passages"]
