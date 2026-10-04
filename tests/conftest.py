"""Shared fixtures. Tests run fully offline in demo mode (no LLM key, in-process mock bank)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from banking_ai.agents.orchestrator import BankingAssistant  # noqa: E402
from banking_ai.bank.client import InProcessBankClient  # noqa: E402
from banking_ai.bank.data_store import load_bank_data  # noqa: E402
from banking_ai.bank.service import BankService  # noqa: E402
from banking_ai.config.settings import Settings  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    return Settings.from_env().with_overrides(
        bank_api_mode="inprocess", llm_provider="mock", llm_api_key="", log_file=None, log_level="ERROR",
        vector_store_dir=tmp_path_factory.mktemp("vector_store"), bank_api_key="")


@pytest.fixture(scope="session")
def bank_service(settings):
    return BankService(load_bank_data(settings.bank_data_dir))


@pytest.fixture(scope="session")
def assistant(settings, bank_service):
    return BankingAssistant.from_settings(settings, bank=InProcessBankClient(bank_service))
