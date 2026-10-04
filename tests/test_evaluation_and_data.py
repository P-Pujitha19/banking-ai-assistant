import json
import re

from banking_ai.datagen import catalog
from banking_ai.evaluation import metrics as m


def test_percentile_and_ratio():
    assert m.percentile([1, 2, 3, 4, 5], 95) == 5 and m.percentile([], 95) == 0.0 and m.safe_ratio(1, 4) == 0.25


def test_entity_scoring_tolerates_float_noise():
    assert m.entity_scores({"loan_amount": 500000, "loan_type": "Personal_Loan"}, {"loan_amount": 500000.0, "loan_type": "personal_loan"}) == (2, 2)
    assert m.entity_scores({"a": 1}, {}) == (0, 1)


def test_tool_set_comparison_and_reciprocal_rank():
    assert m.tool_sets_equal(["a", "b"], ["b", "a"]) and not m.tool_sets_equal(["a"], ["a", "b"])
    assert m.reciprocal_rank(["x"], ["y", "x"]) == 0.5


def test_placeholder_resolution():
    assert m.resolve_placeholders("bal {avail:C1}", lambda kind, args: f"<{kind}-{args[0]}>") == "bal <avail-C1>"


def test_knowledge_base_matches_catalogue_numbers(settings):
    """The RAG documents and the mock API must agree on key figures (single source of truth: catalog.py)."""
    kb = "\n".join(p.read_text(encoding="utf-8") for p in settings.kb_dir.glob("*.md"))
    assert f"₹{catalog.ACCOUNT_TYPES['savings']['min_balance']:,}" in kb
    assert "₹25,000" in kb and "₹300" in kb and "₹750" in kb
    for loan in catalog.LOAN_PRODUCTS.values():
        assert f"{loan['rate_min']:.2f}%" in kb and f"{loan['rate_max']:.2f}%" in kb


def test_evaluation_sets_are_well_formed(settings):
    base = settings.bank_data_dir.parent / "evaluation"
    queries = json.loads((base / "eval_queries.json").read_text(encoding="utf-8"))
    assert len(queries) >= 50 and all({"query", "expected_tools", "expected_status"} <= q.keys() for q in queries)
    assert len(json.loads((base / "rag_eval.json").read_text())["positive"]) >= 20


def test_no_real_looking_secrets_in_repo_defaults():
    env = (settings_path := __import__("pathlib").Path(__file__).resolve().parents[1] / ".env.example").read_text()
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", env) and "LLM_API_KEY=" in env
