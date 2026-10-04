import json

import pytest

from banking_ai.rag.documents import chunk_document, load_documents
from banking_ai.rag.service import RagService, extract_relevant_lines


@pytest.fixture(scope="module")
def rag(settings):
    return RagService(settings).ensure_index()


def test_documents_load_and_chunk_by_section(settings):
    docs = load_documents(settings.kb_dir)
    assert len(docs) >= 10
    chunks = chunk_document(next(d for d in docs if d.doc_id == "minimum_balance"), 600, 80)
    assert chunks and all(c.section and c.text.startswith(c.title) for c in chunks)


def test_long_sections_are_split_with_overlap(settings):
    doc = load_documents(settings.kb_dir)[0]
    doc.text = "# T\n## S\n" + "\n".join(f"- line number {i} with some padding words here" for i in range(30))
    chunks = chunk_document(doc, chunk_size=200, overlap=60)
    assert len(chunks) > 3
    assert chunks[0].lines[-1] in chunks[1].lines  # overlap keeps context


@pytest.mark.parametrize("query,doc", [
    ("What is the minimum balance for a savings account?", "minimum_balance"),
    ("what is the daily UPI limit", "transaction_limits"),
    ("personal loan processing fee", "loan_products"),
    ("what credit score is needed for a home loan", "loan_eligibility_rules"),
    ("fixed deposit rate for 1 year", "interest_rates"),
])
def test_retrieval_finds_the_right_document(rag, query, doc):
    result = rag.retrieve(query)
    assert result.grounded and doc in [c.doc_id for c in result.chunks[:3]]


def test_irrelevant_queries_are_rejected(rag):
    for query in ("tell me a joke", "who won the cricket match", "capital of france"):
        assert not rag.retrieve(query).grounded


def test_extractive_lines_contain_the_answer(rag):
    query = "What is the minimum balance for a savings account?"
    lines = " ".join(line for line, _ in extract_relevant_lines(query, rag.retrieve(query)))
    assert "5,000" in lines


def test_index_is_persisted_and_reused(settings, rag):
    manifest = json.loads((settings.vector_store_dir / "manifest.json").read_text())
    reloaded = RagService(settings).ensure_index()
    assert manifest["chunks"] == len(reloaded.chunks) and reloaded.retrieve("minimum balance").grounded
