#!/usr/bin/env python
"""Evaluate the assistant. Computes REAL metrics from the evaluation sets - nothing is hard-coded.

    python scripts/evaluate.py [--bank-mode inprocess|http] [--output evaluation_reports/latest.json]

Metrics: intent accuracy & confidence, entity accuracy, tool-selection accuracy, RAG retrieval (hit@k, MRR,
negative rejection), end-to-end resolution accuracy (multi-turn included) and latency vs the project targets
(>= 80% resolution accuracy, response time < 15 s).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from banking_ai.agents.orchestrator import BankingAssistant  # noqa: E402
from banking_ai.bank.calculators import calculate_emi, format_inr  # noqa: E402
from banking_ai.bank.client import InProcessBankClient  # noqa: E402
from banking_ai.bank.data_store import load_bank_data  # noqa: E402
from banking_ai.bank.service import BankService  # noqa: E402
from banking_ai.config.logging_config import configure_logging  # noqa: E402
from banking_ai.config.settings import get_settings  # noqa: E402
from banking_ai.evaluation import metrics as m  # noqa: E402

TARGET_ACCURACY, TARGET_LATENCY_S = 0.80, 15.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the banking assistant")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_reports" / "latest.json")
    parser.add_argument("--bank-mode", choices=["inprocess", "http"], default="inprocess")
    args = parser.parse_args()

    settings = get_settings().with_overrides(bank_api_mode=args.bank_mode, log_file=None, log_level="WARNING")
    configure_logging("WARNING", None, True)
    service = BankService(load_bank_data(settings.bank_data_dir, settings.bank_as_of_date or None))
    assistant = BankingAssistant.from_settings(settings, bank=InProcessBankClient(service) if args.bank_mode == "inprocess" else None)

    def resolver(kind: str, a: list[str]) -> str:
        if kind == "avail":
            return format_inr(service.get_balance(a[0])["accounts"][0]["available_balance"])
        if kind in {"txn_desc", "txn_id"}:
            txn = service.list_transactions(a[0], limit=int(a[1]) + 1)["transactions"][int(a[1])]
            return txn["description"] if kind == "txn_desc" else txn["transaction_id"]
        if kind == "emi":
            return format_inr(calculate_emi(float(a[0]), float(a[1]), int(a[2])), decimals=True)
        raise ValueError(kind)

    data_dir = ROOT / "data" / "evaluation"
    singles = json.loads((data_dir / "eval_queries.json").read_text(encoding="utf-8"))
    conversations = json.loads((data_dir / "eval_conversations.json").read_text(encoding="utf-8"))
    rag_eval = json.loads((data_dir / "rag_eval.json").read_text(encoding="utf-8"))

    rows, latencies = [], []
    for item in singles:
        query = m.resolve_placeholders(item["query"], resolver)
        started = time.perf_counter()
        reply = assistant.handle(item["customer_id"], query)
        latencies.append(time.perf_counter() - started)
        rows.append(score_turn(item, reply, resolver))

    conv_rows, conv_ok = [], 0
    for conv in conversations:
        session_id, turn_ok = None, []
        for turn in conv["turns"]:
            started = time.perf_counter()
            reply = assistant.handle(conv["customer_id"], turn["query"], session_id)
            latencies.append(time.perf_counter() - started)
            session_id = reply.session_id
            row = score_turn({**turn, "customer_id": conv["customer_id"], "expected_entities": {}, "expected_status": None}, reply, resolver)
            turn_ok.append(row["resolved"])
            conv_rows.append({"conversation": conv["name"], **row})
        conv_ok += all(turn_ok)

    rag = evaluate_rag(assistant, rag_eval)
    report = summarise(rows, conv_rows, conv_ok, len(conversations), rag, latencies, assistant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": report, "single_turn": rows, "multi_turn": conv_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(report, rows + conv_rows)
    print(f"\nFull report: {args.output}")
    return 0 if report["targets"]["resolution_accuracy_met"] and report["targets"]["latency_met"] else 1


def score_turn(item: dict, reply, resolver) -> dict:
    called = [t["tool"] for t in reply.tools]
    expected_intent = item.get("expected_intent")
    needles = [m.resolve_placeholders(n, resolver) for n in item.get("expect_contains", [])]
    matched, total = m.entity_scores(item.get("expected_entities", {}), reply.entities)
    status_ok = item.get("expected_status") in (None, reply.status) or (item.get("expected_status") == "success" and reply.status == "clarification" and not item.get("expected_tools"))
    intent_ok = expected_intent is None or reply.intent == expected_intent
    tools_ok = m.tool_sets_equal(item.get("expected_tools", []), called)
    content_ok = m.contains_all(reply.response, needles)
    tools_failed = any(not t["ok"] for t in reply.tools)
    return {
        "query": item["query"], "intent_expected": expected_intent, "intent_pred": reply.intent, "confidence": round(reply.confidence, 3),
        "intent_ok": intent_ok, "entities_matched": matched, "entities_expected": total, "tools_expected": item.get("expected_tools", []),
        "tools_called": called, "tools_ok": tools_ok, "status": reply.status, "content_ok": content_ok, "missing": [n for n in needles if n.lower() not in reply.response.lower()],
        "resolved": bool(status_ok and tools_ok and content_ok and not tools_failed), "grounding_ok": reply.grounding_ok,
    }


def evaluate_rag(assistant: BankingAssistant, rag_eval: dict) -> dict:
    hit1 = hit3 = 0
    rr = []
    for item in rag_eval["positive"]:
        docs = [c.doc_id for c in assistant.rag.retrieve(item["query"], k=3).chunks]
        hit1 += bool(docs[:1] and docs[0] in item["expected_docs"])
        hit3 += any(d in item["expected_docs"] for d in docs[:3])
        rr.append(m.reciprocal_rank(item["expected_docs"], docs))
    rejected = sum(not assistant.rag.retrieve(q).grounded for q in rag_eval["negative"])
    n = len(rag_eval["positive"])
    return {"queries": n, "hit_at_1": m.safe_ratio(hit1, n), "hit_at_3": m.safe_ratio(hit3, n), "mrr": round(statistics.mean(rr), 4),
            "negative_queries": len(rag_eval["negative"]), "negative_rejection_rate": m.safe_ratio(rejected, len(rag_eval["negative"]))}


def summarise(rows, conv_rows, conv_ok, n_conv, rag, latencies, assistant) -> dict:
    intent_rows = [r for r in rows + conv_rows if r["intent_expected"]]
    correct = [r for r in intent_rows if r["intent_ok"]]
    wrong = [r for r in intent_rows if not r["intent_ok"]]
    ent_matched, ent_total = sum(r["entities_matched"] for r in rows), sum(r["entities_expected"] for r in rows)
    ent_items = [r for r in rows if r["entities_expected"]]
    all_rows = rows + conv_rows
    resolution = m.safe_ratio(sum(r["resolved"] for r in all_rows), len(all_rows))
    p95 = m.percentile(latencies, 95)
    return {
        "mode": assistant.mode, "intent_backend": assistant.classifier.backend.name, "vector_backend": assistant.rag.backend,
        "queries_evaluated": len(all_rows),
        "intent_accuracy": m.safe_ratio(len(correct), len(intent_rows)),
        "intent_mean_confidence_correct": round(statistics.mean(r["confidence"] for r in correct), 4) if correct else None,
        "intent_mean_confidence_wrong": round(statistics.mean(r["confidence"] for r in wrong), 4) if wrong else None,
        "entity_field_accuracy": m.safe_ratio(ent_matched, ent_total),
        "entity_item_exact_accuracy": m.safe_ratio(sum(r["entities_matched"] == r["entities_expected"] for r in ent_items), len(ent_items)),
        "tool_selection_accuracy": m.safe_ratio(sum(r["tools_ok"] for r in all_rows), len(all_rows)),
        "rag": rag,
        "end_to_end_resolution_accuracy": resolution,
        "multi_turn_conversations_fully_resolved": f"{conv_ok}/{n_conv}",
        "latency_seconds": {"mean": round(statistics.mean(latencies), 4), "p50": round(m.percentile(latencies, 50), 4),
                            "p95": round(p95, 4), "max": round(max(latencies), 4)},
        "targets": {"resolution_accuracy_target": TARGET_ACCURACY, "resolution_accuracy_met": resolution >= TARGET_ACCURACY,
                    "latency_target_seconds": TARGET_LATENCY_S, "latency_met": max(latencies) < TARGET_LATENCY_S},
    }


def print_report(report: dict, rows: list) -> None:
    print("\n=== EVALUATION REPORT ===")
    for key, value in report.items():
        print(f"{key:42} {value}")
    failures = [r for r in rows if not r["resolved"]]
    if failures:
        print(f"\nUnresolved ({len(failures)}):")
        for r in failures:
            print(f"  - {r['query']!r}: intent {r['intent_pred']} (want {r['intent_expected']}), tools {r['tools_called']} (want {r['tools_expected']}), missing {r['missing']}, status {r['status']}")


if __name__ == "__main__":
    raise SystemExit(main())
