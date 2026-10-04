#!/usr/bin/env python
"""Draw docs/architecture.png (requires matplotlib)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

fig, ax = plt.subplots(figsize=(15, 10.2))
ax.set_xlim(0, 150); ax.set_ylim(0, 102); ax.axis("off")


def box(x, y, w, h, title, sub="", color="#e8f0fe", edge="#1a56db"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h * 0.64 if sub else y + h / 2, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0b2a6b")
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=8.2, color="#333")


def arrow(a, b, text="", dx=1.0, dy=0.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.5, color="#444"))
    if text:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, text, fontsize=7.8, color="#444")


ax.text(75, 100, "AI-Powered Banking Customer Query Assistant - Architecture", ha="center", fontsize=15, fontweight="bold")
box(4, 86, 34, 9, "1. Streamlit UI", "chat · customer picker · debug panel", "#fff4e0", "#d97706")
box(46, 86, 44, 9, "2. Assistant API (FastAPI)", "POST /chat · sessions · request-id · validation", "#fff4e0", "#d97706")
box(98, 86, 48, 9, "13. Guardrails (security/)", "input guard · injection block · masking · grounding check", "#fde8e8", "#c81e1e")
arrow((38, 90.5), (46, 90.5)); arrow((90, 90.5), (98, 90.5))
ax.add_patch(FancyBboxPatch((4, 22), 142, 60, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#f7fafc", ec="#64748b", lw=1.2, ls="--"))
ax.text(7, 79, "BankingAssistant orchestrator (agents/orchestrator.py)", fontsize=10.5, fontweight="bold", color="#334155")
box(7, 62, 30, 11, "3. Query understanding", "normalise · placeholders", "#e6fffa", "#0f766e")
box(41, 62, 32, 11, "4. TensorFlow intent model", "Embedding→BiGRU→Softmax\nintent + confidence", "#e6fffa", "#0f766e")
box(77, 62, 32, 11, "5. Entity extraction", "rules + optional LLM\n(Pydantic schema)", "#e6fffa", "#0f766e")
box(113, 62, 30, 11, "12. Memory / context", "session slots · history\nper-customer isolation", "#f3e8ff", "#7e22ce")
arrow((37, 67.5), (41, 67.5)); arrow((73, 67.5), (77, 67.5)); arrow((109, 67.5), (113, 67.5))
box(41, 44, 68, 12, "6. Agentic orchestration", "ContextResolver → LLM tool-calling agent (LangChain)  |  demo: declarative planner", "#e0f2fe", "#0369a1")
arrow((75, 62), (75, 56), "intent + entities + context")
box(113, 44, 30, 12, "11. LLM (Generative AI)", "OpenAI / Gemini via LangChain\nfinal grounded answer", "#fce7f3", "#be185d")
arrow((109, 48), (113, 48)); ax.text(108.2, 52.2, "facts", fontsize=7.8, color="#444", ha="right")
box(7, 26, 30, 14, "7. Banking tools", "balance · txns · eligibility\ncharges · explain\n(customer-bound executor)", "#ecfccb", "#4d7c0f")
box(41, 26, 32, 14, "9. RAG service", "load→chunk→embed→retrieve\nsearch_knowledge_base", "#ecfccb", "#4d7c0f")
box(77, 26, 32, 14, "Calculator tool", "calculate_emi\n(also exposed by the bank API)", "#ecfccb", "#4d7c0f")
arrow((52, 44), (22, 40)); arrow((65, 44), (57, 40)); arrow((90, 44), (93, 40))
box(113, 26, 30, 14, "14. Logging & Evaluation", "JSON logs · request id\nscripts/evaluate.py · pytest", "#f1f5f9", "#475569")
box(7, 4, 34, 13, "8. Mock Banking API (FastAPI)", "customers · accounts · txns · loans\nrates · eligibility · EMI  (synthetic)", "#fff7ed", "#c2410c")
box(50, 4, 34, 13, "10. Vector database", "ChromaDB (NumPy fallback)\nembeddings of KB chunks", "#fff7ed", "#c2410c")
box(93, 4, 50, 13, "Knowledge base + synthetic data files", "12 markdown documents · 24 customers · 1,479 transactions", "#fff7ed", "#c2410c")
arrow((22, 26), (22, 17)); arrow((60, 26), (65, 17)); arrow((93, 10.5), (84, 10.5), "indexed from", -1.0, 1.2)
arrow((64, 86), (64, 82))
Path("docs").mkdir(exist_ok=True)
fig.savefig("docs/architecture.png", dpi=130, bbox_inches="tight", facecolor="white")
