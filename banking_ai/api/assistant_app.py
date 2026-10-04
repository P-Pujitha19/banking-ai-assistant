"""FastAPI service exposing the AI assistant.

Run:  uvicorn banking_ai.api.assistant_app:app --port 8000
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from banking_ai.agents.orchestrator import AssistantReply, BankingAssistant
from banking_ai.api.schemas import ChatRequest, ChatResponse, DebugInfo
from banking_ai.config.logging_config import configure_logging, get_logger
from banking_ai.config.settings import Settings, get_settings
from banking_ai.memory.session import SessionOwnershipError

logger = get_logger("assistant_api")


def create_assistant_app(settings: Optional[Settings] = None, assistant: Optional[BankingAssistant] = None) -> FastAPI:
    """Application factory; pass a pre-built ``assistant`` in tests."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_file, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started = time.perf_counter()
        app.state.assistant = assistant or BankingAssistant.from_settings(settings)
        logger.info("assistant ready in %.1fs (mode=%s)", time.perf_counter() - started, app.state.assistant.mode)
        yield

    app = FastAPI(title="AI Banking Customer Query Assistant", version="1.0.0", lifespan=lifespan,
                  description="Intent (TensorFlow) -> entities -> agent -> tools / RAG -> grounded response.")

    @app.exception_handler(SessionOwnershipError)
    async def ownership_handler(_: Request, exc: SessionOwnershipError) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": "Session does not belong to this customer."})

    @app.get("/health", tags=["system"])
    def health(request: Request) -> dict:
        a: BankingAssistant = request.app.state.assistant
        return {"status": "ok", "mode": a.mode, "intent_backend": a.classifier.backend.name, "vector_backend": a.rag.backend,
                "chunks": len(a.rag.chunks)}

    @app.post("/chat", response_model=ChatResponse, tags=["assistant"])
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        a: BankingAssistant = request.app.state.assistant
        reply: AssistantReply = a.handle(body.customer_id, body.message, body.session_id)
        debug = None
        if body.debug and settings.demo_mode:
            debug = DebugInfo(request_id=reply.request_id, intent=reply.intent, confidence=round(reply.confidence, 4),
                              intent_backend=reply.intent_backend, entities=reply.entities, tools=reply.tools,
                              sources=reply.sources, rag_documents=reply.rag_documents, response_source=reply.response_source,
                              grounding_ok=reply.grounding_ok, notes=reply.notes, latency_ms=reply.latency_ms, mode=a.mode)
        return ChatResponse(session_id=reply.session_id, request_id=reply.request_id, response=reply.response,
                            status=reply.status, latency_ms=reply.latency_ms, debug=debug)

    @app.delete("/sessions/{session_id}", tags=["assistant"])
    def clear_session(session_id: str, request: Request, customer_id: str = Query(pattern=r"^CUST\d{4}$")) -> dict:
        cleared = request.app.state.assistant.clear_session(session_id, customer_id)
        if not cleared:
            raise HTTPException(status_code=404, detail="Session not found.")
        return {"cleared": True}

    @app.get("/demo/customers", tags=["demo"])
    def demo_customers(request: Request) -> dict:
        """Customer picker for the demo UI. Disabled when DEMO_MODE=false (real systems use authentication)."""
        if not settings.demo_mode:
            raise HTTPException(status_code=404, detail="Not available.")
        bank = request.app.state.assistant.bank
        from banking_ai.bank.client import HttpBankClient, InProcessBankClient  # noqa: F401

        service = getattr(bank, "_service", None)
        names = ({c: v["full_name"] for c, v in service.data.customers.items()} if service else _names_via_http(bank))
        return {"customers": [{"customer_id": c, "name": n} for c, n in sorted(names.items())]}

    return app


def _names_via_http(bank) -> dict[str, str]:
    names = {}
    for n in range(1001, 1025):
        try:
            names[f"CUST{n}"] = bank.get_customer(f"CUST{n}")["full_name"]
        except Exception:
            continue
    return names


app = create_assistant_app()
