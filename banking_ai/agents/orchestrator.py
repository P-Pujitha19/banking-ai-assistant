"""BankingAssistant: the end-to-end pipeline that ties every component together.

    guard -> session memory -> intent (TensorFlow) -> entities -> context resolution -> agent/planner -> tools
          -> grounded response (LLM or template) -> grounding check -> memory update -> structured log
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from banking_ai.agents.context import SMALL_TALK, ContextResolver, ResolvedQuery
from banking_ai.agents.planner import plan_for
from banking_ai.bank.client import BankClient, build_bank_client
from banking_ai.config.logging_config import get_logger, log_event, request_id_var
from banking_ai.config.settings import Settings
from banking_ai.llm.prompts import PromptLibrary
from banking_ai.llm.provider import build_chat_model
from banking_ai.llm.templates import SERVICE_ERROR, TemplateResponder
from banking_ai.memory.session import Session, SessionOwnershipError, SessionStore
from banking_ai.ml.intent_classifier import IntentClassifier, IntentPrediction
from banking_ai.nlp.entities import ExtractedEntities, HybridEntityExtractor
from banking_ai.rag.service import RagService
from banking_ai.security.grounding import GroundingValidator
from banking_ai.security.guard import InputGuard, validate_customer_id
from banking_ai.tools.toolkit import BankingToolkit, ToolExecutor, ToolOutcome

logger = get_logger("assistant")


@dataclass
class AssistantReply:
    """Everything the API/UI needs about one handled message."""

    session_id: str
    request_id: str
    response: str
    status: str  # success | clarification | fallback | blocked | error
    intent: str = ""
    confidence: float = 0.0
    intent_backend: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    rag_documents: list[dict[str, Any]] = field(default_factory=list)
    response_source: str = "template"
    grounding_ok: Optional[bool] = None
    notes: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class BankingAssistant:
    """Agentic banking assistant (see docs/ARCHITECTURE in README)."""

    def __init__(self, settings: Settings, classifier: IntentClassifier, extractor: HybridEntityExtractor, rag: RagService,
                 bank: BankClient, sessions: SessionStore, chat_model: Any = None, prompts: Optional[PromptLibrary] = None) -> None:
        self.settings, self.classifier, self.extractor, self.rag, self.bank, self.sessions = settings, classifier, extractor, rag, bank, sessions
        self.chat_model = chat_model
        self.prompts = prompts or PromptLibrary(settings.prompts_dir)
        self.guard = InputGuard(settings.max_message_chars)
        self.resolver = ContextResolver(settings.intent_confidence_threshold)
        self.templates = TemplateResponder()
        self.validator = GroundingValidator()
        self._agent = self._responder = None
        if chat_model is not None:
            from banking_ai.agents.langchain_agent import LangChainToolAgent
            from banking_ai.llm.chains import LlmResponder

            self._agent = LangChainToolAgent(chat_model, self.prompts, settings.llm_max_tool_steps)
            self._responder = LlmResponder(chat_model, self.prompts)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_settings(cls, settings: Settings, *, bank: Optional[BankClient] = None, rag: Optional[RagService] = None,
                      classifier: Optional[IntentClassifier] = None, chat_model: Any = "auto") -> "BankingAssistant":
        """Wire every component from configuration."""
        prompts = PromptLibrary(settings.prompts_dir)
        model = build_chat_model(settings) if chat_model == "auto" else chat_model
        llm_extractor = None
        if model is not None and settings.use_llm_entity_extraction:
            from banking_ai.llm.chains import LlmEntityExtractor

            llm_extractor = LlmEntityExtractor(model, prompts)
        return cls(settings, classifier or IntentClassifier.load(settings), HybridEntityExtractor(llm_extractor=llm_extractor),
                   rag or RagService(settings).ensure_index(), bank or build_bank_client(settings),
                   SessionStore(settings.session_ttl_minutes, settings.max_history_turns), model, prompts)

    @property
    def mode(self) -> str:
        return "llm" if self.chat_model is not None else "demo"

    # ------------------------------------------------------------------ public API
    def handle(self, customer_id: str, message: str, session_id: Optional[str] = None) -> AssistantReply:
        """Process one user message. Raises ``SessionOwnershipError`` if the session belongs to someone else."""
        request_id = uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            if not validate_customer_id(customer_id):
                return AssistantReply("", request_id, "Invalid customer id.", "blocked", notes=["invalid_customer_id"])
            session = self.sessions.get_or_create(session_id, customer_id)
            reply = self._process(session, message, request_id)
        except Exception as exc:
            if isinstance(exc, SessionOwnershipError):
                raise
            logger.exception("unhandled error while processing request")
            reply = AssistantReply(session_id or "", request_id, SERVICE_ERROR, "error", notes=["internal_error"])
        finally:
            reply_latency = (time.perf_counter() - started) * 1000
            request_id_var.reset(token)
        reply.latency_ms = round(reply_latency, 1)
        log_event(logger, "assistant_request", request_id=request_id, session=reply.session_id[:8],
                  customer=hashlib.sha256(customer_id.encode()).hexdigest()[:10], intent=reply.intent,
                  confidence=round(reply.confidence, 3), tools=[t["tool"] for t in reply.tools], status=reply.status,
                  success=reply.status in {"success", "clarification", "fallback"}, latency_ms=reply.latency_ms,
                  response_source=reply.response_source, grounding_ok=reply.grounding_ok, mode=self.mode)
        return reply

    # ------------------------------------------------------------------ pipeline
    def _process(self, session: Session, message: str, request_id: str) -> AssistantReply:
        guard = self.guard.inspect(message, session.customer_id)
        if not guard.allowed:
            return AssistantReply(session.session_id, request_id, guard.user_message or "", "blocked", notes=[f"blocked:{guard.category}"])
        text = guard.cleaned

        prediction: IntentPrediction = self.classifier.predict(text)
        entities = self.extractor.extract(text, prediction.intent)
        resolved = self.resolver.resolve(prediction, entities, session)
        # entities may be richer now (memory fill) - give the LLM extractor a second chance with the resolved intent
        if resolved.intent != prediction.intent:
            resolved.entities = self.extractor.extract(text, resolved.intent).merged_with(resolved.entities)

        reply = AssistantReply(session.session_id, request_id, "", "success", resolved.intent, prediction.confidence,
                               prediction.backend, resolved.entities.non_empty(), notes=list(resolved.notes))
        history = session.history()

        if resolved.intent in SMALL_TALK or resolved.intent == "fallback":
            reply.response = self.templates.compose(resolved.intent, resolved.entities, [], text)
            reply.status = "fallback" if resolved.intent == "fallback" else "success"
        elif resolved.low_confidence and self._agent is None:
            reply.response = self._clarify(prediction)
            reply.status = "clarification"
        else:
            self._answer_with_tools(reply, resolved, text, session, history)

        session.add_turn("user", text, resolved.intent)
        session.add_turn("assistant", reply.response, resolved.intent)
        self.resolver.update_memory(session, resolved)
        return reply

    def _answer_with_tools(self, reply: AssistantReply, resolved: ResolvedQuery, text: str, session: Session,
                           history: list[tuple[str, str]]) -> None:
        toolkit = BankingToolkit(session.customer_id, self.bank, self.rag)
        executor = ToolExecutor(toolkit.specs())
        plan = plan_for(resolved.intent, resolved.entities, text)

        if self._agent is not None:  # LLM decides which tools to call
            try:
                from banking_ai.agents.langchain_agent import build_hints

                hints = build_hints(resolved.intent, reply.confidence, resolved.entities.non_empty(), resolved.low_confidence)
                self._agent.run(message=text, history=history, hints=hints, executor=executor)
                reply.notes.append("tools_selected_by_llm")
            except Exception:
                logger.exception("LLM agent failed; falling back to deterministic planner")
                executor.outcomes.clear()
        if not executor.outcomes:  # demo mode, LLM failure, or LLM called no tool
            if plan.clarification:
                reply.response, reply.status = plan.clarification, "clarification"
                return
            for call in plan.calls:
                executor.execute(call.name, call.args)
            if self._agent is not None and plan.calls:
                reply.notes.append("tools_selected_by_planner_fallback")
        reply.notes += plan.notes
        outcomes = executor.outcomes
        reply.tools = [o.to_public() for o in outcomes]
        reply.sources = sorted({o.source_type for o in outcomes if o.ok})
        kb = next((o for o in outcomes if o.name == "search_knowledge_base" and o.ok), None)
        if kb:
            reply.rag_documents = [{"title": p["title"], "section": p["section"], "score": p["score"]} for p in kb.data["passages"]]

        reply.response, reply.response_source, reply.grounding_ok = self._compose(resolved, outcomes, text, history, plan.notes)
        if not outcomes:
            reply.status = "fallback"

    def _compose(self, resolved: ResolvedQuery, outcomes: list[ToolOutcome], text: str, history: list[tuple[str, str]],
                 notes: list[str]) -> tuple[str, str, Optional[bool]]:
        user_notes = list(notes)  # only genuine assumptions (e.g. "assumed a personal loan"); debug notes stay out of the answer
        template_answer = self.templates.compose(resolved.intent, resolved.entities, outcomes, text, user_notes)
        successful = [o for o in outcomes if o.ok]
        if self._responder is None or not successful:
            return template_answer, "template", None
        facts = {o.name: o.data for o in successful}
        kb_context = ""
        kb = next((o for o in successful if o.name == "search_knowledge_base"), None)
        if kb and kb.data["grounded"]:
            kb_context = "\n\n".join(f"[{i}] {p['title']} / {p['section']}\n" + "\n".join(f"- {l}" for l in p["lines"])
                                     for i, p in enumerate(kb.data["passages"], 1))
        try:
            answer = self._responder.respond(question=text, intent=resolved.intent, facts=facts, kb_context=kb_context,
                                             notes=user_notes, history=history)
        except Exception:
            logger.exception("LLM response generation failed; using template answer")
            return template_answer, "template", None
        allowed_extra = [v for v in resolved.entities.model_dump().values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        check = self.validator.check(answer, facts, extra_allowed=allowed_extra)
        if not answer or not check.ok:
            logger.warning("grounding check failed (%d ungrounded figures); using template answer", len(check.ungrounded))
            return template_answer, "template", False
        return answer, "llm", True

    @staticmethod
    def _clarify(prediction: IntentPrediction) -> str:
        options = ", ".join(i.replace("_", " ") for i, _ in prediction.top_k[:2])
        return ("I want to make sure I get this right - I'm not fully sure what you're asking "
                f"(it could be about: {options}). Could you rephrase, for example *\"What is my balance?\"* or *\"Show my last 5 transactions\"*?")

    def clear_session(self, session_id: str, customer_id: str) -> bool:
        return self.sessions.clear(session_id, customer_id)
