"""Conversation-aware query resolution: merges NLU output with session memory (follow-ups, slot filling)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from banking_ai.memory.session import LOAN_SLOT_KEYS, Session
from banking_ai.ml.intent_classifier import IntentPrediction
from banking_ai.nlp.entities import ExtractedEntities

LOAN_INTENTS = {"loan_eligibility", "loan_information", "interest_rate", "emi_calculation", "loan_comparison"}
SMALL_TALK = {"greeting", "goodbye", "help"}
FOLLOWUP_FIELDS = ("loan_amount", "amount", "tenure_months", "interest_rate", "loan_type", "monthly_income")


@dataclass
class ResolvedQuery:
    intent: str
    confidence: float
    entities: ExtractedEntities
    notes: list[str] = field(default_factory=list)
    low_confidence: bool = False
    original_intent: Optional[str] = None


class ContextResolver:
    """Turns a (possibly elliptical) message into a complete request using the session's slots."""

    def __init__(self, threshold: float = 0.60) -> None:
        self.threshold = threshold

    def resolve(self, prediction: IntentPrediction, entities: ExtractedEntities, session: Session) -> ResolvedQuery:
        intent, confidence, notes = prediction.intent, prediction.confidence, []
        original = intent
        has_followup_data = any(getattr(entities, f) is not None for f in FOLLOWUP_FIELDS)

        # "I need 5 lakh" / "what about 10 years?" -> continue the previous loan topic
        followup = False
        if (intent == "fallback" or confidence < self.threshold) and has_followup_data and session.last_intent in LOAN_INTENTS:
            intent, followup = session.last_intent, True
            notes = notes + ["Understood as a follow-up to the previous loan question."]

        data = entities.model_dump()
        if intent in LOAN_INTENTS:
            if data.get("loan_amount") is None and data.get("amount") is not None and intent in {"loan_eligibility", "emi_calculation"}:
                data["loan_amount"] = data["amount"]
            allowed = LOAN_SLOT_KEYS if intent in {"loan_eligibility", "emi_calculation"} else ("loan_type",)
            for key in allowed:
                if data.get(key) is None and session.slots.get(key) is not None and not (key == "loan_type" and data.get("loan_types")):
                    data[key] = session.slots[key]
                    notes.append(f"Using {key.replace('_', ' ')} from earlier in the conversation.")
        resolved = ExtractedEntities(**data)
        low = not followup and intent != "fallback" and intent not in SMALL_TALK and confidence < self.threshold
        return ResolvedQuery(intent, confidence, resolved, notes, low_confidence=low, original_intent=original)

    @staticmethod
    def update_memory(session: Session, resolved: ResolvedQuery) -> None:
        """Remember loan slots while the loan topic is active; forget them when the topic changes."""
        if resolved.intent in LOAN_INTENTS:
            session.remember_loan_slots(resolved.entities.model_dump())
        elif resolved.intent not in SMALL_TALK and resolved.intent != "fallback":
            session.clear_loan_slots()
        if resolved.intent not in SMALL_TALK and resolved.intent != "fallback":
            session.last_intent = resolved.intent
