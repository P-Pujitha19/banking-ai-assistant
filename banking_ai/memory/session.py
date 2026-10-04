"""Per-session conversation memory with strict customer isolation."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

LOAN_SLOT_KEYS = ("loan_type", "loan_amount", "tenure_months", "interest_rate", "monthly_income")


class SessionOwnershipError(Exception):
    """A session id was presented by a different customer than the one that created it."""


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    intent: Optional[str] = None


@dataclass
class Session:
    """Conversation state: recent turns plus 'slots' (facts remembered across turns, e.g. the loan being discussed)."""

    session_id: str
    customer_id: str
    created_at: datetime
    last_active: datetime
    turns: list[Turn] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    last_intent: Optional[str] = None
    max_turns: int = 10

    def add_turn(self, role: str, content: str, intent: Optional[str] = None) -> None:
        self.turns.append(Turn(role, content, intent))
        self.turns = self.turns[-self.max_turns * 2:]
        self.last_active = datetime.now(timezone.utc)

    def history(self, last_n: int = 6) -> list[tuple[str, str]]:
        """Recent ``(role, content)`` pairs (oldest first)."""
        return [(t.role, t.content) for t in self.turns[-last_n * 2:]]

    def remember_loan_slots(self, entities: dict[str, Any]) -> None:
        for key in LOAN_SLOT_KEYS:
            if entities.get(key) is not None:
                self.slots[key] = entities[key]

    def clear_loan_slots(self) -> None:
        for key in LOAN_SLOT_KEYS:
            self.slots.pop(key, None)


class SessionStore:
    """Thread-safe in-memory store. Production would use Redis with the same interface."""

    def __init__(self, ttl_minutes: int = 30, max_turns: int = 10) -> None:
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_turns = max_turns
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: Optional[str], customer_id: str) -> Session:
        """Return the caller's session. Unknown/expired ids get a NEW server-generated id (no client-chosen ids)."""
        with self._lock:
            self._purge_locked()
            existing = self._sessions.get(session_id) if session_id else None
            if existing is not None:
                if existing.customer_id != customer_id:
                    raise SessionOwnershipError("This session belongs to a different customer.")
                return existing
            now = datetime.now(timezone.utc)
            session = Session(uuid.uuid4().hex, customer_id, now, now, max_turns=self._max_turns)
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str, customer_id: str) -> Optional[Session]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.customer_id != customer_id:
                raise SessionOwnershipError("This session belongs to a different customer.")
            return session

    def clear(self, session_id: str, customer_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            if session.customer_id != customer_id:
                raise SessionOwnershipError("This session belongs to a different customer.")
            del self._sessions[session_id]
            return True

    def _purge_locked(self) -> None:
        cutoff = datetime.now(timezone.utc) - self._ttl
        for sid in [sid for sid, s in self._sessions.items() if s.last_active < cutoff]:
            del self._sessions[sid]

    def __len__(self) -> int:
        return len(self._sessions)
