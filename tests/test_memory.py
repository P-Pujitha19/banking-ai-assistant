import pytest

from banking_ai.memory.session import SessionOwnershipError, SessionStore


def test_new_session_gets_server_generated_id():
    store = SessionStore()
    s = store.get_or_create(None, "CUST1001")
    assert len(s.session_id) == 32 and store.get_or_create(s.session_id, "CUST1001") is s


def test_unknown_client_chosen_id_is_not_adopted():
    store = SessionStore()
    s = store.get_or_create("attacker-chosen-id", "CUST1001")
    assert s.session_id != "attacker-chosen-id"


def test_sessions_are_isolated_between_customers():
    store = SessionStore()
    a = store.get_or_create(None, "CUST1001")
    a.add_turn("user", "secret question")
    a.remember_loan_slots({"loan_amount": 500000})
    with pytest.raises(SessionOwnershipError):
        store.get_or_create(a.session_id, "CUST1002")
    b = store.get_or_create(None, "CUST1002")
    assert b.slots == {} and b.turns == []


def test_history_is_bounded_and_slots_can_be_cleared():
    store = SessionStore(max_turns=2)
    s = store.get_or_create(None, "CUST1001")
    for i in range(10):
        s.add_turn("user", f"m{i}")
    assert len(s.turns) == 4
    s.remember_loan_slots({"loan_type": "personal_loan", "loan_amount": 1.0})
    s.clear_loan_slots()
    assert s.slots == {}


def test_clear_requires_ownership():
    store = SessionStore()
    s = store.get_or_create(None, "CUST1001")
    with pytest.raises(SessionOwnershipError):
        store.clear(s.session_id, "CUST1002")
    assert store.clear(s.session_id, "CUST1001") and len(store) == 0
