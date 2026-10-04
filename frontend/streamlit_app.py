"""Streamlit chat UI for the AI Banking Assistant.  Run: streamlit run frontend/streamlit_app.py"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("ASSISTANT_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 45

st.set_page_config(page_title="Banking Assistant", page_icon="🏦", layout="wide")
from backend_bootstrap import ensure_backend
ensure_backend()

@st.cache_data(ttl=60, show_spinner=False)
def load_customers() -> list[dict]:
    response = requests.get(f"{API_URL}/demo/customers", timeout=10)
    response.raise_for_status()
    return response.json()["customers"]


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.session_id = None


def call_api(customer_id: str, text: str, debug: bool) -> dict:
    payload = {"customer_id": customer_id, "message": text, "session_id": st.session_state.session_id, "debug": debug}
    response = requests.post(f"{API_URL}/chat", json=payload, timeout=TIMEOUT)
    if response.status_code == 403:
        raise RuntimeError("This session belongs to another customer. Start a new conversation.")
    response.raise_for_status()
    return response.json()


def show_debug(info: dict) -> None:
    with st.expander("🛠 Execution summary", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Intent", info["intent"] or "-")
        cols[1].metric("Confidence", f"{info['confidence']:.2f}")
        cols[2].metric("Latency", f"{info['latency_ms']:.0f} ms")
        cols[3].metric("Answer by", info["response_source"])
        st.caption(f"Request ID `{info['request_id']}` · intent model `{info['intent_backend']}` · mode `{info['mode']}` · grounding ok: {info['grounding_ok']}")
        st.write("**Entities**", info["entities"] or "none")
        st.write("**Tools called**", [f"{t['tool']} ({'ok' if t['ok'] else t['error']}, {t['latency_ms']} ms, {t['source']})" for t in info["tools"]] or "none")
        if info["rag_documents"]:
            st.write("**RAG documents**", [f"{d['title']} / {d['section']} ({d['score']:.2f})" for d in info["rag_documents"]])
        if info["notes"]:
            st.write("**Notes**", info["notes"])


st.title("🏦 Banking Assistant")
st.caption("Demo with synthetic data only. Ask about balances, transactions, loans, EMIs, charges and more.")

with st.sidebar:
    st.header("Session")
    try:
        customers = load_customers()
    except requests.RequestException:
        st.error(f"Cannot reach the assistant API at {API_URL}. Start it with:\n\n`uvicorn banking_ai.api.assistant_app:app --port 8000`")
        st.stop()
    labels = {c["customer_id"]: f"{c['name']} ({c['customer_id']})" for c in customers}
    selected = st.selectbox("Customer", list(labels), format_func=labels.get, key="customer_id")
    show_debug_panel = st.toggle("Developer panel", value=True)
    st.button("🗑 Clear conversation", on_click=reset_conversation, use_container_width=True)
    st.markdown("**Try:**\n- What is my balance?\n- Show my last 5 transactions\n- Can I get a personal loan of 5 lakh for 5 years?\n- Calculate EMI for 5 lakh for 5 years at 10%\n- What is the minimum balance for a savings account?\n- Compare personal loan and home loan")

if "messages" not in st.session_state:
    reset_conversation()
if st.session_state.get("active_customer") != selected:  # switching customer never carries context over
    reset_conversation()
    st.session_state.active_customer = selected

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("debug") and show_debug_panel:
            show_debug(message["debug"])

if prompt := st.chat_input("Type your banking question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = call_api(selected, prompt, debug=True)
                st.session_state.session_id = result["session_id"]
                answer, debug = result["response"], result.get("debug")
            except requests.Timeout:
                answer, debug = "The request took too long. Please try again.", None
            except requests.RequestException:
                answer, debug = "Something went wrong while contacting the assistant. Please try again.", None
            except RuntimeError as exc:
                answer, debug = str(exc), None
        st.markdown(answer)
        if debug and show_debug_panel:
            show_debug(debug)
    st.session_state.messages.append({"role": "assistant", "content": answer, "debug": debug})
