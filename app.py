"""Streamlit chat UI.

    streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

from docpilot import create_pilot
from docpilot.ingest import chunk_file

st.set_page_config(page_title="DocPilot", page_icon="🧭", layout="wide")


@st.cache_resource
def pilot():
    return create_pilot()


dp, settings = pilot()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Add PDF, Markdown or text", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if uploads and st.button("Index files"):
        with tempfile.TemporaryDirectory() as tmp:
            for u in uploads:
                path = Path(tmp) / u.name
                path.write_bytes(u.getvalue())
                dp.store.delete_source(u.name)
                n = dp.store.add(chunk_file(path, settings.chunk_size, settings.chunk_overlap))
                st.success(f"{u.name}: {n} chunks")
    st.caption(f"{dp.store.count()} chunks indexed")
    for s in dp.store.sources():
        st.write(f"• {s}")
    show_trace = st.toggle("Show agent steps", value=True)

st.title("🧭 DocPilot")
st.caption("Ask questions about your documents. Answers cite their sources, and the agent checks itself.")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about your documents"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            r = dp.ask(question)
        st.markdown(r["answer"])
        if not r["grounded"]:
            st.warning("The self-check could not verify every claim against the sources.")
        if r["sources"]:
            with st.expander(f"Sources ({len(r['sources'])})"):
                for s in r["sources"]:
                    page = f", p.{s['page']}" if s["page"] else ""
                    st.markdown(f"**[{s['id']}] {s['source']}{page}** (score {s['score']})")
                    st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))
        if show_trace:
            st.caption(" → ".join(r["trace"]) + f"  ·  {r['latency_s']}s")
    st.session_state.history.append({"role": "assistant", "content": r["answer"]})
