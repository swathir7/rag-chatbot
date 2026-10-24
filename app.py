import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.chain import answer
from rag.loader import SUPPORTED, load_file
from rag.vectorstore import VectorIndex

load_dotenv()
st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_index() -> VectorIndex:
    idx = VectorIndex()
    idx.load()
    idx.embeddings.embed_query('warmup')
    return idx


index = get_index()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("📚 Documents")
    key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if key:
        os.environ["GROQ_API_KEY"] = key
    files = st.file_uploader("Upload files", accept_multiple_files=True,
                             type=[e.lstrip(".") for e in sorted(SUPPORTED)])
    k = st.slider("Chunks to retrieve (k)", 1, 10, 4)
    if st.button("Index documents", disabled=not files, use_container_width=True):
        total = 0
        with st.spinner("Processing..."):
            for f in files:
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / f.name
                    p.write_bytes(f.getbuffer())
                    try:
                        total += index.add_documents(load_file(p))
                    except Exception as e:
                        st.error(f"{f.name}: {e}")
            index.save()
        st.success(f"Added {total} chunks")
    st.caption(f"Index size: **{index.size}** chunks")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Chat with your documents")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if q := st.chat_input("Ask a question about your documents"):
    if index.size == 0:
        st.warning("Upload and index some documents first.")
        st.stop()
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        try:
            stream, docs, ms = answer(index, q, history, k=k)
            text = st.write_stream(c.content for c in stream)
            st.caption(f"⚡ Retrieval: {ms:.1f} ms")
            with st.expander("Sources"):
                for d in docs:
                    st.markdown(f"**{d.metadata.get('source')}** {({kk: v for kk, v in d.metadata.items() if kk != 'source'})}")
                    st.text(d.page_content[:400])
            st.session_state.messages.append({"role": "assistant", "content": text})
        except Exception as e:
            st.error(str(e))
