"""Conversational RAG chain on Groq-hosted LLMs (default: GPT-OSS-120B)."""
import os
from itertools import zip_longest
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .vectorstore import VectorIndex

DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM = (
    "You are a helpful assistant answering questions about the user's documents. "
    "Answer ONLY from the provided context. If the answer is not in the context, say "
    "you couldn't find it in the documents. Be concise and cite sources like [file, page]."
)

REWRITE_PROMPT = (
    "Rewrite the user's last message as a detailed search query for a document search engine. "
    "Use the chat history to resolve references like 'it' or 'that'. "
    "Expand it with closely related technical terms and synonyms. Output only the query."
)


def get_llm(temperature: float = 0.1) -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or the sidebar.")
    return ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_MODEL), temperature=temperature)


def _fmt_context(docs) -> str:
    parts = []
    for d in docs:
        m = d.metadata
        loc = next((f"{k} {m[k]}" for k in ("page", "slide", "sheet", "row") if k in m), "")
        parts.append(f"[{m.get('source', '?')}{', ' + loc if loc else ''}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _to_messages(history: List[Dict]):
    return [HumanMessage(h["content"]) if h["role"] == "user" else AIMessage(h["content"]) for h in history]


def condense_question(llm, question: str, history: List[Dict]) -> str:
    """Rewrite the question into a richer, standalone search query."""
    msgs = [SystemMessage(REWRITE_PROMPT)]
    msgs += _to_messages(history[-6:]) + [HumanMessage(question)]
    try:
        out = str(llm.invoke(msgs).content).strip()
    except Exception:
        return question
    return out or question


def answer(index: VectorIndex, question: str, history: List[Dict], k: int = 4):
    """Returns (answer_stream, source_docs, retrieval_ms)."""
    llm = get_llm()
    queries = [question]
    rewritten = condense_question(llm, question, history)
    if rewritten != question:
        queries.append(rewritten)

    results, total_ms = [], 0.0
    for q in queries:
        found, ms = index.search(q, k=k)
        results.append(found)
        total_ms += ms

    seen, docs = set(), []
    for group in zip_longest(*results):
        for d in group:
            if d is None:
                continue
            key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content)
            if key not in seen:
                seen.add(key)
                docs.append(d)
    docs = docs[: 2 * k]

    context = _fmt_context(docs)
    msgs = [SystemMessage(f"{SYSTEM}\n\nContext:\n{context}")]
    msgs += _to_messages(history[-6:]) + [HumanMessage(question)]
    return llm.stream(msgs), docs, total_ms
