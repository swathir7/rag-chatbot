"""Conversational RAG chain on Groq-hosted Llama 3.3 70B."""
import os
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .vectorstore import VectorIndex

MODEL = "llama-3.3-70b-versatile"

SYSTEM = (
    "You are a helpful assistant answering questions about the user's documents. "
    "Answer ONLY from the provided context. If the answer is not in the context, say "
    "you couldn't find it in the documents. Be concise and cite sources like [file, page]."
)


def get_llm(temperature: float = 0.1) -> ChatGroq:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env or the sidebar.")
    return ChatGroq(model=MODEL, temperature=temperature)


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
    """Rewrite a follow-up into a standalone search query."""
    if not history:
        return question
    msgs = [SystemMessage("Rewrite the user's last message as a standalone search query "
                          "using the chat history. Output only the query.")]
    msgs += _to_messages(history[-6:]) + [HumanMessage(question)]
    return llm.invoke(msgs).content.strip()


def answer(index: VectorIndex, question: str, history: List[Dict], k: int = 4):
    """Returns (answer_stream, source_docs, retrieval_ms)."""
    llm = get_llm()
    query = condense_question(llm, question, history)
    docs, ms = index.search(query, k=k)
    context = _fmt_context(docs)
    msgs = [SystemMessage(f"{SYSTEM}\n\nContext:\n{context}")]
    msgs += _to_messages(history[-6:]) + [HumanMessage(question)]
    return llm.stream(msgs), docs, ms
