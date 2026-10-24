"""Chunking, embedding (all-MiniLM-L6-v2) and FAISS indexing."""
import time
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def chunk_documents(docs: List[Document], chunk_size: int = 800, overlap: int = 120) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)


class VectorIndex:
    def __init__(self, embeddings: Optional[HuggingFaceEmbeddings] = None):
        self.embeddings = embeddings or get_embeddings()
        self.store: Optional[FAISS] = None

    def add_documents(self, docs: List[Document]) -> int:
        chunks = chunk_documents(docs)
        if not chunks:
            return 0
        if self.store is None:
            self.store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.store.add_documents(chunks)
        return len(chunks)

    def search(self, query: str, k: int = 4) -> Tuple[List[Document], float]:
        """Returns (docs, retrieval latency in ms)."""
        if self.store is None:
            return [], 0.0
        t0 = time.perf_counter()
        vec = self.embeddings.embed_query(query)
        t1 = time.perf_counter()
        docs = self.store.similarity_search_by_vector(vec, k=k)
        t2 = time.perf_counter()
        print(f"[timing] embed={(t1 - t0) * 1000:.1f} ms  faiss={(t2 - t1) * 1000:.1f} ms", flush=True)
        return docs, (t2 - t0) * 1000

    def save(self, path: str = "faiss_index") -> None:
        if self.store:
            self.store.save_local(path)

    def load(self, path: str = "faiss_index") -> bool:
        if not Path(path).exists():
            return False
        self.store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return True

    @property
    def size(self) -> int:
        return self.store.index.ntotal if self.store else 0
