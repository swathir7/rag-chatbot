"""Measure real FAISS retrieval latency on your own corpus.
Usage: python benchmark.py sample_docs  (or any folder)"""
import statistics
import sys

from rag.loader import load_directory
from rag.vectorstore import VectorIndex

folder = sys.argv[1] if len(sys.argv) > 1 else "sample_docs"
idx = VectorIndex()
print(f"Chunks indexed: {idx.add_documents(load_directory(folder))}")
queries = ["What is this document about?", "key findings", "summary of main points",
           "who is responsible", "what are the risks"] * 10
idx.search("warmup")
times = [idx.search(q)[1] for q in queries]
print(f"Retrieval latency over {len(times)} queries (ms): "
      f"mean={statistics.mean(times):.1f} p50={statistics.median(times):.1f} "
      f"p95={sorted(times)[int(len(times) * .95) - 1]:.1f}")
