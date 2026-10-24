# 📚 LLM-Powered RAG Chatbot

Conversational search over your documents using **LangChain, FAISS, Groq GPT-OSS-120B and Streamlit**.

## Features
- Retrieval-Augmented Generation grounded in your documents, with source citations
- Semantic search with `all-MiniLM-L6-v2` embeddings, recursive chunking, FAISS vector index
- Ingestion for 10 formats: PDF, DOCX, TXT, MD, CSV, HTML, PPTX, XLSX, JSON
- Conversational memory (follow-up questions are rewritten into standalone queries)
- Persistent index (saved to `faiss_index/`), streaming answers, per-query retrieval latency display

## Architecture
```
Files -> loaders -> chunking -> MiniLM embeddings -> FAISS
Question -> condense (history) -> FAISS top-k -> GPT-OSS-120B (Groq) -> answer + sources
```

## Setup
```bash
git clone https://github.com/<your-username>/rag-chatbot.git && cd rag-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # add your free key from https://console.groq.com
streamlit run app.py
```

## Benchmark retrieval latency
```bash
python benchmark.py path/to/your/docs
```

## Project structure
```
app.py              Streamlit UI
rag/loader.py       Multi-format document loaders
rag/vectorstore.py  Chunking, embeddings, FAISS index
rag/chain.py        Conversational RAG chain (Groq Llama 3.3)
benchmark.py        Latency benchmark
```

## License
MIT

## OCR for image-based PDF pages (optional)
Slides and scans often keep their text inside images. If `tesseract` is installed, pages with little extractable text are OCR'd automatically:
```bash
brew install tesseract    # macOS (Linux: apt install tesseract-ocr)
```
Without it, the app still works and skips image-only pages.

## Configuration
Set `GROQ_MODEL` in `.env` to use a different Groq model (default: `openai/gpt-oss-120b`).
