# 🧠 DocuMind AI — Hybrid RAG & LLM Comparison Studio

A Generative AI academic project for question answering over your own PDFs, DOCX, TXT and Markdown documents. Features a Streamlit dashboard, local ChromaDB vector storage, multilingual-capable LLM providers (Ollama or Gemini), hybrid semantic/BM25 retrieval, source-page evidence, comparison of baseline LLM vs RAG, and session analytics.

## Architecture

```text
Documents → Text extraction → Overlapping chunks → MiniLM embeddings → ChromaDB
                                                               ↓
Question → Semantic vector search + BM25 → Reciprocal rank fusion → Top chunks
                                                               ↓
                                    Grounded prompt → Ollama / Gemini → Answer + citations
```

## Quick start (Windows PowerShell)

1. Install Python 3.11 or 3.12. Open a terminal in this project folder.
2. Create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Choose a model provider:
   - **Local Ollama:** Install Ollama from its official site, run `ollama pull llama3.2`, and ensure Ollama is running. Choose Ollama / `llama3.2` in the sidebar.
   - **Gemini:** Copy `.env.example` to `.env`, add your Gemini API key, and select Gemini in the sidebar. Do not commit `.env`.
4. Launch:

```powershell
streamlit run app.py
```

5. Open **Knowledge Base**, upload documents, and click **Index selected documents**. Then use **AI Assistant** or **RAG vs LLM**.

The first run downloads the embedding model. A local model requires adequate RAM. Gemini needs internet access and may have quotas or charges.

## Key capabilities

- PDF page-level references (DOCX/TXT/MD use page 1 as a placeholder because original pagination is unavailable)
- Persistent local embeddings; content-hash-based deduplication for identical files
- Hybrid vector + BM25 retrieval with reciprocal rank fusion
- Evidence panel showing retrieved chunks and scores
- Standalone LLM vs RAG side-by-side comparison
- Query latency and retrieval analytics, exportable as CSV
- Prompt instructions to abstain when evidence is missing

## Important limitations

- Scanned PDFs need OCR, which is not included.
- Retrieval uses dense search + BM25; a neural reranker is not included.
- Scores are ranking scores, **not** confidence probabilities.
- No RAGAS or ground-truth evaluation is implemented; use a labeled test set for formal metrics.
- DOCX page numbers are not recoverable from paragraphs alone.
- A question that refers to prior chat without restating its subject may retrieve suboptimal evidence.
- The local knowledge base is intended for a single-user classroom demo, not public multi-tenant deployment.
- Uploaded documents may contain sensitive information; Gemini sends retrieved excerpts to the provider. Use Ollama for local processing.
- Generated answers can still be incorrect. Verify source passages.

## Run tests

```powershell
pytest -q
```

## Suggested demo

1. Upload two course PDFs.
2. Ask a question requiring evidence from both documents.
3. Open the evidence panel and inspect pages.
4. Compare the same question in the standalone and RAG modes.
5. Review latency in Analytics and export metrics.

## Research directions

Add a cross-encoder reranker, evaluation with reference answers (faithfulness, context recall), multi-user authentication, OCR, and conversational query rewriting.

## Security

Never commit API keys. `.env`, local data, and ChromaDB are excluded via `.gitignore`.
