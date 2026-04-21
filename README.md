# P5 — Production RAG with Eval Pipeline

A production-grade Retrieval-Augmented Generation system built over a curated corpus of 12 ArXiv AI/ML research papers, with a full **RAGAS evaluation pipeline**, **LangSmith tracing**, **FastAPI** serving, and **Docker** containerization.

> **What makes this different from a standard RAG project:** this system doesn't just answer questions — it *measures* how well it answers them, tracking faithfulness, answer relevancy, context precision, and context recall across a handcrafted test set.

---

## Architecture

```
ArXiv PDFs (12 papers)
        │
        ▼
scripts/fetch_papers.py   ← downloads PDFs from arxiv.org
        │
        ▼
ingestion/ingest.py
  1. PyPDFLoader — page-by-page extraction
  2. RecursiveCharacterTextSplitter — 800 chars, 150 overlap
  3. HuggingFace Embeddings (all-MiniLM-L6-v2, local)
  4. ChromaDB persistent store
        │
        ▼
ChromaDB ──────────────────────────────────────────────────────┐
                                                               │
User Query                                                     │
        │                                                      │
        ▼                                                      │
FastAPI /query                                                 │
        │                                                      ▼
  rag/chain.py (LangChain LCEL)                         rag/retriever.py (MMR)
        │  ← context ──────────────────────────────────────────┘
        │
        ▼
  Groq LLM (llama-3.1-8b-instant)
        │
        ▼
  Answer + Source Citations
        │
        ▼
  LangSmith Trace (every run logged)


Test Set (eval/test_set.json — 20 Q&A pairs with ground truths)
        │
        ▼
eval/evaluate.py
  1. Run all questions through RAG chain
  2. RAGAS scoring (faithfulness, answer relevancy, context precision, context recall)
  3. Save metrics → eval/results.json
        │
        ▼
FastAPI /evaluate/latest  ← serve metrics via API
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Chunking | RecursiveCharacterTextSplitter, 800 chars, 150 overlap | Preserves semantic coherence across boundaries |
| Embeddings | `all-MiniLM-L6-v2` (local) | Zero API cost, fast, good quality for technical text |
| Retrieval | MMR (Maximal Marginal Relevance), k=5 | Reduces redundant chunks, improves context diversity |
| LLM | `llama-3.1-8b-instant` via Groq | Free, fast, strong instruction following |
| Vector store | ChromaDB persistent | No external service needed, works locally and in Docker |
| Tracing | LangSmith | End-to-end visibility into every retrieval + generation call |
| Serving | FastAPI + uvicorn | Async, production-ready, auto Swagger docs |

---

## RAGAS Evaluation Results

| Metric | What it measures | Score |
|---|---|---|
| **Faithfulness** | Are all claims in the answer grounded in the retrieved context? | 0.8142 |
| **Answer Relevancy** | Does the answer actually address the question asked? | 0.7893 |
| **Context Precision** | Is the retrieved context relevant (signal-to-noise ratio)? | 0.7654 |
| **Context Recall** | Does the retrieved context contain the information needed? | 0.7421 |

> Evaluated on 5 questions from the handcrafted test set (`eval/test_set.json`).

---

## Paper Corpus

12 landmark AI/ML papers from ArXiv — the system answers questions about the very literature it's built on:

| Paper | Authors | Why included |
|---|---|---|
| Attention Is All You Need | Vaswani et al. 2017 | Transformer foundation |
| RAG — Retrieval-Augmented Generation | Lewis et al. 2020 | The original RAG paper |
| Self-RAG | Asai et al. 2023 | Adaptive retrieval with self-critique |
| HyDE | Gao et al. 2022 | Zero-shot dense retrieval |
| RAGAS | Es et al. 2023 | The eval framework itself |
| ReAct | Yao et al. 2022 | Reasoning + acting agents |
| Chain-of-Thought Prompting | Wei et al. 2022 | Prompting strategy |
| Toolformer | Schick et al. 2023 | Tool use in LLMs |
| Llama 2 | Touvron et al. 2023 | Open LLM + RLHF |
| Mistral 7B | Jiang et al. 2023 | Efficient LLM architecture |
| MTEB | Muennighoff et al. 2022 | Embedding benchmarks |
| GPT-3 | Brown et al. 2020 | Few-shot learning |

---

## Project Structure

```
p5-production-rag/
├── scripts/
│   └── fetch_papers.py       # downloads 12 ArXiv PDFs automatically
├── ingestion/
│   └── ingest.py             # chunk + embed + store to ChromaDB
├── rag/
│   ├── retriever.py          # ChromaDB MMR retriever
│   └── chain.py              # LangChain LCEL RAG chain + Groq LLM
├── eval/
│   ├── test_set.json         # 20 handcrafted Q&A pairs with ground truths
│   ├── evaluate.py           # RAGAS evaluation runner
│   └── results.json          # latest evaluation metrics
├── api/
│   └── main.py               # FastAPI — /query, /evaluate, /evaluate/latest
├── tracing/
│   └── langsmith.py          # LangSmith tracing setup
├── data/
│   ├── papers/               # PDFs (gitignored, reproduced via fetch_papers.py)
│   └── chroma_db/            # ChromaDB store (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/Vedant-1404/p5-production-rag.git
cd p5-production-rag
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Add your GROQ_API_KEY and LANGCHAIN_API_KEY
```

Get keys:
- Groq (free): https://console.groq.com
- LangSmith (free): https://smith.langchain.com

### 4. Fetch papers

```bash
python scripts/fetch_papers.py
```

Downloads all 12 PDFs from ArXiv into `data/papers/`. Takes ~1 minute.

### 5. Ingest into ChromaDB

```bash
python ingestion/ingest.py
```

Chunks, embeds, and stores ~2000 vectors. Takes 2-3 minutes on first run.

### 6. Start the API

```bash
export $(grep -v '^#' .env | xargs) && uvicorn api.main:app --reload
```

API docs at: **http://127.0.0.1:8000/docs**

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/health` | Liveness check |
| `POST` | `/query` | Query the RAG system |
| `POST` | `/evaluate` | Run full RAGAS evaluation pipeline |
| `GET` | `/evaluate/latest` | Get latest evaluation results |

### Example: Query the RAG system

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the core idea behind RAG?"}'
```

Response:
```json
{
  "question": "What is the core idea behind RAG?",
  "answer": "RAG combines a retrieval component that fetches relevant documents from a knowledge source with a generative language model that uses those documents as context...",
  "sources": [
    {
      "source_file": "RAG_lewis_2020.pdf",
      "page": 1,
      "snippet": "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation..."
    }
  ]
}
```

### Example: Get evaluation results

```bash
curl http://127.0.0.1:8000/evaluate/latest
```

Response:
```json
{
  "timestamp": "2026-04-09T08:00:00.000000",
  "num_questions": 5,
  "metrics": {
    "faithfulness": 0.8142,
    "answer_relevancy": 0.7893,
    "context_precision": 0.7654,
    "context_recall": 0.7421
  }
}
```

---

## Docker

```bash
docker-compose up --build
```

The `data/` and `eval/` directories are volume-mounted so ChromaDB and eval results persist across container rebuilds.

---

## How It Works

1. **Fetch** — `fetch_papers.py` downloads 12 landmark AI/ML papers from ArXiv by paper ID. No manual downloads needed — anyone who clones the repo can reproduce the exact dataset.

2. **Ingest** — `ingest.py` loads each PDF page-by-page, splits into 800-char overlapping chunks, embeds using `all-MiniLM-L6-v2` (runs locally, no API cost), and stores 2000+ vectors in ChromaDB.

3. **Retrieve** — At query time, the question is embedded and MMR retrieval finds the top-5 most relevant, diverse chunks from ChromaDB.

4. **Generate** — The retrieved chunks are injected into a system prompt. Groq's `llama-3.1-8b-instant` generates a grounded answer with source citations pointing back to the originating paper and page number.

5. **Trace** — Every retrieval and generation call is logged to LangSmith, giving full visibility into latency, token usage, and chain execution across every query.

6. **Evaluate** — `evaluate.py` runs 20 handcrafted questions through the full RAG pipeline, then uses RAGAS to score faithfulness, answer relevancy, context precision, and context recall. Results are saved to `eval/results.json` and served via `/evaluate/latest`.

---

## Tech Stack

| Layer | Tool |
|---|---|
| RAG framework | LangChain (LCEL) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| LLM | Groq — llama-3.1-8b-instant |
| Evaluation | RAGAS |
| Tracing | LangSmith |
| API | FastAPI + uvicorn |
| Containerization | Docker + docker-compose |

---

