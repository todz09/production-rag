"""
main.py
FastAPI application exposing two endpoints:
  POST /query     — query the RAG system
  POST /evaluate  — run RAGAS evaluation and return metrics
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag.chain import query as rag_query
from tracing.langsmith import init_tracing

# ── startup ──────────────────────────────────────────────────────────────────

init_tracing(project_name="p5-production-rag")

app = FastAPI(
    title="P5 — Production RAG API",
    description="RAG over ArXiv AI/ML papers with RAGAS evaluation pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── schemas ───────────────────────────────────────────────────────────────────

from typing import Union, Dict, List

class QueryRequest(BaseModel):
    question: str

class SourceDoc(BaseModel):
    source_file: str
    page: Union[str, int]
    snippet: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDoc]

class EvalResponse(BaseModel):
    timestamp: str
    num_questions: int
    metrics: Dict[str, float]

# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "P5 Production RAG API",
        "endpoints": ["/query", "/evaluate", "/health", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = rag_query(req.question)
        return QueryResponse(
            question=req.question,
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvalResponse)
def evaluate_endpoint():
    """
    Runs the full RAGAS evaluation pipeline over the test set.
    This is a long-running operation (~5-10 min depending on test set size).
    Returns the latest metrics once complete.
    """
    results_path = ROOT / "eval" / "results.json"

    try:
        # import here to avoid loading heavy deps on every request
        from eval.evaluate import main as run_eval
        run_eval()

        if not results_path.exists():
            raise HTTPException(status_code=500, detail="Evaluation did not produce results")

        return json.loads(results_path.read_text())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate/latest", response_model=EvalResponse)
def get_latest_eval():
    """Returns the most recent evaluation results without re-running."""
    results_path = ROOT / "eval" / "results.json"

    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation results found. Run POST /evaluate first."
        )

    return json.loads(results_path.read_text())
