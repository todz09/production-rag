"""
evaluate.py
Runs RAGAS evaluation over the test set.
Produces a metrics report and saves results to eval/results.json.

Run: python eval/evaluate.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from ragas.run_config import RunConfig
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag.retriever import get_retriever
from rag.chain import build_rag_chain, format_docs
from tracing.langsmith import init_tracing

TEST_SET_PATH = ROOT / "eval" / "test_set.json"
RESULTS_PATH = ROOT / "eval" / "results.json"


def build_eval_dataset(test_set: list) -> Dataset:
    """
    For each question in the test set:
    - retrieve context chunks
    - generate an answer
    - collect into RAGAS dataset format
    """
    retriever = get_retriever()
    chain, _ = build_rag_chain()

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"\n[eval] Running inference on {len(test_set)} questions...\n")

    for i, item in enumerate(test_set):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i+1}/{len(test_set)}] {question[:70]}...")

        docs = retriever.invoke(question)
        answer = chain.invoke(question)
        context_texts = [doc.page_content for doc in docs]

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def run_ragas(dataset: Dataset) -> dict:
    print("\n[eval] Running RAGAS metrics...")

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig

    eval_llm = LangchainLLMWrapper(
       ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    )
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False,
        run_config=RunConfig(
            max_workers=1,
            max_wait=180,
            timeout=120,
        ),
    )

    return result

def save_results(result, dataset: Dataset):
    def safe_float(val):
        try:
            if val is None:
                return 0.0
            return round(float(val), 4)
        except:
            return 0.0

    if result is None:
        metrics = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }
    else:
        metrics = {
            "faithfulness": safe_float(result["faithfulness"]),
            "answer_relevancy": safe_float(result["answer_relevancy"]),
            "context_precision": safe_float(result["context_precision"]),
            "context_recall": safe_float(result["context_recall"]),
        }

    scores = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_questions": len(dataset),
        "metrics": metrics,
    }

    RESULTS_PATH.write_text(json.dumps(scores, indent=2))
    return scores

def main():
    print("=" * 50)
    print("P5 — RAGAS Evaluation Pipeline")
    print("=" * 50)

    init_tracing(project_name="p5-production-rag-eval")

    test_set = json.loads(TEST_SET_PATH.read_text())[:5]

    dataset = build_eval_dataset(test_set)
    result = run_ragas(dataset)
    scores = save_results(result, dataset)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in scores["metrics"].items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<22} {score:.4f}  {bar}")
    print("=" * 50)
    print(f"\n[eval] Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
