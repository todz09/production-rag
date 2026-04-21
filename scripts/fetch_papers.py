"""
fetch_papers.py
Downloads the curated set of ArXiv papers into data/papers/.
Run once before ingestion: python scripts/fetch_papers.py
"""

import os
import time
import requests
from pathlib import Path

PAPERS = [
    {
        "id": "2005.11401",
        "title": "RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "filename": "RAG_lewis_2020.pdf",
    },
    {
        "id": "2310.11511",
        "title": "Self-RAG - Learning to Retrieve Generate and Critique",
        "filename": "SelfRAG_asai_2023.pdf",
    },
    {
        "id": "2212.10560",
        "title": "HyDE - Precise Zero-Shot Dense Retrieval without Relevance Labels",
        "filename": "HyDE_gao_2022.pdf",
    },
    {
        "id": "2309.15217",
        "title": "RAGAS - Automated Evaluation of Retrieval Augmented Generation",
        "filename": "RAGAS_es_2023.pdf",
    },
    {
        "id": "2210.03629",
        "title": "ReAct - Synergizing Reasoning and Acting in Language Models",
        "filename": "ReAct_yao_2022.pdf",
    },
    {
        "id": "2201.11903",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "filename": "CoT_wei_2022.pdf",
    },
    {
        "id": "2302.04761",
        "title": "Toolformer - Language Models Can Teach Themselves to Use Tools",
        "filename": "Toolformer_schick_2023.pdf",
    },
    {
        "id": "2307.09288",
        "title": "Llama 2 - Open Foundation and Fine-Tuned Chat Models",
        "filename": "Llama2_touvron_2023.pdf",
    },
    {
        "id": "2310.06825",
        "title": "Mistral 7B",
        "filename": "Mistral_jiang_2023.pdf",
    },
    {
        "id": "2210.11610",
        "title": "MTEB - Massive Text Embedding Benchmark",
        "filename": "MTEB_muennighoff_2022.pdf",
    },
    {
        "id": "2005.14165",
        "title": "GPT-3 - Language Models are Few-Shot Learners",
        "filename": "GPT3_brown_2020.pdf",
    },
    {
        "id": "1706.03762",
        "title": "Attention Is All You Need",
        "filename": "Attention_vaswani_2017.pdf",
    },
]

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "papers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}.pdf"


def fetch_paper(paper: dict) -> bool:
    url = ARXIV_PDF_URL.format(paper_id=paper["id"])
    out_path = OUTPUT_DIR / paper["filename"]

    if out_path.exists():
        print(f"  [skip] {paper['filename']} already exists")
        return True

    print(f"  [download] {paper['title'][:60]}...")
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        out_path.write_bytes(response.content)
        print(f"  [ok] saved → {paper['filename']} ({len(response.content) // 1024} KB)")
        return True
    except Exception as e:
        print(f"  [error] {paper['filename']}: {e}")
        return False


def main():
    print(f"\nFetching {len(PAPERS)} papers to {OUTPUT_DIR}\n")
    success, failed = 0, []

    for paper in PAPERS:
        ok = fetch_paper(paper)
        if ok:
            success += 1
        else:
            failed.append(paper["filename"])
        time.sleep(1.5)  # be polite to ArXiv

    print(f"\nDone: {success}/{len(PAPERS)} downloaded successfully")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
