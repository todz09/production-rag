"""
ingest.py
Loads PDFs from data/papers/, chunks them, embeds via HuggingFace,
and stores into a persistent ChromaDB collection.

Run: python ingestion/ingest.py
"""

import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

ROOT = Path(__file__).parent.parent
PAPERS_DIR = ROOT / "data" / "papers"
CHROMA_DIR = ROOT / "data" / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "arxiv_papers"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_pdfs(papers_dir: Path) -> list:
    docs = []
    pdf_files = list(papers_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[error] No PDFs found in {papers_dir}")
        print("Run: python scripts/fetch_papers.py")
        sys.exit(1)

    print(f"[ingest] Found {len(pdf_files)} PDFs\n")

    for pdf_path in pdf_files:
        print(f"  Loading {pdf_path.name}...")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            # attach source filename to metadata
            for page in pages:
                page.metadata["source_file"] = pdf_path.name
            docs.extend(pages)
        except Exception as e:
            print(f"  [warn] Could not load {pdf_path.name}: {e}")

    print(f"\n[ingest] Loaded {len(docs)} pages total")
    return docs


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    print(f"[ingest] Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[ingest] Building ChromaDB at {CHROMA_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"[ingest] Stored {vectorstore._collection.count()} vectors")
    return vectorstore


def main():
    print("=" * 50)
    print("P5 — Ingestion Pipeline")
    print("=" * 50 + "\n")

    docs = load_pdfs(PAPERS_DIR)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)

    print("\n[ingest] Done. ChromaDB ready.")


if __name__ == "__main__":
    main()
