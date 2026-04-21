"""
retriever.py
Loads the persisted ChromaDB vectorstore and returns a LangChain retriever.
"""

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

ROOT = Path(__file__).parent.parent
CHROMA_DIR = ROOT / "data" / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "arxiv_papers"
TOP_K = 5


def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",  # maximal marginal relevance — reduces redundancy
        search_kwargs={"k": TOP_K, "fetch_k": 20},
    )
    return retriever
