"""
chain.py
Builds the RAG chain using LangChain LCEL + Groq LLM.
LangSmith tracing is activated automatically via env vars.
"""

from operator import itemgetter

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

from rag.retriever import get_retriever

SYSTEM_PROMPT = """You are an expert AI research assistant. Answer the user's question
using ONLY the context retrieved from the research paper corpus below.

Rules:
- Be precise and cite which paper/concept you are drawing from when possible.
- If the context does not contain enough information, say so clearly.
- Do not hallucinate or use knowledge outside the provided context.

Context:
{context}
"""

HUMAN_PROMPT = "{question}"


def format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source_file', 'unknown')}, "
        f"Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def build_rag_chain():
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
    )

    # LCEL chain — clean, traceable, serialisable
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def query(question: str) -> dict:
    """
    Run a question through the RAG chain.
    Returns answer + retrieved source docs.
    """
    chain, retriever = build_rag_chain()

    # retrieve docs separately so we can return them
    docs = retriever.invoke(question)
    answer = chain.invoke(question)

    sources = [
        {
            "source_file": d.metadata.get("source_file", "unknown"),
            "page": d.metadata.get("page", "?"),
            "snippet": d.page_content[:300],
        }
        for d in docs
    ]

    return {"answer": answer, "sources": sources}
