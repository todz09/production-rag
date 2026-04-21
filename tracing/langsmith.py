"""
langsmith.py
Activates LangSmith tracing by setting the required environment variables.
Import this at the top of main.py and evaluate.py to enable tracing.
"""

import os


def init_tracing(project_name: str = "p5-production-rag"):
    """
    Call this once at app startup.
    Reads LANGCHAIN_API_KEY from env and activates LangSmith tracing.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")

    if not api_key:
        print("[tracing] LANGCHAIN_API_KEY not set — LangSmith tracing disabled")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    print(f"[tracing] LangSmith enabled → project: {project_name}")
    return True
