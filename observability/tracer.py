import os
import time
from contextlib import contextmanager
from typing import Generator, Optional


def setup_langsmith_tracing() -> bool:
    """
    Check if LangSmith environment variables are configured.
    Normalizes LANGSMITH_* and LANGCHAIN_* environment variables.
    Returns True if tracing is active.
    """
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1") or \
                      os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1")

    if api_key and (tracing_enabled or os.getenv("LANGSMITH_API_KEY")):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
        if project:
            os.environ["LANGCHAIN_PROJECT"] = project
        return True
    return False


def setup_phoenix_tracing() -> bool:
    """
    Attempt to initialize Arize Phoenix if installed.
    Returns True if successfully initialized, False otherwise.
    """
    try:
        import phoenix as px
        from openinference.instrumentation.langchain import LangChainInstrumentor

        px.launch_app()
        LangChainInstrumentor().instrument()
        return True
    except (ImportError, Exception):
        return False


@contextmanager
def measure_latency() -> Generator[dict[str, float], None, None]:
    """Context manager to measure execution latency in milliseconds."""
    timing = {"latency_ms": 0.0}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        timing["latency_ms"] = (time.perf_counter() - start) * 1000.0
