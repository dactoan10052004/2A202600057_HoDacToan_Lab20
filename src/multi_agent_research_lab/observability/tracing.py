"""Tracing providers: LangSmith (auto via env vars) + Langfuse (via context manager).

Import this module BEFORE any langgraph/workflow imports so that the
tracing env vars are set before the first graph invocation.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env into os.environ so os.getenv() picks up values from the file.
# pydantic-settings reads .env for Settings but does NOT write to os.environ.
load_dotenv(override=False)


# ── LangSmith ────────────────────────────────────────────────────────────────


def _activate_langsmith() -> bool:
    """Wire LANGSMITH_* and LANGCHAIN_* env vars for LangGraph auto-tracing."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return False
    project = os.getenv("LANGSMITH_PROJECT", "multi-agent-research-lab")
    # New LangSmith SDK vars (v0.2+)
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    # Legacy LangChain vars (still used by langchain-core / langgraph)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    logger.info("LangSmith tracing active -> project '%s'", project)
    return True


# ── Langfuse ─────────────────────────────────────────────────────────────────


def _activate_langfuse() -> bool:
    """Confirm Langfuse credentials are present in os.environ."""
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if not pk or not sk:
        return False
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    os.environ.setdefault("LANGFUSE_HOST", host)
    logger.info("Langfuse tracing active -> %s", host)
    return True


def flush_langfuse() -> None:
    """Flush pending Langfuse traces (call after CLI commands finish)."""
    if not langfuse_active:
        return
    try:
        from langfuse import get_client

        get_client().flush()
        logger.debug("Langfuse traces flushed.")
    except Exception as exc:
        logger.debug("Langfuse flush skipped: %s", exc)


# ── Activate at import time ───────────────────────────────────────────────────

langsmith_active: bool = _activate_langsmith()
langfuse_active: bool = _activate_langfuse()


# ── Langfuse context manager ──────────────────────────────────────────────────


@contextmanager
def langfuse_trace(name: str) -> Iterator[None]:
    """Create a Langfuse agent span if Langfuse is active."""
    if not langfuse_active:
        yield
        return
    try:
        from langfuse import Langfuse

        lf = Langfuse()
        with lf.start_as_current_observation(name=name, as_type="agent"):
            yield
    except Exception as exc:
        logger.debug("Langfuse span skipped: %s", exc)
        yield


# ── Local span context manager ────────────────────────────────────────────────


@contextmanager
def trace_span(
    name: str, attributes: dict[str, Any] | None = None
) -> Iterator[dict[str, Any]]:
    """Lightweight local span; real traces go to LangSmith / Langfuse."""
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
    }
    try:
        yield span
    finally:
        elapsed = perf_counter() - started
        span["duration_seconds"] = elapsed
        logger.debug("span '%s' finished in %.3fs", name, elapsed)
