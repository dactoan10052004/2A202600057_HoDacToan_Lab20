"""Benchmark runner for single-agent vs multi-agent comparison."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def _estimate_quality(state: ResearchState) -> float:
    """Score 0–10 based on answer length and citation coverage."""
    if not state.final_answer:
        return 0.0
    words = len(state.final_answer.split())
    citations = len(re.findall(r"\[\d+\]", state.final_answer))
    word_score = min(words / 500 * 5.0, 5.0)
    citation_score = min(citations * 0.5, 5.0)
    return round(word_score + citation_score, 1)


def _total_cost(state: ResearchState) -> float | None:
    """Sum cost_usd across all agent results."""
    total = 0.0
    found = False
    for r in state.agent_results:
        c = r.metadata.get("cost_usd")
        if c is not None:
            total += float(c)
            found = True
    return round(total, 6) if found else None


def _citation_coverage(state: ResearchState) -> str:
    """Fraction of retrieved sources cited in the final answer."""
    if not state.sources or not state.final_answer:
        return "N/A"
    cited = len(re.findall(r"\[\d+\]", state.final_answer))
    return f"{cited}/{len(state.sources)}"


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality, and citation coverage for one run."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    quality = _estimate_quality(state)
    cost = _total_cost(state)
    coverage = _citation_coverage(state)
    errors = len(state.errors)

    notes = f"citations={coverage} errors={errors}"
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 2),
        estimated_cost_usd=cost,
        quality_score=quality,
        notes=notes,
    )
    return state, metrics
