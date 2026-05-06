"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics

GuardrailsResults = list[tuple[str, str, float]]


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    guardrails_results: GuardrailsResults | None = None,
) -> str:
    """Render benchmark metrics to a markdown report with analysis."""
    lines = [
        "# Benchmark Report",
        "",
        "## Summary",
        "",
    ]

    # Aggregate stats
    if metrics:
        avg_latency = sum(m.latency_seconds for m in metrics) / len(metrics)
        total_cost = sum(m.estimated_cost_usd or 0 for m in metrics)
        avg_quality = sum(m.quality_score or 0 for m in metrics) / len(metrics)
        lines += [
            f"- Runs: {len(metrics)}",
            f"- Average latency: {avg_latency:.2f}s",
            f"- Total estimated cost: ${total_cost:.6f}",
            f"- Average quality score: {avg_quality:.1f}/10",
            "",
        ]

    lines += [
        "## Per-Run Results",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.6f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f}"
            f" | {cost} | {quality} | {item.notes} |"
        )

    # Comparison analysis (baseline vs multi-agent pairs)
    baseline_runs = [m for m in metrics if "baseline" in m.run_name.lower()]
    multi_runs = [m for m in metrics if "multi" in m.run_name.lower()]
    if baseline_runs and multi_runs:
        b_lat = sum(m.latency_seconds for m in baseline_runs) / len(baseline_runs)
        m_lat = sum(m.latency_seconds for m in multi_runs) / len(multi_runs)
        b_q = sum(m.quality_score or 0 for m in baseline_runs) / len(baseline_runs)
        m_q = sum(m.quality_score or 0 for m in multi_runs) / len(multi_runs)
        lat_delta = ((m_lat - b_lat) / b_lat * 100) if b_lat else 0
        q_delta = m_q - b_q
        lines += [
            "",
            "## Baseline vs Multi-Agent Comparison",
            "",
            "| Metric | Baseline | Multi-Agent | Delta |",
            "|---|---:|---:|---:|",
            f"| Avg latency (s) | {b_lat:.2f} | {m_lat:.2f} | {lat_delta:+.1f}% |",
            f"| Avg quality score | {b_q:.1f} | {m_q:.1f} | {q_delta:+.1f} pts |",
            "",
            "## Interpretation",
            "",
            (
                "The multi-agent pipeline trades higher latency for better quality: "
                "dedicated Researcher, Analyst, and Writer agents each add a focused LLM call, "
                "while the single-agent baseline answers in one shot without structured research."
            ),
        ]

    if guardrails_results:
        lines += [
            "",
            "## Guardrails Adversarial Test",
            "",
            "| Query (truncated) | Result | Latency (s) |",
            "|---|:---:|---:|",
        ]
        for query, status, latency in guardrails_results:
            icon = "🛡 BLOCKED" if status == "BLOCKED" else "⚠ ALLOWED"
            lines.append(f"| {query[:70]} | {icon} | {latency:.4f} |")
        blocked = sum(1 for _, s, _ in guardrails_results if s == "BLOCKED")
        lines += [
            "",
            f"**Block rate**: {blocked}/{len(guardrails_results)} "
            f"({blocked / len(guardrails_results) * 100:.0f}%) adversarial queries blocked.",
        ]

    return "\n".join(lines) + "\n"
