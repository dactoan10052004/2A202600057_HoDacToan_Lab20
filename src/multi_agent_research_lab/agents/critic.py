"""Critic agent — fact-checks the final answer against sources."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append critique findings."""
        logger.info("Critic: fact-checking final answer.")
        source_snippets = "\n\n".join(
            f"[{i + 1}] {s.title}: {s.snippet[:300]}"
            for i, s in enumerate(state.sources)
        )
        system = (
            "You are a rigorous fact-checking agent. Review the draft answer against the "
            "original sources and produce:\n"
            "1. VERIFIED — claims well-supported by sources\n"
            "2. UNVERIFIED — claims not found in sources (potential hallucination)\n"
            "3. CITATION GAPS — inline [N] references that are missing or incorrect\n"
            "4. VERDICT — overall quality score 0-10 with one-line justification\n"
            "Be concise and specific."
        )
        user = (
            f"Draft Answer:\n{state.final_answer}\n\n"
            f"Original Sources:\n{source_snippets}"
        )
        response = self._llm.complete(system, user)

        verdict_line = next(
            (line for line in response.content.splitlines() if "VERDICT" in line.upper()),
            "",
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={
                    "verdict": verdict_line,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "critic.done",
            {"verdict": verdict_line, "cost_usd": response.cost_usd},
        )
        logger.info("Critic: %s", verdict_line)
        return state
