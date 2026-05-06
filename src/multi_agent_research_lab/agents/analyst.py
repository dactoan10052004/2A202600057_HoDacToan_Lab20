"""Analyst agent — turns research notes into structured insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        logger.info("Analyst: analysing research notes.")
        try:
            system = (
                "You are an expert analyst. Given research notes, produce a structured analysis:\n"
                "1. KEY CLAIMS — the most important findings (bullet list)\n"
                "2. VIEWPOINTS — different perspectives or approaches identified\n"
                "3. WEAK EVIDENCE — claims that lack strong support or have conflicting sources\n"
                "4. GAPS — what is still unclear or missing\n"
                f"Target audience: {state.request.audience}."
            )
            user = f"Query: {state.request.query}\n\nResearch Notes:\n{state.research_notes}"
            response = self._llm.complete(system, user)

            state.analysis_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.ANALYST,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("analyst.done", {"cost_usd": response.cost_usd})
            logger.info("Analyst: analysis complete.")
        except Exception as exc:
            logger.error("Analyst failed: %s — using fallback analysis.", exc)
            state.analysis_notes = "Analysis unavailable due to an error."
            state.errors.append(f"analyst_error: {exc}")
            state.add_trace_event("analyst.error", {"error": str(exc)})
        return state
