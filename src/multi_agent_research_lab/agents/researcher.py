"""Researcher agent — collects sources and writes research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._search = SearchClient()
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        logger.info("Researcher: searching for '%s'", state.request.query[:60])
        try:
            sources = self._search.search(
                state.request.query, max_results=state.request.max_sources
            )
            state.sources.extend(sources)

            source_text = "\n\n".join(
                f"[{i + 1}] {s.title}\n{s.snippet}" for i, s in enumerate(sources)
            )
            system = (
                "You are a research assistant. Summarize the provided sources into concise "
                "research notes. Include key facts, dates, and inline citations as [N] "
                f"references. Target audience: {state.request.audience}."
            )
            user = f"Query: {state.request.query}\n\nSources:\n{source_text}"
            response = self._llm.complete(system, user)

            state.research_notes = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={
                        "sources_found": len(sources),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event(
                "researcher.done",
                {"sources_found": len(sources), "cost_usd": response.cost_usd},
            )
            logger.info("Researcher: %d sources, notes written.", len(sources))
        except Exception as exc:
            logger.error("Researcher failed: %s — using fallback notes.", exc)
            state.research_notes = "Research unavailable due to an error. Proceeding with limited information."  # noqa: E501
            state.errors.append(f"researcher_error: {exc}")
            state.add_trace_event("researcher.error", {"error": str(exc)})
        return state
