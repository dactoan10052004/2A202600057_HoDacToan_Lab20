"""Writer agent — synthesises a final answer with citations."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        logger.info("Writer: composing final answer.")
        try:
            source_refs = "\n".join(
                f"[{i + 1}] {s.title} — {s.url or 'no URL'}"
                for i, s in enumerate(state.sources)
            )
            system = (
                "You are a skilled technical writer. Using the research notes and analysis "
                "below, write a clear, well-structured response (~500 words) to the user's query. "
                "You MUST include inline citations as [N] references for every factual claim, "
                "matching the source list. End with a 'References' section listing all cited sources. "  # noqa: E501
                f"Target audience: {state.request.audience}."
            )
            user = (
                f"Query: {state.request.query}\n\n"
                f"Research Notes:\n{state.research_notes}\n\n"
                f"Analysis:\n{state.analysis_notes}\n\n"
                f"Sources:\n{source_refs}"
            )
            response = self._llm.complete(system, user)

            state.final_answer = response.content
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=response.content,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                    },
                )
            )
            state.add_trace_event("writer.done", {"cost_usd": response.cost_usd})
            logger.info("Writer: final answer written.")
        except Exception as exc:
            logger.error("Writer failed: %s — using fallback answer.", exc)
            state.final_answer = (
                f"Unable to generate a full answer due to an error. "
                f"Research notes are available but synthesis failed.\n\n"
                f"{state.research_notes or ''}"
            )
            state.errors.append(f"writer_error: {exc}")
            state.add_trace_event("writer.error", {"error": str(exc)})
        return state
