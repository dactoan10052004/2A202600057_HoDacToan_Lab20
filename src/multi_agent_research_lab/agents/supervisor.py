"""Supervisor / router agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    _MIN_QUERY_WORDS = 3

    def run(self, state: ResearchState) -> ResearchState:
        """Route to the next agent or terminate the pipeline."""
        settings = get_settings()

        if state.iteration >= settings.max_iterations:
            logger.warning("Max iterations (%d) reached — stopping.", settings.max_iterations)
            next_route = "done"
        elif len(state.request.query.split()) < self._MIN_QUERY_WORDS:
            logger.warning("Query too short for research pipeline — skipping.")
            state.final_answer = (
                "Query is too short or too simple for the multi-agent research pipeline. "
                "Please provide a more detailed research question (at least 4 words)."
            )
            state.errors.append("query_too_short")
            next_route = "done"
        elif state.research_notes is None:
            next_route = "researcher"
        elif state.analysis_notes is None:
            next_route = "analyst"
        elif state.final_answer is None:
            next_route = "writer"
        else:
            next_route = "done"

        logger.info("Supervisor -> %s (iteration %d)", next_route, state.iteration)
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor.route",
            {"next": next_route, "iteration": state.iteration},
        )
        return state
