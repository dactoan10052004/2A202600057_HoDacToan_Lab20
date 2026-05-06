"""LangGraph multi-agent workflow."""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import langfuse_trace

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()
        self._critic = CriticAgent()

    def build(self) -> Any:
        """Create and compile a LangGraph graph.

        Flow: supervisor -> {researcher|analyst|writer|done}
              researcher/analyst loop back to supervisor.
              writer -> critic -> END.
        """

        def supervisor_node(state: ResearchState) -> dict[str, Any]:
            return self._supervisor.run(state).model_dump()

        def researcher_node(state: ResearchState) -> dict[str, Any]:
            return self._researcher.run(state).model_dump()

        def analyst_node(state: ResearchState) -> dict[str, Any]:
            return self._analyst.run(state).model_dump()

        def writer_node(state: ResearchState) -> dict[str, Any]:
            return self._writer.run(state).model_dump()

        def critic_node(state: ResearchState) -> dict[str, Any]:
            return self._critic.run(state).model_dump()

        def route_from_supervisor(state: ResearchState) -> str:
            last = state.route_history[-1] if state.route_history else "done"
            return END if last == "done" else last

        builder = StateGraph(ResearchState)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("researcher", researcher_node)
        builder.add_node("analyst", analyst_node)
        builder.add_node("writer", writer_node)
        builder.add_node("critic", critic_node)

        builder.set_entry_point("supervisor")
        builder.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END,
            },
        )
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "critic")
        builder.add_edge("critic", END)

        return builder.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the compiled graph and return the final state."""
        compiled = self.build()
        with langfuse_trace("multi-agent-workflow"):
            result: dict[str, Any] = compiled.invoke(state.model_dump())
        return ResearchState(**result)
