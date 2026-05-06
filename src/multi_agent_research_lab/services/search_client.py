"""Search client abstraction for ResearcherAgent."""

import logging

from tavily import TavilyClient

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Tavily-backed search client."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = TavilyClient(api_key=settings.tavily_api_key)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        try:
            response = self._client.search(query=query, max_results=max_results)
            results: list[dict[str, object]] = response.get("results", [])
            docs = [
                SourceDocument(
                    title=str(r.get("title", "Untitled")),
                    url=str(r.get("url")) if r.get("url") else None,
                    snippet=str(r.get("content", "")),
                    metadata={"score": r.get("score", 0)},
                )
                for r in results
            ]
            logger.info("Search: %d results for '%s'", len(docs), query[:60])
            return docs
        except Exception as exc:
            logger.warning("Search failed (%s) — returning empty list", exc)
            return []
