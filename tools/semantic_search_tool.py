"""Dense semantic vector search tool with relevance score passthrough."""

import logging
from langchain_core.tools import tool
from tools.vector_store import build_or_get_vectorstore

logger = logging.getLogger("doc-agent.tools.semantic")


@tool
def semantic_docs_search(query: str) -> str:
    """
    Search documentation semantically using dense vector similarity when exact
    keywords or badge names are uncertain. Returns ranked results with relevance
    scores for confidence grounding.

    Args:
        query: Natural language search query about certifications or documentation

    Returns:
        Formatted results with relevance scores and badge metadata.
    """
    try:
        vectorstore = build_or_get_vectorstore()
        results = vectorstore.similarity_search_with_relevance_scores(query, k=5)

        if not results:
            return "No matching documentation specifications found in vector index."

        formatted = []
        for i, (doc, score) in enumerate(results, start=1):
            badge_name = doc.metadata.get("badge_name", "Unknown")
            group_idx = doc.metadata.get("group_index", -1)
            group_label = f"Alternative {group_idx + 1}" if group_idx >= 0 else "General"
            formatted.append(
                f"[{i}] (relevance: {score:.3f}) Badge: {badge_name} | {group_label}\n"
                f"{doc.page_content}"
            )

        logger.info(f"Semantic search returned {len(results)} results for: {query[:60]}")
        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return f"Error during semantic search: {str(e)}"