from langchain_core.tools import tool
from tools.vector_store import build_or_get_vectorstore


@tool
def semantic_docs_search(query: str) -> str:
    """Search documentation semantically using dense vector similarity when exact keywords or badge names are uncertain."""
    try:
        vectorstore = build_or_get_vectorstore()
        results = vectorstore.similarity_search(query, k=3)
        if not results:
            return "No matching documentation specifications found in vector index."

        formatted = []
        for i, doc in enumerate(results, start=1):
            formatted.append(f"[{i}] {doc.page_content}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Error during semantic search: {str(e)}"