import os
import yaml
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
YAML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "alternative_docs.yaml")

_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_or_get_vectorstore() -> Chroma:
    """Creates or loads a persistent Chroma vector store from YAML specifications."""
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)

    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"YAML file not found at: {YAML_PATH}")

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    documents = []
    badges = data.get("badges", []) if isinstance(data, dict) else []

    for item in badges:
        name = item.get("name", "Unknown")
        description = item.get("description", "")
        alternatives = item.get("alternatives", [])
        alt_str = ", ".join([str(a) for a in alternatives]) if isinstance(alternatives, list) else str(alternatives)

        page_content = f"Specification: {name}\nDescription: {description}\nAlternatives: {alt_str}"
        doc = Document(
            page_content=page_content,
            metadata={"name": name, "source": "alternative_docs.yaml"}
        )
        documents.append(doc)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=_embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectorstore