"""Persistent ChromaDB vector store with granular per-alternative-group chunking."""

import os
import hashlib
import logging

import yaml
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import settings

logger = logging.getLogger("doc-agent.tools.vectorstore")

_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

CHROMA_DIR = str(settings.chroma_dir)
YAML_PATH = str(settings.yaml_path)
HASH_FILE = os.path.join(CHROMA_DIR, ".yaml_hash")


def _compute_yaml_hash() -> str:
    """Compute SHA-256 hash of the YAML file for change detection."""
    with open(YAML_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _needs_rebuild() -> bool:
    """Check if the vector store needs rebuilding based on YAML content hash."""
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        return True
    if not os.path.exists(HASH_FILE):
        return True
    try:
        with open(HASH_FILE, "r") as f:
            stored_hash = f.read().strip()
        return stored_hash != _compute_yaml_hash()
    except Exception:
        return True


def _save_hash(yaml_hash: str) -> None:
    """Persist the YAML hash after a successful index build."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(HASH_FILE, "w") as f:
        f.write(yaml_hash)


def _build_documents(data: dict) -> list[Document]:
    """Build granular Document chunks — one per alternative group for fine-grained retrieval."""
    documents = []
    badges = data.get("badges", []) if isinstance(data, dict) else []

    for item in badges:
        name = item.get("name", "Unknown")
        key = item.get("key", "")
        mandatory = item.get("mandatory_documents", [])
        mandatory_str = ", ".join(str(d) for d in mandatory) if mandatory else "None"

        alt_groups = item.get("alternative_groups", [])

        if not alt_groups:
            # Badge with no alternatives still gets indexed
            page_content = (
                f"Badge: {name} (key: {key})\n"
                f"Mandatory Documents: {mandatory_str}\n"
                f"No alternative document groups defined."
            )
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "badge_name": name,
                        "badge_key": key,
                        "group_index": -1,
                        "score": 0,
                        "source": "alternative_docs.yaml",
                    },
                )
            )
            continue

        # One chunk per alternative group for fine-grained retrieval
        for idx, group in enumerate(alt_groups):
            docs = group.get("documents", [])
            condition = group.get("condition", "")
            score = group.get("score", 0)
            fields = group.get("fields_to_match", [])
            errors = group.get("error_recovery", [])
            combination = "all required" if group.get("combination_required", False) else "any sufficient"

            error_text = "; ".join(
                f"{e.get('error_type', 'unknown')}: {e.get('message', '')}"
                for e in errors
            )

            page_content = (
                f"Badge: {name} (key: {key})\n"
                f"Mandatory Documents: {mandatory_str}\n"
                f"Alternative Option {idx + 1} (confidence: {score}, combination: {combination}):\n"
                f"  Documents: {', '.join(str(d) for d in docs)}\n"
                f"  Condition: {condition}\n"
                f"  Fields to match: {', '.join(str(f) for f in fields)}\n"
                f"  Error recovery: {error_text}"
            )
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "badge_name": name,
                        "badge_key": key,
                        "group_index": idx,
                        "score": score,
                        "source": "alternative_docs.yaml",
                    },
                )
            )

    return documents


def build_or_get_vectorstore() -> Chroma:
    """Creates or loads a persistent Chroma vector store with hash-based rebuild detection."""
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"YAML file not found at: {YAML_PATH}")

    if not _needs_rebuild():
        logger.debug("Loading existing vector store (YAML unchanged)")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)

    logger.info("Building new vector store from YAML specifications...")

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    documents = _build_documents(data)
    logger.info(f"Indexed {len(documents)} document chunks from {len(data.get('badges', []))} badges")

    # Clear existing store if rebuilding
    if os.path.exists(CHROMA_DIR):
        import shutil
        # Preserve hash file logic — delete contents except hash
        for item in os.listdir(CHROMA_DIR):
            item_path = os.path.join(CHROMA_DIR, item)
            if item != ".yaml_hash":
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=_embeddings,
        persist_directory=CHROMA_DIR,
    )

    _save_hash(_compute_yaml_hash())
    logger.info("Vector store built and persisted successfully")
    return vectorstore