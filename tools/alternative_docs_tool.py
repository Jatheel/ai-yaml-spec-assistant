"""Deterministic YAML-backed alternative document lookup tool with caching."""

import re
import logging
from pathlib import Path

import yaml
from langchain.tools import tool

logger = logging.getLogger("doc-agent.tools.yaml")

YAML_PATH = Path(__file__).parent.parent / "data" / "alternative_docs.yaml"

# Module-level cache to avoid re-reading YAML on every tool call
_yaml_cache = None
_yaml_mtime = None


def _load_yaml() -> dict:
    """Load and cache the YAML specification file with mtime-based invalidation."""
    global _yaml_cache, _yaml_mtime
    try:
        current_mtime = YAML_PATH.stat().st_mtime
        if _yaml_cache is not None and _yaml_mtime == current_mtime:
            return _yaml_cache
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            _yaml_cache = yaml.safe_load(f)
            _yaml_mtime = current_mtime
            logger.info(f"YAML loaded: {len(_yaml_cache.get('badges', []))} badges")
            return _yaml_cache
    except FileNotFoundError:
        logger.error(f"YAML file not found: {YAML_PATH}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parse error: {e}")
        return {"badges": [], "_error": f"YAML parse error: {e}"}


def _clean_text_list(values) -> list[str]:
    """Sanitize a list of values into clean, non-empty strings."""
    cleaned = []
    for value in values or []:
        text = str(value).strip() if value is not None else ""
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_key(text: str) -> str:
    """Normalize a badge name or key to a comparable slug."""
    normalized = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    return normalized.strip("_")


def _find_badge(badges: list[dict], badge_input: str) -> dict | None:
    """Multi-strategy badge lookup: key → exact name → partial name match."""
    input_key = _normalize_key(badge_input)

    # Strategy 1: Explicit key match
    badge = next(
        (b for b in badges if _normalize_key(str(b.get("key", ""))) == input_key),
        None,
    )
    if badge:
        return badge

    # Strategy 2: Exact name match (case-insensitive)
    badge = next(
        (
            b
            for b in badges
            if str(b.get("name", "")).strip().lower() == badge_input.strip().lower()
        ),
        None,
    )
    if badge:
        return badge

    # Strategy 3: Partial display-name match
    return next(
        (
            b
            for b in badges
            if badge_input.strip().lower() in str(b.get("name", "")).strip().lower()
        ),
        None,
    )


def _format_alternatives(badge: dict) -> str:
    """Format a badge's alternative document options into a structured text block."""
    lines = []
    lines.append(f"Badge: {badge['name']}")
    mandatory_docs = _clean_text_list(badge.get("mandatory_documents", []))
    primary_docs_text = ", ".join(mandatory_docs) if mandatory_docs else "Not specified"
    lines.append(f"Primary document(s): {primary_docs_text}\n")

    groups = badge.get("alternative_groups", [])

    if not groups:
        lines.append("No alternative document groups are defined for this badge.")
        return "\n".join(lines)

    sorted_groups = sorted(groups, key=lambda x: x.get("score", 0), reverse=True)
    lines.append(f"There are {len(sorted_groups)} alternative option(s):\n")

    for i, group in enumerate(sorted_groups, 1):
        score = group.get("score", 0)
        docs = _clean_text_list(group.get("documents", []))
        condition = group.get("condition", "")
        required = group.get("combination_required", False)
        fields = group.get("fields_to_match", [])
        errors = group.get("error_recovery", [])

        if score >= 0.8:
            confidence_label = "High"
        elif score >= 0.6:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"

        doc_type = "ALL of these documents" if required else "ANY of these documents"

        lines.append(f"Option {i} — Confidence: {confidence_label} ({score})")
        lines.append(f"  You need to provide {doc_type}:")
        if docs:
            for doc in docs:
                lines.append(f"    - {doc}")
        else:
            lines.append("    - No document names configured")

        if condition:
            lines.append(f"  Condition: {condition}")

        if fields:
            lines.append(
                f"  Fields that must match across all documents: {', '.join(fields)}"
            )

        if errors:
            lines.append("  Common issues to watch for:")
            for e in errors:
                lines.append(f"    ! {e['error_type']}: {e['message']}")

        lines.append("")

    lines.append("Recommended Action:")
    lines.append(
        "  Start with Option 1 (highest confidence) and upload all required document(s)."
    )

    return "\n".join(lines)


@tool
def get_alternative_docs(badge_name: str) -> str:
    """
    Use this tool when a supplier cannot provide the primary certification
    document for a badge. Given a badge name, returns all acceptable alternative
    document combinations the supplier can submit instead, along with
    conditions and next steps.

    Args:
        badge_name: The certification badge name e.g. 'FSSC 22000', 'USDA Organic'

    Returns:
        A formatted string describing alternative document options and guidance.
    """
    try:
        data = _load_yaml()
    except FileNotFoundError:
        return "Error: Specification YAML file not found. Please check data/alternative_docs.yaml exists."
    except Exception as e:
        return f"Error loading specifications: {str(e)}"

    badges = data.get("badges", [])
    if not badges:
        return "Error: No badges found in the specification file."

    badge = _find_badge(badges, badge_name)

    if not badge:
        available = [
            f"{b.get('name', 'Unknown')} (key: {b.get('key', _normalize_key(str(b.get('name', ''))))})"
            for b in badges
        ]
        return (
            f"No badge found matching '{badge_name}'.\n"
            f"Available badges: {', '.join(available)}"
        )

    logger.info(f"Badge lookup hit: {badge.get('name')} for query '{badge_name}'")
    return _format_alternatives(badge)