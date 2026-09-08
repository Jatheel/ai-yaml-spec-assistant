import yaml
import re
from pathlib import Path
from langchain.tools import tool

# Path to your YAML file
YAML_PATH = Path(__file__).parent.parent / "data" / "alternative_docs.yaml"

def _load_yaml():
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _clean_text_list(values):
    cleaned = []
    for value in values or []:
        text = str(value).strip() if value is not None else ""
        if text:
            cleaned.append(text)
    return cleaned


def _normalize_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    return normalized.strip("_")


def _find_badge(badges: list[dict], badge_input: str) -> dict | None:
    input_key = _normalize_key(badge_input)

    # Robust lookup: explicit key match first.
    badge = next((b for b in badges if _normalize_key(str(b.get("key", ""))) == input_key), None)
    if badge:
        return badge

    # Backward compatible: exact name match.
    badge = next((b for b in badges if str(b.get("name", "")).strip().lower() == badge_input.strip().lower()), None)
    if badge:
        return badge

    # Fallback: partial display-name match.
    return next((b for b in badges if badge_input.strip().lower() in str(b.get("name", "")).strip().lower()), None)

def _format_alternatives(badge: dict) -> str:
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
            lines.append(f"  Fields that must match across all documents: {', '.join(fields)}")

        if errors:
            lines.append("  Common issues to watch for:")
            for e in errors:
                lines.append(f"    ! {e['error_type']}: {e['message']}")

        lines.append("")

    lines.append("Recommended Action:")
    lines.append("  Start with Option 1 (highest confidence) and upload all required document(s).")

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
    data = _load_yaml()

    badge = _find_badge(data["badges"], badge_name)

    if not badge:
        available = [f"{b.get('name', 'Unknown')} (key: {b.get('key', _normalize_key(str(b.get('name', ''))))})" for b in data["badges"]]
        return (
            f"No badge found matching '{badge_name}'.\n"
            f"Available badges: {', '.join(available)}"
        )

    return _format_alternatives(badge)