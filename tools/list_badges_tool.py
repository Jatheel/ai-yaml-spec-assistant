"""Tool to list all available certification badges in the specification database."""

import logging
from langchain_core.tools import tool
from tools.alternative_docs_tool import _load_yaml, _normalize_key

logger = logging.getLogger("doc-agent.tools.badges")


@tool
def list_available_badges() -> str:
    """
    Lists all certification badges available in the YAML specification database.
    Use this tool when the user asks what certifications are supported, or when
    a badge lookup fails and you need to show available options.

    Returns:
        A formatted list of all badge names and keys with alternative counts.
    """
    try:
        data = _load_yaml()
        badges = data.get("badges", [])

        if not badges:
            return "No badges found in the specification database."

        lines = [f"Available Certification Badges ({len(badges)} total):\n"]
        for b in badges:
            name = b.get("name", "Unknown")
            key = b.get("key", _normalize_key(name))
            mandatory = b.get("mandatory_documents", [])
            alt_count = len(b.get("alternative_groups", []))
            mandatory_str = ", ".join(str(d) for d in mandatory) if mandatory else "None"

            lines.append(f"  • {name} (key: {key})")
            lines.append(f"    Primary doc: {mandatory_str}")
            lines.append(f"    Alternative options: {alt_count}")
            lines.append("")

        logger.info(f"Listed {len(badges)} badges")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Badge listing failed: {e}")
        return f"Error listing badges: {str(e)}"
