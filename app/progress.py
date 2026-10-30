import logging
from typing import Any

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)


def log(message: str) -> None:
    """Send a progress message to Streamlit (falls back to the logger)."""
    try:
        get_stream_writer()({"message": message})
    except Exception:
        logger.info(message)


def describe(node: str, update: dict[str, Any]) -> str:
    """Turn a finished graph node into a readable progress message."""
    if node == "researcher":
        return f"Research complete ({len(update.get('sources', []))} web sources)"
    if node == "planner":
        return "Outline created"
    if node == "writer":
        return f"Draft #{update.get('revision', '?')} written"
    if node == "critic":
        return f"Critic scored the draft {update.get('score', '?')}/10"
    if node == "editor":
        return "Final blog post polished"
    if node == "linkedin":
        return "LinkedIn post ready"
    if node == "twitter":
        return "Twitter/X thread ready"
    return f"{node} finished"