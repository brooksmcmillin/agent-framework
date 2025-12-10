"""Generic tools for agents."""

from .web_reader import fetch_web_content
from .slack import send_slack_message
from .memory import save_memory, get_memories, search_memories

__all__ = [
    "fetch_web_content",
    "send_slack_message",
    "save_memory",
    "get_memories",
    "search_memories",
]
