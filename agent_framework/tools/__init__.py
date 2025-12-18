"""Generic tools for agents."""

from .memory import get_memories, save_memory, search_memories
from .slack import send_slack_message
from .web_reader import fetch_web_content

__all__ = [
    "fetch_web_content",
    "send_slack_message",
    "save_memory",
    "get_memories",
    "search_memories",
]
