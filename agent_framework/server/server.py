"""Base MCP server implementation.

This module provides utilities for creating MCP servers with tool registration.
"""

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool

from agent_framework.tools import (
    fetch_web_content,
    get_memories,
    save_memory,
    search_memories,
    send_slack_message,
)

logger = logging.getLogger(__name__)


class MCPServerBase:
    """
    Base class for MCP servers with tool registration.

    This provides a clean interface for building MCP servers with automatic
    tool registration and error handling.
    """

    def __init__(self, name: str, setup_defaults: bool = True):
        """
        Initialize MCP server.

        Args:
            name: Server name
            setup_defaults: Whether or not to set up default tools
        """
        self.app = Server(name)
        self.tools: dict[str, dict[str, Any]] = {}
        self._tool_handlers: dict[str, Callable] = {}

        if setup_defaults:
            setup_default_tools(self)

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable,
    ):
        """
        Register a tool with the server.

        Args:
            name: Tool name
            description: Tool description
            input_schema: JSON schema for tool inputs
            handler: Async function to handle tool calls
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
        }
        self._tool_handlers[name] = handler
        logger.info(f"Registered tool: {name}")

    def setup_handlers(self):
        """Set up MCP handlers for tool listing and calling."""

        @self.app.list_tools()
        async def list_tools() -> list[Tool]:
            """List available MCP tools."""
            logger.info("Listing available tools")
            return [
                Tool(
                    name=tool_info["name"],
                    description=tool_info["description"],
                    inputSchema=tool_info["input_schema"],
                )
                for tool_info in self.tools.values()
            ]

        @self.app.call_tool()
        async def call_tool(
            name: str, arguments: Any
        ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            """Execute a tool with the given arguments."""
            logger.info(f"Calling tool: {name} with arguments: {arguments}")

            try:
                # Check if tool exists
                if name not in self._tool_handlers:
                    raise ValueError(f"Unknown tool: {name}")

                # Call the handler
                handler = self._tool_handlers[name]
                result = await handler(**arguments)

                # Return as TextContent with JSON
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2),
                    )
                ]

            except ValueError as e:
                logger.error(f"Validation error in {name}: {e}")
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "validation_error",
                                "message": str(e),
                                "tool": name,
                            }
                        ),
                    )
                ]

            except PermissionError as e:
                logger.error(f"Auth error in {name}: {e}")
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "authentication_required",
                                "message": str(e),
                                "tool": name,
                                "action_required": "Please complete OAuth authentication flow",
                            }
                        ),
                    )
                ]

            except Exception as e:
                logger.exception(f"Error executing tool {name}: {e}")
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "execution_error",
                                "message": str(e),
                                "tool": name,
                            }
                        ),
                    )
                ]

    async def run(self):
        """Run the MCP server."""
        logger.info(f"Starting MCP Server: {self.app.name}")

        # Run the server using stdio transport
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP server running on stdio")
            await self.app.run(
                read_stream,
                write_stream,
                self.app.create_initialization_options(),
            )


def create_mcp_server(name: str) -> MCPServerBase:
    """
    Create a new MCP server.

    This is a convenience function for creating servers.

    Args:
        name: Server name

    Returns:
        MCPServerBase instance

    Example:
        ```python
        server = create_mcp_server("my-agent")

        server.register_tool(
            name="my_tool",
            description="Does something useful",
            input_schema={
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                },
                "required": ["param"]
            },
            handler=my_tool_handler
        )

        server.setup_handlers()
        await server.run()
        ```
    """
    return MCPServerBase(name)


def setup_default_tools(server: MCPServerBase) -> None:
    server.register_tool(
        name="fetch_web_content",
        description=(
            "Fetch web content and convert to clean, LLM-readable markdown format. "
            "Extracts the main content from a webpage, removes navigation and ads, "
            "and returns it as markdown. Useful for reading articles, blog posts, "
            "documentation, or any web content you want to analyze or comment on."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must start with http:// or https://)",
                },
                "max_length": {
                    "type": "integer",
                    "minimum": 1000,
                    "maximum": 100000,
                    "default": 50000,
                    "description": "Maximum content length in characters (default: 50000)",
                },
            },
            "required": ["url"],
        },
        handler=fetch_web_content,
    )

    server.register_tool(
        name="save_memory",
        description=(
            "Save important information to persistent memory. Use this to remember "
            "user preferences, goals, insights from analyses, brand voice, and any "
            "other details that should be recalled in future conversations."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Unique identifier (e.g., 'user_blog_url', 'brand_voice', 'twitter_goal')",
                },
                "value": {
                    "type": "string",
                    "description": "The information to remember",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category: 'user_preference', 'fact', 'goal', 'insight', etc.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for organization (e.g., ['seo', 'twitter'])",
                },
                "importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Importance level 1-10 (1=low, 5=medium, 10=critical)",
                },
            },
            "required": ["key", "value"],
        },
        handler=save_memory,
    )

    server.register_tool(
        name="get_memories",
        description=(
            "Retrieve stored memories from previous conversations. Returns memories "
            "sorted by importance. Use this at the start of conversations to recall "
            "context about the user."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category (e.g., 'user_preference', 'goal')",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags (returns memories with any matching tag)",
                },
                "min_importance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Only return memories with importance >= this value",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                    "description": "Maximum number of memories to return",
                },
            },
            "required": [],
        },
        handler=get_memories,
    )

    server.register_tool(
        name="search_memories",
        description=(
            "Search for memories by keyword. Searches both keys and values. "
            "Useful when you don't know the exact memory key."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term (case-insensitive)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Maximum number of results",
                },
            },
            "required": ["query"],
        },
        handler=search_memories,
    )
    server.register_tool(
        name="send_slack_message",
        description=(
            "Send a message to Slack using an incoming webhook. "
            "Useful for posting content, notifications, and updates to Slack channels. "
            "The webhook URL can be provided or will use SLACK_WEBHOOK_URL from environment. "
            "Supports custom usernames, emoji icons, and channel overrides."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The message text to send (supports Slack markdown formatting)",
                },
                "webhook_url": {
                    "type": "string",
                    "description": "Optional Slack webhook URL. If not provided, uses SLACK_WEBHOOK_URL from environment",
                },
                "username": {
                    "type": "string",
                    "description": "Optional custom username for the message",
                },
                "icon_emoji": {
                    "type": "string",
                    "description": "Optional emoji icon (e.g., 'robot_face', 'tada', ':rocket:')",
                },
                "channel": {
                    "type": "string",
                    "description": "Optional channel override (e.g., '#general', '@username')",
                },
            },
            "required": ["text"],
        },
        handler=send_slack_message,
    )
