"""Base agent class for building LLM agents with MCP tools.

This module provides the foundational Agent class that handles:
- Conversation management
- Tool execution via MCP
- Token usage tracking
- Interactive CLI interface
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlock, ToolUseBlock
from dotenv import load_dotenv

from .mcp_client import MCPClient
from .remote_mcp_client import RemoteMCPClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class InvalidToolName(Exception):
    def __init__(self, message: str):            
        super().__init__(f"{message} tool not found!")


class Agent(ABC):
    """
    Base agent class using Claude and MCP tools.

    This class provides the core agentic loop that:
    1. Accepts user requests
    2. Calls Claude via Anthropic SDK
    3. Executes MCP tools as needed
    4. Processes results and continues until done

    Subclasses should override:
    - get_system_prompt(): Return the system prompt for the agent
    - get_greeting(): Return the greeting message shown to users (optional)
    - get_agent_name(): Return the agent name for display (optional)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        mcp_server_path: str = "mcp_server/server.py",
        mcp_urls: list[str] | None = None
    ):
        """
        Initialize the agent.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
            mcp_server_path: Path to MCP server script
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.model = model
        self.mcp_server_path = mcp_server_path
        self.mcp_urls: list[str] = mcp_urls or []
        self.tools: dict[str, list[str]] = {}

        # Initialize Anthropic client
        self.client = AsyncAnthropic(api_key=self.api_key)

        # Initialize MCP client
        self.mcp_client = MCPClient(mcp_server_path)

        # Conversation history
        self.messages: list[MessageParam] = []

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        logger.info(f"Initialized {self.get_agent_name()} with model: {model}")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent.

        This defines the agent's role, capabilities, and behavior.
        Should be implemented by subclasses.

        Returns:
            System prompt string
        """
        pass

    def get_agent_name(self) -> str:
        """
        Return the agent name for display.

        Override this to customize the agent name shown in the CLI.

        Returns:
            Agent name (defaults to class name)
        """
        return self.__class__.__name__

    def get_greeting(self) -> str:
        """
        Return the greeting message shown to users.

        Override this to customize the greeting.

        Returns:
            Greeting message
        """
        return f"Hello! I'm {self.get_agent_name()}. How can I help you today?"

    async def _call_mcp_tool_with_reconnect(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Call an MCP tool with automatic reconnection.

        This allows the MCP server to be restarted between calls
        without losing the agent's conversation context.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """

        # Local tools should take precedence over remote tools if there are any name collisions.
        # TODO: Throw an error if there are name collisions? 
        if tool_name in self.tools["local"]:
            async with self.mcp_client.connect():
                return await self.mcp_client.call_tool(tool_name, arguments)

        for url in self.mcp_urls:
            async with RemoteMCPClient(url) as mcp:
                result = await mcp.call_tool(tool_name, arguments)

                # Handle result - could be string or dict
                if isinstance(result, str):
                    try:
                        # Try to parse as JSON
                        result_dict = json.loads(result)
                        return result_dict
                    except json.JSONDecodeError:
                        return {"result": result}
                else:
                    return result

        # If the tool isn't found, raise an exception.
        raise InvalidToolName(tool_name)


    async def _get_available_tools(self) -> list[str]:
        """Get list of available MCP tools (reconnects to server)."""

        # Get tools from local MCP server
        async with self.mcp_client.connect():
            self.tools["local"] = self.mcp_client.get_available_tools()

        # Get tools from remote MCP server(s) if applicable
        for url in self.mcp_urls:
            async with RemoteMCPClient(url) as mcp:
                mcp_tools = await mcp.list_tools()
                self.tools[url] = [tool["name"] for tool in mcp_tools]

        # Return the concatenation of all the tool lists
        return [item for lst in self.tools.values() for item in lst]
                

    async def start(self):
        """Start an interactive session with the agent."""
        logger.info(f"Starting {self.get_agent_name()} interactive session")

        print("\n" + "=" * 70)
        print(self.get_agent_name().upper())
        print("=" * 70)
        print(self.get_greeting())
        print("\nType 'exit' or 'quit' to end the session.")
        print("Type 'stats' to see token usage statistics.")
        print("Type 'reload' to reconnect to MCP server and discover updated tools.")
        print("=" * 70 + "\n")

        # Discover available tools (will reconnect each time we need them)
        try:
            tools_list = await self._get_available_tools()
            logger.info(f"Discovered MCP tools: {tools_list}")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            print(f"\n⚠️  Warning: Could not connect to MCP server: {e}")
            print("Make sure the MCP server is running and try again.\n")

        # Test remote MCP connection(s)
        for url in self.mcp_urls:
            try:
                print(f"🔌 Connecting to remote MCP server {url}...", flush=True)
                async with RemoteMCPClient(url) as mcp:
                    tools = await asyncio.wait_for(mcp.list_tools(), timeout=10.0)
                    logger.info(f"Connected to MCP server with {len(tools)} tools")
                    print(f"✅ Connected to {url}")
                    print(f"✅ Found {len(tools)} tools\n", flush=True)
            except asyncio.TimeoutError:
                print(f"❌ Timeout while connecting to MCP server at {url}")
                print("The connection was established but listing tools timed out.")
                return
            except Exception as e:
                print(f"❌ Failed to connect to MCP server at {url}")
                print(f"Error: {e}")
                print("\nPlease ensure:")
                print("1. The MCP server is running")
                print("2. The URL is correct")
                print("3. The server is accessible")
                return

        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.lower() in ["exit", "quit"]:
                    print("\nGoodbye! 👋")
                    break

                if user_input.lower() == "stats":
                    self._print_stats()
                    continue

                if user_input.lower() == "reload":
                    print("\n🔄 Reconnecting to MCP server...")
                    try:
                        tools_list = await self._get_available_tools()
                        print(f"✓ Connected! Available tools: {', '.join(tools_list)}")
                    except Exception as e:
                        print(f"✗ Failed to connect: {e}")
                    continue

                # Process user message
                response = await self.process_message(user_input)

                # Display response
                print(f"\nAssistant: {response}")

            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye! 👋")
                break

            except Exception as e:
                logger.exception(f"Error in interaction loop: {e}")
                print(f"\nError: {e}")
                print("Please try again or type 'exit' to quit.")

    async def process_message(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.

        This implements the agentic loop:
        1. Add user message to conversation
        2. Call Claude with available tools
        3. Execute any tool calls via MCP
        4. Continue until Claude provides a final response

        Args:
            user_message: The user's input message

        Returns:
            The agent's response as a string
        """
        # Add user message to conversation history
        self.messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # Convert MCP tools to Anthropic tool format (reconnects to get latest)
        tools = await self._convert_mcp_tools_to_anthropic()

        # Agentic loop - continue until we get a text response
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}")

            try:
                # Call Claude
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.get_system_prompt(),
                    messages=self.messages,
                    tools=tools,
                )

                # Track token usage
                self.total_input_tokens += response.usage.input_tokens
                self.total_output_tokens += response.usage.output_tokens

                logger.info(
                    f"Claude response - input tokens: {response.usage.input_tokens}, "
                    f"output tokens: {response.usage.output_tokens}"
                )

                # Check stop reason
                if response.stop_reason == "end_turn":
                    # Extract text response
                    text_response = self._extract_text_from_response(response.content)

                    # Add assistant response to conversation
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": response.content,
                        }
                    )

                    return text_response

                elif response.stop_reason == "tool_use":
                    # Extract tool calls
                    tool_calls = [
                        block for block in response.content if isinstance(block, ToolUseBlock)
                    ]

                    if not tool_calls:
                        logger.warning("No tool calls found despite tool_use stop reason")
                        text_response = self._extract_text_from_response(response.content)
                        self.messages.append(
                            {
                                "role": "assistant",
                                "content": response.content,
                            }
                        )
                        return text_response

                    # Add assistant response to conversation (with tool calls)
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": response.content,
                        }
                    )

                    # Execute tool calls and collect results
                    tool_results = []
                    for tool_call in tool_calls:
                        logger.info(f"Executing tool: {tool_call.name}")

                        try:
                            # Call MCP tool (reconnects to server each time)
                            result = await self._call_mcp_tool_with_reconnect(
                                tool_call.name,
                                tool_call.input,
                            )

                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_call.id,
                                    "content": str(result),
                                }
                            )

                        except PermissionError as e:
                            # Handle auth errors
                            logger.warning(f"Authentication error for {tool_call.name}: {e}")
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_call.id,
                                    "content": f"Authentication required: {e}",
                                    "is_error": True,
                                }
                            )

                        except Exception as e:
                            # Handle other tool errors
                            logger.error(f"Tool execution error for {tool_call.name}: {e}")
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_call.id,
                                    "content": f"Tool execution failed: {e}",
                                    "is_error": True,
                                }
                            )

                    # Add tool results to conversation
                    self.messages.append(
                        {
                            "role": "user",
                            "content": tool_results,
                        }
                    )

                    # Continue loop to get Claude's response to tool results

                else:
                    # Unexpected stop reason
                    logger.warning(f"Unexpected stop reason: {response.stop_reason}")
                    text_response = self._extract_text_from_response(response.content)
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": response.content,
                        }
                    )
                    return text_response

            except Exception as e:
                logger.exception(f"Error in agent loop: {e}")
                return f"I encountered an error: {e}. Please try again."

        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached")
        return "I apologize, but I'm having trouble completing this request. Please try rephrasing or breaking it into smaller steps."

    async def _convert_mcp_tools_to_anthropic(self) -> list[dict[str, Any]]:
        """
        Convert MCP tool definitions to Anthropic tool format.

        Reconnects to MCP server to get latest tool definitions.
        This allows tools to be updated without restarting the agent.

        Returns:
            List of tool definitions in Anthropic format
        """
        anthropic_tools: list[dict[str, str]] = []

        # Reconnect to get latest tools
        async with self.mcp_client.connect():
            for _, tool_info in self.mcp_client.available_tools.items():
                anthropic_tools.append({
                    "name": tool_info.name,
                    "description": tool_info.description,
                    "input_schema": tool_info.inputSchema,
                })

        # Get remote MCP Server tools
        for url in self.mcp_urls:
            async with RemoteMCPClient(url) as mcp:
                mcp_tools = await mcp.list_tools()

                # Convert to Anthropic format
                anthropic_tools += [{
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"]
                } for tool in mcp_tools]

        return anthropic_tools

    def _extract_text_from_response(self, content: list[Any]) -> str:
        """
        Extract text content from Claude's response.

        Args:
            content: Response content blocks

        Returns:
            Concatenated text content
        """
        text_parts = []
        for block in content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)

        return "\n\n".join(text_parts) if text_parts else ""

    def _print_stats(self):
        """Print token usage statistics."""
        total_tokens = self.total_input_tokens + self.total_output_tokens

        print("\n" + "=" * 70)
        print("TOKEN USAGE STATISTICS")
        print("=" * 70)
        print(f"Input tokens:  {self.total_input_tokens:,}")
        print(f"Output tokens: {self.total_output_tokens:,}")
        print(f"Total tokens:  {total_tokens:,}")
        print(f"Conversations: {len([m for m in self.messages if m['role'] == 'user'])}")
        print("=" * 70)

    def reset_conversation(self):
        """Reset the conversation history."""
        self.messages = []
        logger.info("Conversation history reset")
