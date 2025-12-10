# Agent Framework

A reusable LLM agent framework built on the Model Context Protocol (MCP), extracted from production agent implementations.

## Features

### Core Agent System
- **Base Agent Class**: Agentic loop with conversation management
- **MCP Client**: Connect to MCP servers for tool discovery and execution
- **Hot Reload**: Update tools without losing conversation context
- **Token Tracking**: Monitor LLM token usage

### Storage & Memory
- **Memory System**: Persistent storage across conversations with categories, tags, and search
- **Token Store**: Encrypted OAuth token storage with automatic refresh
- **Pluggable Backends**: Easy migration from file-based to database storage


### Generic Tools
- **Web Reader**: Fetch and convert web content to clean markdown
- **Slack Integration**: Post messages via incoming webhooks
- **Memory Tools**: Save, retrieve, and search persistent memories

### MCP Server Infrastructure
- **Tool Registry**: Clean tool registration and discovery
- **Error Handling**: Validation, authentication, and execution error categories
- **JSON Schema**: Type-safe tool definitions

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

## Quick Start

### Creating an Agent

```python
from agent_framework.core import Agent

class MyAgent(Agent):
    """Your custom agent implementation."""

    def get_system_prompt(self) -> str:
        return """You are a helpful assistant that..."""

# Run the agent
async def main():
    agent = MyAgent(
        api_key="your-anthropic-api-key",
        mcp_server_path="path/to/your/mcp_server.py"
    )
    await agent.start()

import asyncio
asyncio.run(main())
```

### Creating Custom Tools

```python
# In your_tools.py
async def my_custom_tool(param: str) -> dict[str, Any]:
    """Your tool implementation."""
    return {"result": f"Processed: {param}"}

# In your MCP server
from agent_framework.server import create_mcp_server

app = create_mcp_server("my-agent")

@app.register_tool(
    name="my_custom_tool",
    description="What this tool does",
    input_schema={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "Parameter description"}
        },
        "required": ["param"]
    }
)
async def handle_my_tool(arguments: dict) -> dict:
    return await my_custom_tool(**arguments)
```

## Architecture

The framework separates concerns into distinct modules:

```
agent_framework/
├── core/              # Agent orchestration and MCP client
├── tools/             # Reusable tools (web, slack, memory)
├── storage/           # Memory and token storage
├── server/            # MCP server base classes
└── utils/             # Logging, errors, config
```

## Documentation

For more detailed information, see:
- `QUICKSTART.md` - Step-by-step guide to building your first agent
- `FRAMEWORK_SUMMARY.md` - Detailed framework overview and design decisions

## License

MIT
