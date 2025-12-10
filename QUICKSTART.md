# Agent Framework - Quick Start Guide

This guide will help you get started building agents with the Agent Framework.

## Installation

```bash
# Clone or navigate to the framework
cd agent-framework

# Install with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or with pip
pip install -e .
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your Anthropic API key:
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Building Your First Agent

### Step 1: Create Your Agent Class

Create a new Python file (e.g., `my_agent.py`):

```python
from agent_framework import Agent


class MyAgent(Agent):
    """Your custom agent."""

    def get_system_prompt(self) -> str:
        """Define what your agent does."""
        return """You are a helpful assistant that can:
        - Read web content
        - Remember important information
        - Send Slack notifications

        Be helpful and use tools when appropriate."""

    def get_agent_name(self) -> str:
        """Agent display name."""
        return "My Custom Agent"

    def get_greeting(self) -> str:
        """Greeting message."""
        return "Hello! I'm your custom agent. How can I help?"
```

### Step 2: Create Your MCP Server

Create `my_server.py`:

```python
import asyncio
import logging
from agent_framework.server import create_mcp_server
from agent_framework.tools import (
    fetch_web_content,
    save_memory,
    get_memories,
    search_memories,
)

logging.basicConfig(level=logging.INFO)

async def main():
    server = create_mcp_server("my-agent")

    # Register tools
    server.register_tool(
        name="fetch_web_content",
        description="Fetch and read web content as markdown",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"}
            },
            "required": ["url"]
        },
        handler=fetch_web_content,
    )

    # Add memory tools
    server.register_tool(
        name="save_memory",
        description="Save information to memory",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"]
        },
        handler=save_memory,
    )

    server.register_tool(
        name="get_memories",
        description="Retrieve saved memories",
        input_schema={"type": "object", "properties": {}},
        handler=get_memories,
    )

    server.setup_handlers()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Run Your Agent

Create a runner script `run_agent.py`:

```python
import asyncio
from my_agent import MyAgent


async def main():
    agent = MyAgent(mcp_server_path="my_server.py")
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python run_agent.py
```

## Adding Custom Tools

### Create a Custom Tool

```python
# my_tools.py
async def my_custom_tool(input_text: str) -> dict:
    """Your custom tool implementation."""
    result = input_text.upper()  # Example processing

    return {
        "status": "success",
        "result": result,
        "message": "Converted to uppercase"
    }
```

### Register It

In your `my_server.py`:

```python
from my_tools import my_custom_tool

server.register_tool(
    name="my_custom_tool",
    description="Converts text to uppercase",
    input_schema={
        "type": "object",
        "properties": {
            "input_text": {
                "type": "string",
                "description": "Text to convert"
            }
        },
        "required": ["input_text"]
    },
    handler=my_custom_tool,
)
```

## Using Built-in Features

### Memory System

The framework includes a persistent memory system:

```python
# In your agent's system prompt, mention memory usage:
"""You can remember information using:
- save_memory(key, value, category, tags, importance)
- get_memories(category, tags, min_importance)
- search_memories(query)

Always save important user preferences and context!"""
```

### Token Storage

The framework includes encrypted token storage for OAuth workflows:

```python
from agent_framework.storage import TokenStore, TokenData
from datetime import datetime, timedelta

# Set up in your .env:
# TOKEN_ENCRYPTION_KEY=... (generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Use in your tool:
token_store = TokenStore(
    storage_path=Path("./tokens"),
    encryption_key=os.getenv("TOKEN_ENCRYPTION_KEY"),
)

# Save a token
token_data = TokenData(
    access_token="your_token",
    expires_at=datetime.utcnow() + timedelta(hours=1)
)
token_store.save_token("service_name", token_data)

# Retrieve a token
token = token_store.get_token("service_name")
if token and not token.is_expired():
    # Use the token
    pass
```

### Slack Integration

```python
from agent_framework.tools import send_slack_message

# Set SLACK_WEBHOOK_URL in .env

# Register in your server
server.register_tool(
    name="send_slack_message",
    description="Send message to Slack",
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    },
    handler=send_slack_message,
)
```

### Web Content Reading

```python
from agent_framework.tools import fetch_web_content

# Already implemented - just register it!
# Fetches web pages and converts to clean markdown
```

## Advanced: Custom Storage Backends

The framework uses file-based storage by default. To use a database:

```python
from agent_framework.storage import MemoryStore

class DatabaseMemoryStore(MemoryStore):
    """Custom database-backed memory store."""

    def __init__(self, db_connection):
        self.db = db_connection
        # Implement the same interface:
        # - save_memory()
        # - get_memory()
        # - get_all_memories()
        # - search_memories()
        # - delete_memory()
```


## Next Steps

1. **Customize your agent** - Modify the system prompt to define behavior
2. **Add custom tools** - Create tools specific to your use case
3. **Integrate with APIs** - Use OAuth for external services
4. **Deploy** - Run your agent as a service or CLI tool

## Need Help?

- Read the source code - it's well documented!
- Check `FRAMEWORK_SUMMARY.md` for design decisions and architecture details
- The framework was extracted from real production agents, so patterns are battle-tested

## Architecture Overview

```
┌─────────────────┐
│  Your Agent     │ (Extends Agent base class)
│  - System Prompt│
│  - Greeting     │
└────────┬────────┘
         │
         │ Uses
         ▼
┌─────────────────┐
│  MCP Client     │ (Discovers and calls tools)
└────────┬────────┘
         │
         │ Connects to
         ▼
┌─────────────────┐
│  MCP Server     │ (Your custom server)
│  - Tool Registry│
│  - Error Handler│
└────────┬────────┘
         │
         │ Provides
         ▼
┌─────────────────┐
│  Tools          │
│  - Web Reader   │
│  - Memory       │
│  - Slack        │
│  - Custom Tools │
└─────────────────┘
```

Happy building! 🚀
