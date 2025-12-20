# Getting Started

This guide will walk you through installing the Agent Framework and building your first agent.

## Prerequisites

- Python 3.10 or higher
- An Anthropic API key ([get one here](https://console.anthropic.com/))

## Installation

### 1. Install the Framework

```bash
# Clone or navigate to the framework
cd agent-framework

# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Configure Environment

Create a `.env` file in your project root:

```bash
# Copy the example
cp .env.example .env
```

Edit `.env` and add your API key:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional: For token encryption (generate with the command below)
# TOKEN_ENCRYPTION_KEY=...

# Optional: For Slack integration
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

Generate an encryption key if needed:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Building Your First Agent

### Step 1: Create Your Agent Class

Create `my_agent.py`:

```python
from agent_framework import Agent


class MyAgent(Agent):
    """A simple agent that can read web content and remember information."""

    def get_system_prompt(self) -> str:
        """Define your agent's behavior and capabilities."""
        return """You are a helpful research assistant. You can:
        - Read and summarize web content
        - Remember important information for later
        - Help organize and retrieve saved information

        When users ask you to remember something, use the save_memory tool.
        When they ask about past information, use get_memories or search_memories."""

    def get_agent_name(self) -> str:
        """Display name for your agent."""
        return "Research Assistant"

    def get_greeting(self) -> str:
        """Greeting shown when the agent starts."""
        return "Hello! I'm your research assistant. I can read web pages and remember important information. How can I help?"
```

### Step 2: Create Your MCP Server

Create `my_server.py` to register available tools:

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
    # Create the server
    server = create_mcp_server("research-assistant")

    # Register web reading tool
    server.register_tool(
        name="fetch_web_content",
        description="Fetch and convert web content to clean markdown. Returns the page content, word count, and extracted links.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                }
            },
            "required": ["url"]
        },
        handler=fetch_web_content,
    )

    # Register memory tools
    server.register_tool(
        name="save_memory",
        description="Save information to persistent memory for later retrieval",
        input_schema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Unique identifier for this memory"
                },
                "value": {
                    "type": "string",
                    "description": "The information to remember"
                },
                "category": {
                    "type": "string",
                    "description": "Optional category to organize memories"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for filtering"
                },
                "importance": {
                    "type": "integer",
                    "description": "Importance level 1-10 (default: 5)"
                }
            },
            "required": ["key", "value"]
        },
        handler=save_memory,
    )

    server.register_tool(
        name="get_memories",
        description="Retrieve saved memories with optional filtering",
        input_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags"
                },
                "min_importance": {
                    "type": "integer",
                    "description": "Minimum importance level"
                }
            }
        },
        handler=get_memories,
    )

    server.register_tool(
        name="search_memories",
        description="Search memories by keyword in keys or values",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        },
        handler=search_memories,
    )

    # Start the server
    server.setup_handlers()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Create a Runner Script

Create `run_agent.py`:

```python
import asyncio
from my_agent import MyAgent


async def main():
    agent = MyAgent(mcp_server_path="my_server.py")
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
```

### Step 4: Run Your Agent

```bash
python run_agent.py
```

You should see:

```
Hello! I'm your research assistant. I can read web pages and remember important information. How can I help?

You:
```

Try these commands:
- "Read https://example.com and summarize it"
- "Remember that I prefer Python for scripting"
- "What have you remembered about me?"

## Adding Custom Tools

### Create a Custom Tool

Let's add a text analysis tool. Create `my_tools.py`:

```python
async def analyze_text(text: str) -> dict:
    """Analyze text and return statistics."""
    words = text.split()
    sentences = text.split('.')

    return {
        "status": "success",
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "character_count": len(text),
        "average_word_length": sum(len(w) for w in words) / len(words) if words else 0
    }
```

### Register Your Tool

Add to `my_server.py`:

```python
from my_tools import analyze_text

# In the main() function, add:
server.register_tool(
    name="analyze_text",
    description="Analyze text and return word count, sentence count, and other statistics",
    input_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to analyze"
            }
        },
        "required": ["text"]
    },
    handler=analyze_text,
)
```

### Update System Prompt

Update your agent's system prompt to mention the new tool:

```python
def get_system_prompt(self) -> str:
    return """You are a helpful research assistant. You can:
    - Read and summarize web content
    - Remember important information for later
    - Analyze text for statistics and insights

    Use the analyze_text tool when users want word counts or text statistics."""
```

## Using Built-in Features

### Memory System

The framework's memory system supports rich querying:

```python
# Save with metadata
save_memory(
    key="user_preference",
    value="Prefers Python over JavaScript",
    category="preferences",
    tags=["python", "languages"],
    importance=8
)

# Filter by category
get_memories(category="preferences")

# Filter by tags
get_memories(tags=["python"])

# Filter by importance
get_memories(min_importance=7)

# Search by keyword
search_memories(query="python")
```

### Slack Integration

To add Slack notifications:

1. **Set up webhook**: Get a webhook URL from Slack ([instructions](https://api.slack.com/messaging/webhooks))

2. **Add to `.env`**:
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

3. **Register the tool**:
```python
from agent_framework.tools import send_slack_message

server.register_tool(
    name="send_slack_message",
    description="Send a message to Slack",
    input_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Message to send"
            },
            "username": {
                "type": "string",
                "description": "Optional bot username"
            }
        },
        "required": ["text"]
    },
    handler=send_slack_message,
)
```

### Token Storage (OAuth)

For tools that need OAuth tokens:

```python
from agent_framework.storage import TokenStore, TokenData
from datetime import datetime, timedelta
from pathlib import Path
import os

# Initialize store
token_store = TokenStore(
    storage_path=Path("./tokens"),
    encryption_key=os.getenv("TOKEN_ENCRYPTION_KEY"),
)

# Save a token
token_data = TokenData(
    access_token="ya29.a0AfH6...",
    refresh_token="1//0eH...",
    expires_at=datetime.utcnow() + timedelta(hours=1),
    token_type="Bearer"
)
token_store.save_token("google_calendar", token_data)

# Retrieve and use
token = token_store.get_token("google_calendar")
if token and not token.is_expired():
    # Use token.access_token
    pass
elif token and token.refresh_token:
    # Refresh the token
    pass
```

## Using Remote MCP Servers

The framework supports connecting to remote MCP servers over HTTPS with automatic OAuth authentication.

### Connecting to a Remote Server

```python
from agent_framework.core.remote_mcp_client import RemoteMCPClient

# Automatic OAuth (will open browser if needed)
async def main():
    client = RemoteMCPClient("https://mcp.example.com/mcp/")

    async with client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[t['name'] for t in tools]}")

        # Call a tool
        result = await client.call_tool("tool_name", {"arg": "value"})
        print(result)

asyncio.run(main())
```

### OAuth Authentication

The remote client automatically handles OAuth:

1. **Discovery**: Fetches OAuth configuration from `.well-known` endpoints
2. **Registration**: Dynamically registers as an OAuth client
3. **Authorization**: Opens browser for user login
4. **Token Storage**: Saves tokens to `~/.claude-code/tokens`
5. **Auto-Refresh**: Refreshes expired tokens automatically

**First connection:**
```
🔐 AUTHENTICATION REQUIRED
============================================================
Server: https://mcp.example.com/mcp/
Your browser will open for authentication.
Please complete the login process in your browser.
============================================================

[Browser opens for login]
✅ OAuth authentication successful, token saved
✅ Connected to remote MCP server
```

**Subsequent connections:**
```
✅ Connected to remote MCP server
```

### Configuration Options

```python
# Manual token (bypasses OAuth)
client = RemoteMCPClient(
    "https://mcp.example.com/mcp/",
    auth_token="your-manual-token"
)

# Custom OAuth settings
client = RemoteMCPClient(
    "https://mcp.example.com/mcp/",
    enable_oauth=True,
    oauth_redirect_port=8889,  # Local callback port
    oauth_scopes="read write admin",  # Custom scopes
    token_storage_dir="/custom/path/tokens"
)

# Disable OAuth (requires manual token)
client = RemoteMCPClient(
    "https://mcp.example.com/mcp/",
    enable_oauth=False,
    auth_token=os.getenv("MCP_AUTH_TOKEN")
)
```

### Using Remote Servers with Agents

You can use remote MCP servers with your agents:

```python
from agent_framework import Agent
from agent_framework.core.remote_mcp_client import RemoteMCPClient

class MyAgent(Agent):
    def __init__(self, remote_server_url: str, **kwargs):
        super().__init__(**kwargs)
        self.remote_client = RemoteMCPClient(remote_server_url)

    async def start(self):
        async with self.remote_client:
            # Now you can use remote_client.list_tools() and call_tool()
            await super().start()
```

### Token Management

```python
# Clear saved tokens (useful for logout or debugging)
client.clear_tokens()

# Check server health
is_healthy = await client.health_check()
```

### Environment Variables

```bash
# Use manual token from environment
export MCP_AUTH_TOKEN="your-token-here"

# Then in code (auth_token will be read from environment)
client = RemoteMCPClient("https://mcp.example.com/mcp/")
```

## Project Organization

For larger projects, organize your code like this:

```
my-agent/
├── my_agent/
│   ├── __init__.py
│   ├── agent.py              # Your Agent subclass
│   ├── server.py             # MCP server setup
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── custom_tool1.py
│   │   └── custom_tool2.py
│   └── prompts.py            # System prompts
├── run_agent.py              # Entry point
├── .env                      # Configuration
├── pyproject.toml            # Dependencies
└── README.md
```

## Next Steps

Now that you have a working agent:

1. **Customize the system prompt** to define specific behaviors
2. **Add domain-specific tools** for your use case
3. **Integrate with external APIs** using token storage
4. **Deploy as a service** or package as a CLI tool

For deeper understanding of the framework architecture and design patterns, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Troubleshooting

### Agent won't start
- Check that your `.env` file has `ANTHROPIC_API_KEY` set
- Verify the MCP server path is correct
- Check for Python syntax errors in your agent/server files

### Tools not working
- Ensure tools are registered before `server.setup_handlers()`
- Check the input schema matches your tool's parameters
- Look for errors in the console output

### Memory not persisting
- Check that the storage directory exists and is writable
- Verify you're using the same storage path between runs
- Look for file permissions issues

## Getting Help

- Check the source code - it's well-documented
- See [ARCHITECTURE.md](ARCHITECTURE.md) for design patterns
- Open an issue on GitHub for bugs or questions
