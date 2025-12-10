# Agent Framework - Summary

This document provides a detailed overview of the framework architecture and design decisions.

## What Is This Framework

The Agent Framework provides generic, reusable components for building LLM agents with the Model Context Protocol (MCP).

### Core Framework Components

#### 1. **Core Agent System** (`agent_framework/core/`)
- **`agent.py`**: Base `Agent` class with agentic loop
  - Conversation management
  - Tool execution via MCP
  - Token usage tracking
  - Interactive CLI interface
  - Abstract methods for customization (system prompt, greeting, agent name)

- **`mcp_client.py`**: MCP protocol client
  - Connects to MCP servers via stdio
  - Discovers available tools dynamically
  - Executes tool calls with error handling
  - Hot-reload support (reconnects for each tool call)

- **`config.py`**: Configuration management
  - Pydantic settings with `.env` support
  - Auto-creates storage directories
  - Type-safe configuration

#### 2. **Storage Layer** (`agent_framework/storage/`)
- **`memory_store.py`**: Persistent memory system
  - File-based JSON storage (easy migration to DB)
  - Categories, tags, importance levels
  - Search and filter capabilities
  - Timestamped entries

- **`token_store.py`**: Secure OAuth token storage
  - Encrypted file storage (Fernet)
  - Token expiration checking
  - Automatic refresh support
  - Proper file permissions (600)

#### 3. **Generic Tools** (`agent_framework/tools/`)
- **`web_reader.py`**: Web content fetching
  - Converts HTML to clean markdown
  - Removes navigation, ads, footers
  - Configurable max length
  - Metadata extraction (word count, links, images)

- **`slack.py`**: Slack webhook integration
  - Incoming webhook support
  - Custom username, emoji, channel
  - Markdown formatting
  - URL validation

- **`memory.py`**: Memory tool implementations
  - `save_memory()` - Store information
  - `get_memories()` - Retrieve with filters
  - `search_memories()` - Keyword search
  - Global memory store instance

#### 4. **MCP Server Infrastructure** (`agent_framework/server/`)
- **`server.py`**: Base MCP server
  - Clean tool registration API
  - Automatic error handling (validation, auth, execution)
  - JSON schema support
  - Stdio transport
  - `create_mcp_server()` factory function

#### 5. **Utilities** (`agent_framework/utils/`)
- **`errors.py`**: Error types
  - `AgentError` base class
  - `ValidationError`
  - `AuthenticationError`
  - `ToolExecutionError`

### Package Structure

```
agent-framework/
├── agent_framework/           # Main package
│   ├── __init__.py           # Exports Agent, MCPClient, Settings
│   ├── core/                 # Core functionality
│   │   ├── agent.py          # Base Agent class
│   │   ├── mcp_client.py     # MCP client
│   │   └── config.py         # Configuration
│   ├── storage/              # Storage backends
│   │   ├── memory_store.py   # Memory storage
│   │   └── token_store.py    # Token storage
│   ├── tools/                # Generic tools
│   │   ├── web_reader.py     # Web content fetcher
│   │   ├── slack.py          # Slack integration
│   │   └── memory.py         # Memory tools
│   ├── server/               # MCP server
│   │   └── server.py         # Server base class
│   └── utils/                # Utilities
│       └── errors.py         # Error types
├── pyproject.toml            # Package metadata
├── LICENSE                   # MIT License
├── README.md                 # Overview
├── FRAMEWORK_SUMMARY.md      # Detailed framework overview
├── QUICKSTART.md             # Quick start guide
└── .env.example              # Environment variables template
```

## How to Use the Framework

### 1. **Build a New Agent**

Extend the `Agent` base class:

```python
from agent_framework import Agent

class MyAgent(Agent):
    def get_system_prompt(self) -> str:
        return "Your agent's instructions..."

    def get_agent_name(self) -> str:
        return "My Agent"

    def get_greeting(self) -> str:
        return "Hello! I'm your agent."
```

### 2. **Create MCP Server**

Use the server factory:

```python
from agent_framework.server import create_mcp_server
from agent_framework.tools import fetch_web_content

server = create_mcp_server("my-agent")

server.register_tool(
    name="fetch_web_content",
    description="Fetch web content",
    input_schema={...},
    handler=fetch_web_content,
)

server.setup_handlers()
await server.run()
```

### 3. **Use Built-in Tools**

The framework provides:
- Web reading
- Memory storage/retrieval
- Slack notifications
- Token storage

Just import and register them in your server!

### 4. **Add Custom Tools**

Create async functions that return dicts:

```python
async def my_tool(param: str) -> dict:
    return {"result": param.upper()}

server.register_tool(
    name="my_tool",
    description="Converts to uppercase",
    input_schema={...},
    handler=my_tool,
)
```

## Extending the Framework

When building domain-specific agents, create separate packages that extend the framework:

### Example Extension Structure

```
my-agent/                      # Extension package
├── my_agent/
│   ├── agent.py              # MyAgent(Agent) - extends framework
│   ├── prompts.py            # Domain-specific system prompts
│   └── tools/
│       ├── custom_tool1.py   # Domain-specific tools
│       ├── custom_tool2.py
│       └── custom_tool3.py
└── pyproject.toml            # Depends on: agent-framework
```

**Your custom agent would:**
1. Import `Agent` from `agent_framework`
2. Extend it with domain-specific system prompt
3. Create MCP server using `agent_framework.server`
4. Register domain-specific tools alongside generic ones

### Example Implementation

```python
# my_agent/agent.py
from agent_framework import Agent
from .prompts import SYSTEM_PROMPT

class MyAgent(Agent):
    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_agent_name(self) -> str:
        return "My Custom Agent"
```

**Dependencies:**
```toml
# pyproject.toml
dependencies = [
    "agent-framework @ file:///path/to/agent-framework",
    # ... domain-specific deps
]
```

**Using framework tools:**
```python
from agent_framework.tools import (
    fetch_web_content,
    save_memory,
    get_memories,
    send_slack_message,
)
```

## Key Design Decisions

1. **MCP Protocol**: Clean separation between agent and tools
2. **Stdio Transport**: Simple, local, easy to debug
3. **Hot Reload**: Reconnect pattern enables tool updates without losing conversation
4. **File-based Storage**: Easy to start, clear migration path to databases
5. **Pydantic Everywhere**: Type safety and validation
6. **Abstract Base Classes**: Enforces extension points (system prompt, etc.)
7. **Error Categories**: Distinguishes validation, auth, and execution errors

## Benefits

- **Code Reuse**: Generic tools work across all agents
- **Consistency**: All agents share the same architecture
- **Maintainability**: Fix once, benefits all agents
- **Testability**: Framework components can be tested independently
- **Extensibility**: Easy to add new tools and capabilities
- **Documentation**: Generic patterns are well-documented

## Potential Enhancements

1. Add database storage backends
2. Add more generic tools (email, calendar, etc.)
3. Add testing utilities and fixtures
4. Improve error handling and logging
5. Add support for streaming responses
6. Publish to PyPI

## Questions?

- Check `QUICKSTART.md` for usage guide
- Read the source - it's well-documented!
- All code patterns are from production agents
