# Agent Framework

A production-ready framework for building LLM agents with the Model Context Protocol (MCP). Build powerful, tool-enabled agents with persistent memory, OAuth integration, and extensible architecture.

## Why This Framework?

- **Battle-tested**: Extracted from production agent implementations
- **MCP-native**: Built on the Model Context Protocol for clean tool separation
- **Batteries included**: Web scraping, memory storage, Slack integration, OAuth handling
- **Type-safe**: Full typing with Pydantic validation throughout
- **Extensible**: Clean abstractions for building domain-specific agents

## Quick Install

```bash
# Using uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Example

```python
from agent_framework import Agent

class MyAgent(Agent):
    def get_system_prompt(self) -> str:
        return "You are a helpful assistant that can read web content and remember information."

# Run it
import asyncio
asyncio.run(MyAgent(mcp_server_path="server.py").start())
```

## Core Features

**Agent System**
- Agentic conversation loop with CLI interface
- Local MCP client for stdio-based tool servers
- Remote MCP client for HTTPS-based servers with OAuth
- Token usage tracking

**Remote MCP & OAuth**
- Connect to remote MCP servers over HTTPS
- Full OAuth 2.0 with PKCE support
- Automatic OAuth discovery via .well-known endpoints
- Dynamic client registration
- Automatic token refresh

**Storage & Memory**
- Persistent memory with categories, tags, and search
- Encrypted OAuth token storage with auto-refresh
- File-based with easy database migration path

**Built-in Tools**
- Web content reader (HTML to markdown)
- Slack webhooks integration
- Memory management (save, retrieve, search)

**MCP Server Infrastructure**
- Simple tool registration
- Structured error handling
- JSON schema validation

## Documentation

- **[Getting Started](GETTING_STARTED.md)** - Installation, configuration, and first agent
- **[Architecture](ARCHITECTURE.md)** - Design decisions, extension patterns, and advanced topics

## Project Structure

```
agent_framework/
├── core/          # Agent base class, local and remote MCP clients
├── oauth/         # OAuth 2.0 discovery, flow, and token management
├── tools/         # Reusable tools (web, slack, memory)
├── storage/       # Memory and token storage backends
├── server/        # MCP server infrastructure
└── utils/         # Errors, config, logging
```

## License

MIT
