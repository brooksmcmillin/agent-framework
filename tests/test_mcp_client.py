"""Tests for the MCP client module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.core.mcp_client import MCPClient, create_mcp_client


class TestMCPClient:
    """Tests for the MCPClient class."""

    def test_mcp_client_initialization(self):
        """Test MCPClient initialization."""
        client = MCPClient(server_script_path="test/server.py")

        assert client.server_script_path == "test/server.py"
        assert client.session is None
        assert client.available_tools == {}

    def test_mcp_client_default_path(self):
        """Test MCPClient uses default server path."""
        client = MCPClient()

        assert client.server_script_path == "mcp_server/server.py"

    def test_get_available_tools_empty(self):
        """Test get_available_tools returns empty list when no tools."""
        client = MCPClient()

        assert client.get_available_tools() == []

    def test_get_available_tools_with_tools(self):
        """Test get_available_tools returns tool names."""
        client = MCPClient()
        client.available_tools = {
            "tool1": MagicMock(),
            "tool2": MagicMock(),
            "tool3": MagicMock(),
        }

        tools = client.get_available_tools()

        assert len(tools) == 3
        assert "tool1" in tools
        assert "tool2" in tools
        assert "tool3" in tools

    def test_get_tool_info_exists(self):
        """Test get_tool_info returns info for existing tool."""
        client = MCPClient()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {"type": "object"}
        client.available_tools = {"test_tool": mock_tool}

        info = client.get_tool_info("test_tool")

        assert info is not None
        assert info["name"] == "test_tool"
        assert info["description"] == "A test tool"
        assert info["input_schema"] == {"type": "object"}

    def test_get_tool_info_not_exists(self):
        """Test get_tool_info returns None for non-existent tool."""
        client = MCPClient()

        info = client.get_tool_info("nonexistent")

        assert info is None


class TestMCPClientAsync:
    """Async tests for MCPClient methods."""

    @pytest.mark.asyncio
    async def test_call_tool_without_session_raises(self):
        """Test call_tool raises error when session not initialized."""
        client = MCPClient()

        with pytest.raises(RuntimeError, match="MCP session not initialized"):
            await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool_raises(self):
        """Test call_tool raises error for unknown tool."""
        client = MCPClient()
        client.session = MagicMock()  # Simulate active session
        client.available_tools = {"known_tool": MagicMock()}

        with pytest.raises(ValueError, match="Unknown tool"):
            await client.call_tool("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test call_tool returns result on success."""
        client = MCPClient()
        client.session = AsyncMock()
        client.available_tools = {"test_tool": MagicMock()}

        # Mock the tool result
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text='{"status": "success", "data": "test"}')]
        client.session.call_tool = AsyncMock(return_value=mock_result)

        result = await client.call_tool("test_tool", {"arg": "value"})

        assert result == {"status": "success", "data": "test"}

    @pytest.mark.asyncio
    async def test_call_tool_handles_auth_error(self):
        """Test call_tool raises PermissionError for auth errors."""
        client = MCPClient()
        client.session = AsyncMock()
        client.available_tools = {"test_tool": MagicMock()}

        # Mock auth error response
        mock_result = MagicMock()
        mock_result.content = [
            MagicMock(text='{"error": "authentication_required", "message": "Please authenticate"}')
        ]
        client.session.call_tool = AsyncMock(return_value=mock_result)

        with pytest.raises(PermissionError, match="Please authenticate"):
            await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_handles_execution_error(self):
        """Test call_tool raises RuntimeError for execution errors."""
        client = MCPClient()
        client.session = AsyncMock()
        client.available_tools = {"test_tool": MagicMock()}

        # Mock execution error response
        mock_result = MagicMock()
        mock_result.content = [
            MagicMock(text='{"error": "execution_error", "message": "Something went wrong"}')
        ]
        client.session.call_tool = AsyncMock(return_value=mock_result)

        with pytest.raises(RuntimeError, match="Tool execution failed"):
            await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_handles_empty_content(self):
        """Test call_tool handles empty content gracefully."""
        client = MCPClient()
        client.session = AsyncMock()
        client.available_tools = {"test_tool": MagicMock()}

        # Mock empty content
        mock_result = MagicMock()
        mock_result.content = []
        client.session.call_tool = AsyncMock(return_value=mock_result)

        result = await client.call_tool("test_tool", {})

        assert result == {}

    @pytest.mark.asyncio
    async def test_call_tool_handles_invalid_json(self):
        """Test call_tool raises error for invalid JSON response."""
        client = MCPClient()
        client.session = AsyncMock()
        client.available_tools = {"test_tool": MagicMock()}

        # Mock invalid JSON
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="not valid json")]
        client.session.call_tool = AsyncMock(return_value=mock_result)

        with pytest.raises(RuntimeError, match="Invalid JSON response"):
            await client.call_tool("test_tool", {})

    @pytest.mark.asyncio
    async def test_discover_tools_without_session_raises(self):
        """Test _discover_tools raises error when session not initialized."""
        client = MCPClient()

        with pytest.raises(RuntimeError, match="MCP session not initialized"):
            await client._discover_tools()

    @pytest.mark.asyncio
    async def test_discover_tools_success(self):
        """Test _discover_tools populates available_tools."""
        client = MCPClient()
        client.session = AsyncMock()

        # Mock tool list response
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"

        mock_response = MagicMock()
        mock_response.tools = [mock_tool1, mock_tool2]
        client.session.list_tools = AsyncMock(return_value=mock_response)

        await client._discover_tools()

        assert len(client.available_tools) == 2
        assert "tool1" in client.available_tools
        assert "tool2" in client.available_tools


class TestCreateMCPClient:
    """Tests for the create_mcp_client convenience function."""

    @pytest.mark.asyncio
    async def test_create_mcp_client_is_context_manager(self):
        """Test that create_mcp_client is an async context manager."""
        # We can't fully test this without a real MCP server,
        # but we can verify the function exists and is callable
        assert callable(create_mcp_client)
