"""Error types for the agent framework."""


class AgentError(Exception):
    """Base exception for agent framework errors."""

    pass


class ValidationError(AgentError):
    """Raised when tool input validation fails."""

    pass


class AuthenticationError(AgentError):
    """Raised when authentication is required or fails."""

    pass


class ToolExecutionError(AgentError):
    """Raised when tool execution fails."""

    pass
