"""Configuration management for agents and MCP servers."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # LLM Configuration
    anthropic_api_key: str | None = Field(
        default=None, description="Anthropic API key for Claude"
    )

    # MCP Server Configuration
    mcp_server_host: str = Field(default="localhost", description="MCP server host")
    mcp_server_port: int = Field(default=8000, description="MCP server port")

    # Token Storage
    token_storage_path: Path = Field(
        default=Path("./tokens"), description="Path to store OAuth tokens"
    )
    token_encryption_key: str | None = Field(
        default=None, description="Key for encrypting stored tokens"
    )

    # Memory Storage
    memory_storage_path: Path = Field(
        default=Path("./memories"), description="Path to store memories"
    )

    # Slack Integration
    slack_webhook_url: str | None = Field(
        default=None, description="Default Slack incoming webhook URL"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="agent.log", description="Log file path")

    def __init__(self, **kwargs):
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        # Ensure storage directories exist
        self.token_storage_path.mkdir(parents=True, exist_ok=True)
        self.memory_storage_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
