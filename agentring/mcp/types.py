"""Type definitions for AgentRing MCP extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Union


@dataclass
class ToolDefinition:
    """Standardized tool metadata from MCP servers."""

    name: str
    """Tool name (e.g., 'reset_env', 'step_env')"""

    description: str
    """Tool description"""

    parameters: Dict[str, Any]
    """JSON schema for tool parameters"""

    server_url: str
    """MCP server URL where this tool is available"""

    def __post_init__(self):
        """Validate tool definition after creation."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not self.server_url:
            raise ValueError("Server URL cannot be empty")


class ToolCallable(Protocol):
    """Protocol for tool callables."""

    def __call__(self, *args, **kwargs) -> Any:
        """Call the tool with given arguments."""
        ...


# Type aliases
ToolResult = Dict[str, Any]
"""Result from calling a tool"""

ToolList = List[Union[ToolDefinition, ToolCallable]]
"""List of tools (definitions or callables)"""

AgentInterface = Callable[[str], str]
"""Agent interface that takes a prompt string and returns a response string"""

AsyncAgentInterface = Callable[[str], Any]  # Can return str or coroutine
"""Async agent interface (may return coroutine)"""


@dataclass
class EpisodeResult:
    """Result from running a single episode."""

    episode_num: int
    total_reward: float
    num_steps: int
    success: bool
    observation: Optional[Any] = None
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Check if episode was successful."""
        return self.success and self.error is None


@dataclass
class ServerInfo:
    """Information about an MCP server."""

    url: str
    name: Optional[str] = None
    version: Optional[str] = None
    tools_available: Optional[List[str]] = None
    last_health_check: Optional[float] = None
    is_healthy: bool = False
