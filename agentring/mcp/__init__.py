"""AgentRing MCP Extensions - Generic MCP tool generation and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

__version__ = "0.1.0"

# Import and re-export main functionality
from agentring.mcp.client import MCPServerClient
from agentring.mcp.multi_server import MultiServerClient
from agentring.mcp.results import EpisodeResults
from agentring.mcp.runner import MCPAgentRunner
from agentring.mcp.tool_factory import create_tools
from agentring.mcp.types import EpisodeResult, ToolDefinition

# Import utility modules for direct access
from agentring.mcp import formats, templates, utils

__all__ = [
    # Main classes
    "MCPServerClient",
    "MCPAgentRunner",
    "MultiServerClient",
    "EpisodeResults",

    # Main functions
    "create_tools",

    # Types
    "EpisodeResult",
    "ToolDefinition",

    # Utility modules
    "formats",
    "templates",
    "utils",
]
