"""Tool discovery for MCP servers."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from agentring.mcp.client import MCPServerClient
from agentring.mcp.types import ToolDefinition


def discover_tools(
    server_url: str,
    client: Optional[MCPServerClient] = None,
    use_cache: bool = True
) -> List[ToolDefinition]:
    """
    Discover available tools from an MCP server.

    Args:
        server_url: URL of the MCP server
        client: Optional pre-configured MCPServerClient instance
        use_cache: Whether to use cached server information

    Returns:
        List of discovered ToolDefinition objects

    Raises:
        RuntimeError: If tool discovery fails
    """
    if client is None:
        client = MCPServerClient(server_url)

    tools = []

    # Try MCP protocol first
    try:
        mcp_tools = _discover_via_mcp(client)
        tools.extend(mcp_tools)
    except Exception:
        # Fall back to REST API
        try:
            rest_tools = _discover_via_rest(client)
            tools.extend(rest_tools)
        except Exception as e:
            raise RuntimeError(f"Failed to discover tools from {server_url}: {e}") from e

    return tools


def _discover_via_mcp(client: MCPServerClient) -> List[ToolDefinition]:
    """Discover tools using MCP protocol (/mcp/v1/tools/list)."""
    try:
        # Try the MCP tools list endpoint
        response = client.client.get(f"{client.server_url}/mcp/v1/tools/list")
        response.raise_for_status()
        data = response.json()

        tools = []
        for tool_data in data.get("tools", []):
            try:
                tool_def = _parse_mcp_tool_definition(tool_data, client.server_url)
                tools.append(tool_def)
            except (KeyError, ValueError, TypeError):
                # Skip malformed tool definitions
                continue

        return tools

    except Exception:
        # If MCP endpoint fails, try fallback approach
        return _discover_via_mcp_fallback(client)


def _discover_via_mcp_fallback(client: MCPServerClient) -> List[ToolDefinition]:
    """Fallback MCP discovery using known tool patterns."""
    # For gym-mcp-server, we know the standard tools
    standard_tools = [
        {
            "name": "reset_env",
            "description": "Reset the environment to start a new episode",
            "parameters": {
                "type": "object",
                "properties": {
                    "seed": {
                        "type": ["integer", "null"],
                        "description": "Optional random seed for reproducibility"
                    }
                }
            }
        },
        {
            "name": "step_env",
            "description": "Take an action in the environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action to execute"
                    }
                },
                "required": ["action"]
            }
        },
        {
            "name": "get_env_info",
            "description": "Get information about the environment",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "render_env",
            "description": "Render the environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Render mode (e.g., 'rgb_array', 'human')"
                    }
                }
            }
        },
        {
            "name": "close_env",
            "description": "Close the environment and free resources",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    ]

    tools = []
    for tool_data in standard_tools:
        try:
            tool_def = _parse_mcp_tool_definition(tool_data, client.server_url)
            tools.append(tool_def)
        except (KeyError, ValueError, TypeError):
            continue

    return tools


def _discover_via_rest(client: MCPServerClient) -> List[ToolDefinition]:
    """Discover tools by introspecting REST API responses."""
    tools = []

    # Get environment info to understand what tools are available
    try:
        env_info = client.get_server_info()
        if not env_info:
            return tools
    except Exception:
        return tools

    # Based on the environment type, infer available tools
    # This is a heuristic approach since REST API doesn't expose tool schemas directly

    # All environments should have these basic tools
    base_tools = [
        ToolDefinition(
            name="reset_env",
            description="Reset the environment to start a new episode",
            parameters={
                "type": "object",
                "properties": {
                    "seed": {
                        "type": ["integer", "null"],
                        "description": "Optional random seed for reproducibility"
                    }
                }
            },
            server_url=client.server_url
        ),
        ToolDefinition(
            name="get_env_info",
            description="Get information about the environment",
            parameters={
                "type": "object",
                "properties": {}
            },
            server_url=client.server_url
        ),
        ToolDefinition(
            name="close_env",
            description="Close the environment and free resources",
            parameters={
                "type": "object",
                "properties": {}
            },
            server_url=client.server_url
        )
    ]

    tools.extend(base_tools)

    # Add step_env tool (action space dependent)
    action_space = env_info.get("action_space", {})
    if isinstance(action_space, dict):
        action_schema = _infer_action_schema(action_space)
        tools.append(ToolDefinition(
            name="step_env",
            description="Take an action in the environment",
            parameters={
                "type": "object",
                "properties": {
                    "action": action_schema
                },
                "required": ["action"]
            },
            server_url=client.server_url
        ))

    # Add render_env if rendering is supported
    if env_info.get("render_modes"):
        tools.append(ToolDefinition(
            name="render_env",
            description="Render the environment",
            parameters={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "description": "Render mode",
                        "enum": env_info["render_modes"]
                    }
                }
            },
            server_url=client.server_url
        ))

    return tools


def _parse_mcp_tool_definition(tool_data: Dict, server_url: str) -> ToolDefinition:
    """Parse MCP tool definition into ToolDefinition object."""
    return ToolDefinition(
        name=tool_data["name"],
        description=tool_data.get("description", ""),
        parameters=tool_data.get("parameters", {}),
        server_url=server_url
    )


def _infer_action_schema(action_space: Dict) -> Dict:
    """Infer JSON schema for action space."""
    space_type = action_space.get("type", "")

    if space_type == "Discrete":
        return {
            "type": "integer",
            "minimum": action_space.get("start", 0),
            "maximum": action_space.get("n", 1) - 1 + action_space.get("start", 0),
            "description": f"Discrete action (0-{action_space.get('n', 1) - 1})"
        }

    elif space_type == "Box":
        return {
            "type": "array",
            "items": {"type": "number"},
            "description": "Continuous action vector"
        }

    elif space_type == "MultiBinary":
        return {
            "type": "array",
            "items": {"type": "integer", "enum": [0, 1]},
            "description": "Binary action vector"
        }

    elif space_type == "MultiDiscrete":
        return {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Multi-discrete action vector"
        }

    else:
        # Default to string for text-based environments
        return {
            "type": "string",
            "description": "Action command"
        }
