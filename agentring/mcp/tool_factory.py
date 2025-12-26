"""Generic MCP Tool Factory - Create callable tools from MCP servers."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from agentring.mcp.client import MCPServerClient
from agentring.mcp.discovery import discover_tools
from agentring.mcp.types import ToolCallable, ToolDefinition


class MCPToolFactory:
    """
    Factory for creating callable tools from MCP server tool definitions.

    This factory generates Python functions that can be called directly or
    adapted to work with any agent SDK. The generated tools handle all HTTP
    communication, serialization, and error handling internally.
    """

    def __init__(self, server_url: str, client: Optional[MCPServerClient] = None):
        """
        Initialize tool factory.

        Args:
            server_url: URL of the MCP server
            client: Optional pre-configured MCPServerClient instance
        """
        self.server_url = server_url
        self.client = client or MCPServerClient(server_url)
        self._tool_definitions: Optional[List[ToolDefinition]] = None
        self._callable_tools: Optional[Dict[str, ToolCallable]] = None

    def get_tool_definitions(self, refresh: bool = False) -> List[ToolDefinition]:
        """
        Get tool definitions from the MCP server.

        Args:
            refresh: Force refresh of cached definitions

        Returns:
            List of ToolDefinition objects
        """
        if self._tool_definitions is None or refresh:
            self._tool_definitions = discover_tools(self.server_url, self.client)
        return self._tool_definitions

    def create_callable_tool(self, tool_definition: ToolDefinition) -> ToolCallable:
        """
        Create a callable Python function from a tool definition.

        Args:
            tool_definition: ToolDefinition to convert

        Returns:
            Callable Python function
        """
        def tool_callable(*args, **kwargs) -> Any:
            """Dynamically generated tool function."""
            # Convert positional args to keyword args based on parameter schema
            params = self._args_to_params(tool_definition, args, kwargs)

            # Validate parameters
            self._validate_params(tool_definition, params)

            # Call the tool
            try:
                result = self.client.call_tool(tool_definition.name, params)
                return self._format_result(result)
            except Exception as e:
                # Re-raise with more context
                raise RuntimeError(
                    f"Tool '{tool_definition.name}' failed: {e}"
                ) from e

        # Set function metadata for better introspection
        tool_callable.__name__ = tool_definition.name
        tool_callable.__doc__ = tool_definition.description
        tool_callable.__annotations__ = self._extract_annotations(tool_definition)

        return tool_callable

    def create_tools(
        self,
        tool_names: Optional[List[str]] = None,
        refresh: bool = False
    ) -> List[ToolCallable]:
        """
        Create callable tools from MCP server.

        Args:
            tool_names: Optional list of tool names to create (None for all)
            refresh: Force refresh of tool definitions

        Returns:
            List of callable Python functions
        """
        definitions = self.get_tool_definitions(refresh=refresh)

        if tool_names:
            # Filter to requested tools
            name_set = set(tool_names)
            definitions = [d for d in definitions if d.name in name_set]

        return [self.create_callable_tool(tool_def) for tool_def in definitions]

    def get_tool_names(self, refresh: bool = False) -> List[str]:
        """
        Get names of available tools.

        Args:
            refresh: Force refresh of tool definitions

        Returns:
            List of tool names
        """
        definitions = self.get_tool_definitions(refresh=refresh)
        return [d.name for d in definitions]

    def _args_to_params(
        self,
        tool_definition: ToolDefinition,
        args: tuple,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert function arguments to tool parameters."""
        params = dict(kwargs)

        # If we have positional args, map them to parameter names
        if args:
            param_names = list(tool_definition.parameters.get("properties", {}).keys())
            for i, arg in enumerate(args):
                if i < len(param_names):
                    params[param_names[i]] = arg

        return params

    def _validate_params(self, tool_definition: ToolDefinition, params: Dict[str, Any]) -> None:
        """Basic parameter validation against JSON schema."""
        schema = tool_definition.parameters
        if not isinstance(schema, dict):
            return

        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required parameters
        for req_param in required:
            if req_param not in params:
                raise ValueError(f"Missing required parameter: {req_param}")

        # Basic type checking (could be enhanced)
        for param_name, param_value in params.items():
            if param_name in properties:
                param_schema = properties[param_name]
                self._validate_param_value(param_name, param_value, param_schema)

    def _validate_param_value(self, name: str, value: Any, schema: Dict[str, Any]) -> None:
        """Validate a single parameter value against its schema."""
        expected_type = schema.get("type")

        if expected_type == "string" and not isinstance(value, str):
            raise TypeError(f"Parameter '{name}' must be a string, got {type(value)}")
        elif expected_type == "integer" and not isinstance(value, int):
            raise TypeError(f"Parameter '{name}' must be an integer, got {type(value)}")
        elif expected_type == "number" and not isinstance(value, (int, float)):
            raise TypeError(f"Parameter '{name}' must be a number, got {type(value)}")
        elif expected_type == "boolean" and not isinstance(value, bool):
            raise TypeError(f"Parameter '{name}' must be a boolean, got {type(value)}")
        elif expected_type == "array" and not isinstance(value, (list, tuple)):
            raise TypeError(f"Parameter '{name}' must be an array, got {type(value)}")

        # Check enum values
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"Parameter '{name}' must be one of {schema['enum']}, got {value}")

    def _format_result(self, result: Dict[str, Any]) -> Any:
        """Format tool result for consumption."""
        # If the result has a "result" key, return that
        if "result" in result:
            return result["result"]

        # If it's a successful response with data, return the data
        if result.get("success") and "data" in result:
            return result["data"]

        # Otherwise return the full result
        return result

    def _extract_annotations(self, tool_definition: ToolDefinition) -> Dict[str, Any]:
        """Extract type annotations from tool definition."""
        annotations = {}
        properties = tool_definition.parameters.get("properties", {})

        for param_name, param_schema in properties.items():
            param_type = param_schema.get("type")
            if param_type == "string":
                annotations[param_name] = str
            elif param_type == "integer":
                annotations[param_name] = int
            elif param_type == "number":
                annotations[param_name] = Union[int, float]
            elif param_type == "boolean":
                annotations[param_name] = bool
            elif param_type == "array":
                annotations[param_name] = List

        return annotations


# Convenience functions
def create_tools(
    server_url: str,
    tool_names: Optional[List[str]] = None,
    client: Optional[MCPServerClient] = None
) -> List[ToolCallable]:
    """
    Create callable tools from an MCP server.

    This is the main entry point for creating tools from MCP servers.

    Args:
        server_url: URL of the MCP server
        tool_names: Optional list of tool names to create (None for all)
        client: Optional pre-configured MCPServerClient instance

    Returns:
        List of callable Python functions

    Example:
        ```python
        # Create all tools from a server
        tools = create_tools("http://localhost:8070")

        # Create specific tools
        tools = create_tools("http://localhost:8070", ["reset_env", "step_env"])

        # Use tools
        result = tools[0](seed=42)  # Call reset_env
        ```
    """
    factory = MCPToolFactory(server_url, client)
    return factory.create_tools(tool_names)


def create_tool(
    server_url: str,
    tool_name: str,
    client: Optional[MCPServerClient] = None
) -> ToolCallable:
    """
    Create a single callable tool from an MCP server.

    Args:
        server_url: URL of the MCP server
        tool_name: Name of the tool to create
        client: Optional pre-configured MCPServerClient instance

    Returns:
        Callable Python function

    Raises:
        ValueError: If the tool is not found
    """
    factory = MCPToolFactory(server_url, client)
    tools = factory.create_tools([tool_name])
    if not tools:
        raise ValueError(f"Tool '{tool_name}' not found on server {server_url}")
    return tools[0]
