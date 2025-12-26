"""Utility functions for MCP tools and agents."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from agentring.mcp.types import ToolCallable, ToolDefinition


def compose_tools(*tool_lists: List[Union[ToolDefinition, ToolCallable]]) -> List[Union[ToolDefinition, ToolCallable]]:
    """
    Compose multiple tool lists into a single list.

    Removes duplicates based on tool name/function name.

    Args:
        *tool_lists: Variable number of tool lists to combine

    Returns:
        Combined list of unique tools
    """
    seen_names = set()
    composed = []

    for tool_list in tool_lists:
        for tool in tool_list:
            # Get tool name
            if isinstance(tool, ToolDefinition):
                name = tool.name
            else:
                # Assume it's a callable
                name = getattr(tool, '__name__', str(id(tool)))

            if name not in seen_names:
                seen_names.add(name)
                composed.append(tool)

    return composed


def filter_tools(
    tools: List[Union[ToolDefinition, ToolCallable]],
    names: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> List[Union[ToolDefinition, ToolCallable]]:
    """
    Filter tools by various criteria.

    Args:
        tools: List of tools to filter
        names: Specific tool names to include (if provided, other filters ignored)
        include_patterns: Patterns to include (substring match)
        exclude_patterns: Patterns to exclude (substring match)

    Returns:
        Filtered list of tools
    """
    if names:
        # Filter by exact names
        name_set = set(names)
        return [
            tool for tool in tools
            if _get_tool_name(tool) in name_set
        ]

    filtered = tools

    # Apply include patterns
    if include_patterns:
        filtered = [
            tool for tool in filtered
            if any(pattern in _get_tool_name(tool) for pattern in include_patterns)
        ]

    # Apply exclude patterns
    if exclude_patterns:
        filtered = [
            tool for tool in filtered
            if not any(pattern in _get_tool_name(tool) for pattern in exclude_patterns)
        ]

    return filtered


def validate_tool_call(
    tool: Union[ToolDefinition, ToolCallable],
    args: Dict[str, Any]
) -> tuple[bool, Optional[str]]:
    """
    Validate that arguments are compatible with a tool.

    Args:
        tool: Tool to validate against
        args: Arguments to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if isinstance(tool, ToolDefinition):
        return _validate_tool_definition_args(tool, args)
    else:
        # For callable tools, we can try to inspect the signature
        return _validate_callable_args(tool, args)


def format_tool_result(result: Any, tool_name: Optional[str] = None) -> str:
    """
    Format a tool result for display/logging.

    Args:
        result: Tool execution result
        tool_name: Optional tool name for context

    Returns:
        Formatted string representation
    """
    prefix = f"[{tool_name}] " if tool_name else ""

    if isinstance(result, dict):
        if result.get("success") is False:
            error = result.get("error", "Unknown error")
            return f"{prefix}ERROR: {error}"

        # Format successful results
        parts = []
        for key, value in result.items():
            if key == "success":
                continue
            elif key == "observation" and isinstance(value, str) and len(value) > 100:
                # Truncate long observations
                parts.append(f"{key}: {value[:100]}...")
            else:
                parts.append(f"{key}: {value}")

        return f"{prefix}{' | '.join(parts)}"

    elif isinstance(result, str):
        return f"{prefix}{result}"

    else:
        return f"{prefix}{repr(result)}"


def group_tools_by_server(tools: List[Union[ToolDefinition, ToolCallable]]) -> Dict[str, List[Union[ToolDefinition, ToolCallable]]]:
    """
    Group tools by their server URL.

    Args:
        tools: List of tools to group

    Returns:
        Dictionary mapping server URLs to tool lists
    """
    groups = {}

    for tool in tools:
        if isinstance(tool, ToolDefinition):
            server_url = tool.server_url
        else:
            # For callables, we can't determine server URL
            server_url = "unknown"

        if server_url not in groups:
            groups[server_url] = []
        groups[server_url].append(tool)

    return groups


def get_tool_names(tools: List[Union[ToolDefinition, ToolCallable]]) -> List[str]:
    """
    Extract tool names from a list of tools.

    Args:
        tools: List of tools

    Returns:
        List of tool names
    """
    return [_get_tool_name(tool) for tool in tools]


def find_tool_by_name(
    tools: List[Union[ToolDefinition, ToolCallable]],
    name: str
) -> Optional[Union[ToolDefinition, ToolCallable]]:
    """
    Find a tool by name.

    Args:
        tools: List of tools to search
        name: Tool name to find

    Returns:
        Tool if found, None otherwise
    """
    for tool in tools:
        if _get_tool_name(tool) == name:
            return tool
    return None


def _get_tool_name(tool: Union[ToolDefinition, ToolCallable]) -> str:
    """Get the name of a tool."""
    if isinstance(tool, ToolDefinition):
        return tool.name
    else:
        return getattr(tool, '__name__', str(id(tool)))


def _validate_tool_definition_args(tool_def: ToolDefinition, args: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate arguments against a ToolDefinition."""
    schema = tool_def.parameters
    if not isinstance(schema, dict):
        return True, None  # No schema to validate against

    # Check required parameters
    required = schema.get("required", [])
    for req_param in required:
        if req_param not in args:
            return False, f"Missing required parameter: {req_param}"

    # Check parameter types
    properties = schema.get("properties", {})
    for param_name, param_value in args.items():
        if param_name in properties:
            param_schema = properties[param_name]
            is_valid, error = _validate_param_value(param_name, param_value, param_schema)
            if not is_valid:
                return False, error

    return True, None


def _validate_callable_args(callable_tool: ToolCallable, args: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate arguments against a callable tool's signature."""
    try:
        import inspect
        sig = inspect.signature(callable_tool)

        # Check if all provided args are accepted by the function
        for param_name in args:
            if param_name not in sig.parameters:
                return False, f"Unexpected parameter: {param_name}"

        # Check required parameters
        for param_name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty and param_name not in args:
                return False, f"Missing required parameter: {param_name}"

        return True, None

    except Exception as e:
        # If we can't inspect the signature, assume it's valid
        return True, None


def _validate_param_value(param_name: str, value: Any, schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate a single parameter value against its schema."""
    expected_type = schema.get("type")

    if expected_type == "string" and not isinstance(value, str):
        return False, f"Parameter '{param_name}' must be a string, got {type(value)}"
    elif expected_type == "integer" and not isinstance(value, int):
        return False, f"Parameter '{param_name}' must be an integer, got {type(value)}"
    elif expected_type == "number" and not isinstance(value, (int, float)):
        return False, f"Parameter '{param_name}' must be a number, got {type(value)}"
    elif expected_type == "boolean" and not isinstance(value, bool):
        return False, f"Parameter '{param_name}' must be a boolean, got {type(value)}"
    elif expected_type == "array" and not isinstance(value, (list, tuple)):
        return False, f"Parameter '{param_name}' must be an array, got {type(value)}"
    elif expected_type == "object" and not isinstance(value, dict):
        return False, f"Parameter '{param_name}' must be an object, got {type(value)}"

    # Check enum values
    if "enum" in schema and value not in schema["enum"]:
        return False, f"Parameter '{param_name}' must be one of {schema['enum']}, got {value}"

    # Check string length
    if expected_type == "string":
        min_len = schema.get("minLength")
        max_len = schema.get("maxLength")
        if min_len is not None and len(value) < min_len:
            return False, f"Parameter '{param_name}' must be at least {min_len} characters"
        if max_len is not None and len(value) > max_len:
            return False, f"Parameter '{param_name}' must be at most {max_len} characters"

    # Check numeric ranges
    if expected_type in ("integer", "number"):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            return False, f"Parameter '{param_name}' must be >= {minimum}"
        if maximum is not None and value > maximum:
            return False, f"Parameter '{param_name}' must be <= {maximum}"

    return True, None
