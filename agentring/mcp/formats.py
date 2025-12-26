"""Format converters for ToolDefinition objects."""

from __future__ import annotations

from typing import Any, Dict

from agentring.mcp.types import ToolDefinition


def to_json_schema(tool_definition: ToolDefinition) -> Dict[str, Any]:
    """
    Convert ToolDefinition to JSON Schema format.

    This format is commonly used by OpenAI-style APIs and some agent frameworks.

    Args:
        tool_definition: ToolDefinition to convert

    Returns:
        Dictionary in JSON Schema format
    """
    return {
        "type": "function",
        "function": {
            "name": tool_definition.name,
            "description": tool_definition.description,
            "parameters": tool_definition.parameters
        }
    }


def to_openapi_spec(tool_definition: ToolDefinition) -> Dict[str, Any]:
    """
    Convert ToolDefinition to OpenAPI specification format.

    This format is used by REST API documentation and some agent frameworks.

    Args:
        tool_definition: ToolDefinition to convert

    Returns:
        Dictionary in OpenAPI specification format
    """
    # Convert JSON Schema parameters to OpenAPI format
    parameters = []
    if isinstance(tool_definition.parameters, dict):
        properties = tool_definition.parameters.get("properties", {})
        required = tool_definition.parameters.get("required", [])

        for param_name, param_schema in properties.items():
            param_spec = {
                "name": param_name,
                "in": "query",  # Default to query parameters
                "description": param_schema.get("description", ""),
                "required": param_name in required,
                "schema": _json_schema_to_openapi_schema(param_schema)
            }
            parameters.append(param_spec)

    return {
        "summary": tool_definition.description,
        "operationId": tool_definition.name,
        "parameters": parameters,
        "responses": {
            "200": {
                "description": "Successful tool execution",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "result": {"type": "object"},
                                "error": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    }


def to_function_spec(tool_definition: ToolDefinition) -> Dict[str, Any]:
    """
    Convert ToolDefinition to function specification format.

    This is a simplified format used by some agent frameworks.

    Args:
        tool_definition: ToolDefinition to convert

    Returns:
        Dictionary in function specification format
    """
    return {
        "name": tool_definition.name,
        "description": tool_definition.description,
        "parameters": tool_definition.parameters
    }


def to_crewai_tool(tool_definition: ToolDefinition) -> Dict[str, Any]:
    """
    Convert ToolDefinition to CrewAI tool format.

    CrewAI uses a specific format for tool definitions.

    Args:
        tool_definition: ToolDefinition to convert

    Returns:
        Dictionary compatible with CrewAI tool format
    """
    # CrewAI expects tools to be defined as functions with decorators
    # This returns the metadata that would be used with @tool decorator
    return {
        "name": tool_definition.name,
        "description": tool_definition.description,
        "args_schema": tool_definition.parameters,
        "func": None  # Would be set by the actual function
    }


def to_langchain_tool(tool_definition: ToolDefinition) -> Dict[str, Any]:
    """
    Convert ToolDefinition to LangChain tool format.

    LangChain uses StructuredTool format.

    Args:
        tool_definition: ToolDefinition to convert

    Returns:
        Dictionary compatible with LangChain tool format
    """
    return {
        "name": tool_definition.name,
        "description": tool_definition.description,
        "args_schema": tool_definition.parameters,
        "func": None,  # Would be set by the actual function
        "return_direct": False
    }


def to_google_adk_tool(tool_definition: ToolDefinition) -> Dict[str, Any]:
    """
    Convert ToolDefinition to Google ADK tool format.

    Google ADK uses FunctionTool format.

    Args:
        tool_definition: ToolDefinition to convert

    Returns:
        Dictionary compatible with Google ADK tool format
    """
    return {
        "name": tool_definition.name,
        "description": tool_definition.description,
        "parameters": tool_definition.parameters
    }


def to_openai_tool(tool_definition: ToolDefinition) -> Dict[str, Any]:
    """
    Convert ToolDefinition to OpenAI tool format.

    OpenAI uses a specific format for function calling.

    Args:
        tool_definition: ToolDefinition to convert

    Returns:
        Dictionary in OpenAI tool format
    """
    return {
        "type": "function",
        "function": {
            "name": tool_definition.name,
            "description": tool_definition.description,
            "parameters": tool_definition.parameters
        }
    }


def to_letta_tool(tool_definition: ToolDefinition) -> Dict[str, Any]:
    """
    Convert ToolDefinition to Letta tool format.

    Letta uses JSON schema format for tool definitions.

    Args:
        tool_definition: ToolDefinition to convert

    Returns:
        Dictionary in Letta tool format
    """
    return {
        "name": tool_definition.name,
        "description": tool_definition.description,
        "parameters": tool_definition.parameters,
        "function": {
            "name": tool_definition.name,
            "parameters": tool_definition.parameters
        }
    }


def _json_schema_to_openapi_schema(json_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert JSON Schema type to OpenAPI schema format.

    Args:
        json_schema: JSON Schema definition

    Returns:
        OpenAPI schema definition
    """
    schema_type = json_schema.get("type")

    if schema_type == "string":
        result = {"type": "string"}
        if "enum" in json_schema:
            result["enum"] = json_schema["enum"]
        if "minLength" in json_schema:
            result["minLength"] = json_schema["minLength"]
        if "maxLength" in json_schema:
            result["maxLength"] = json_schema["maxLength"]
        return result

    elif schema_type == "integer":
        result = {"type": "integer"}
        if "minimum" in json_schema:
            result["minimum"] = json_schema["minimum"]
        if "maximum" in json_schema:
            result["maximum"] = json_schema["maximum"]
        return result

    elif schema_type == "number":
        result = {"type": "number"}
        if "minimum" in json_schema:
            result["minimum"] = json_schema["minimum"]
        if "maximum" in json_schema:
            result["maximum"] = json_schema["maximum"]
        return result

    elif schema_type == "boolean":
        return {"type": "boolean"}

    elif schema_type == "array":
        result = {"type": "array"}
        if "items" in json_schema:
            result["items"] = _json_schema_to_openapi_schema(json_schema["items"])
        return result

    elif schema_type == "object":
        result = {"type": "object"}
        if "properties" in json_schema:
            result["properties"] = {
                k: _json_schema_to_openapi_schema(v)
                for k, v in json_schema["properties"].items()
            }
        if "required" in json_schema:
            result["required"] = json_schema["required"]
        return result

    else:
        # Default fallback
        return {"type": "string"}


# Registry of format converters
FORMAT_CONVERTERS = {
    "json_schema": to_json_schema,
    "openapi": to_openapi_spec,
    "function_spec": to_function_spec,
    "crewai": to_crewai_tool,
    "langchain": to_langchain_tool,
    "google_adk": to_google_adk_tool,
    "openai": to_openai_tool,
    "letta": to_letta_tool,
}


def convert_tool_format(
    tool_definition: ToolDefinition,
    format_name: str
) -> Dict[str, Any]:
    """
    Convert ToolDefinition to a specific format.

    Args:
        tool_definition: ToolDefinition to convert
        format_name: Name of the target format

    Returns:
        Dictionary in the requested format

    Raises:
        ValueError: If the format is not supported
    """
    if format_name not in FORMAT_CONVERTERS:
        available_formats = list(FORMAT_CONVERTERS.keys())
        raise ValueError(
            f"Unsupported format '{format_name}'. "
            f"Available formats: {available_formats}"
        )

    converter = FORMAT_CONVERTERS[format_name]
    return converter(tool_definition)
