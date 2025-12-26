"""Tests for MCP format converters."""

from agentring.mcp.formats import (
    to_json_schema, to_openapi_spec, to_function_spec,
    to_crewai_tool, to_langchain_tool, convert_tool_format
)
from agentring.mcp.types import ToolDefinition


class TestFormatConverters:
    """Tests for format conversion functions."""

    def test_to_json_schema(self):
        """Test JSON schema conversion."""
        definition = ToolDefinition(
            name="reset_env",
            description="Reset environment",
            parameters={
                "type": "object",
                "properties": {"seed": {"type": "integer"}},
                "required": ["seed"]
            },
            server_url="http://localhost:8070"
        )

        schema = to_json_schema(definition)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "reset_env"
        assert schema["function"]["description"] == "Reset environment"
        assert "seed" in schema["function"]["parameters"]["properties"]

    def test_to_openapi_spec(self):
        """Test OpenAPI spec conversion."""
        definition = ToolDefinition(
            name="reset_env",
            description="Reset environment",
            parameters={
                "type": "object",
                "properties": {"seed": {"type": "integer"}},
                "required": ["seed"]
            },
            server_url="http://localhost:8070"
        )

        spec = to_openapi_spec(definition)

        assert "summary" in spec
        assert spec["operationId"] == "reset_env"
        assert "parameters" in spec
        assert len(spec["parameters"]) == 1
        assert spec["parameters"][0]["name"] == "seed"

    def test_to_function_spec(self):
        """Test function spec conversion."""
        definition = ToolDefinition(
            name="reset_env",
            description="Reset environment",
            parameters={"type": "object"},
            server_url="http://localhost:8070"
        )

        spec = to_function_spec(definition)

        assert spec["name"] == "reset_env"
        assert spec["description"] == "Reset environment"
        assert spec["parameters"] == {"type": "object"}

    def test_sdk_specific_formats(self):
        """Test SDK-specific format conversions."""
        definition = ToolDefinition(
            name="reset_env",
            description="Reset environment",
            parameters={"type": "object"},
            server_url="http://localhost:8070"
        )

        # Test CrewAI format
        crewai_spec = to_crewai_tool(definition)
        assert crewai_spec["name"] == "reset_env"
        assert "args_schema" in crewai_spec

        # Test LangChain format
        langchain_spec = to_langchain_tool(definition)
        assert langchain_spec["name"] == "reset_env"
        assert "args_schema" in langchain_spec

    def test_convert_tool_format(self):
        """Test the convert_tool_format function."""
        definition = ToolDefinition(
            name="reset_env",
            description="Reset environment",
            parameters={"type": "object"},
            server_url="http://localhost:8070"
        )

        # Test valid format
        schema = convert_tool_format(definition, "json_schema")
        assert schema["type"] == "function"

        # Test invalid format
        with pytest.raises(ValueError, match="Unsupported format"):
            convert_tool_format(definition, "invalid_format")
