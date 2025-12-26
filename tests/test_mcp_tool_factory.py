"""Tests for MCP tool factory."""

import pytest
from unittest.mock import Mock, patch

from agentring.mcp.tool_factory import MCPToolFactory, create_tools
from agentring.mcp.types import ToolDefinition


class TestMCPToolFactory:
    """Tests for MCPToolFactory."""

    def test_initialization(self):
        """Test factory initialization."""
        factory = MCPToolFactory("http://localhost:8070")
        assert factory.server_url == "http://localhost:8070"
        assert factory._tool_definitions is None

    @patch('agentring.mcp.tool_factory.discover_tools')
    def test_get_tool_definitions(self, mock_discover):
        """Test getting tool definitions."""
        mock_discover.return_value = [
            ToolDefinition("reset_env", "Reset", {}, "http://localhost:8070")
        ]

        factory = MCPToolFactory("http://localhost:8070")
        definitions = factory.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0].name == "reset_env"
        mock_discover.assert_called_once_with("http://localhost:8070", factory.client)

        # Test caching
        factory.get_tool_definitions()  # Should not call discover again
        assert mock_discover.call_count == 1

        # Test refresh
        factory.get_tool_definitions(refresh=True)
        assert mock_discover.call_count == 2

    def test_create_callable_tool(self):
        """Test creating callable tools."""
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

        factory = MCPToolFactory("http://localhost:8070")

        with patch.object(factory.client, 'call_tool', return_value={"success": True}):
            tool = factory.create_callable_tool(definition)

            assert callable(tool)
            assert tool.__name__ == "reset_env"
            assert "Reset environment" in tool.__doc__

            # Test calling the tool
            result = tool(seed=42)
            assert result == {"success": True}

    def test_create_tools(self):
        """Test creating multiple tools."""
        factory = MCPToolFactory("http://localhost:8070")

        with patch.object(factory, 'get_tool_definitions', return_value=[
            ToolDefinition("tool1", "Tool 1", {}, "http://localhost:8070"),
            ToolDefinition("tool2", "Tool 2", {}, "http://localhost:8070")
        ]):
            tools = factory.create_tools()

            assert len(tools) == 2
            assert all(callable(t) for t in tools)

    def test_create_tools_filtered(self):
        """Test creating filtered tools."""
        factory = MCPToolFactory("http://localhost:8070")

        with patch.object(factory, 'get_tool_definitions', return_value=[
            ToolDefinition("tool1", "Tool 1", {}, "http://localhost:8070"),
            ToolDefinition("tool2", "Tool 2", {}, "http://localhost:8070"),
            ToolDefinition("tool3", "Tool 3", {}, "http://localhost:8070")
        ]):
            tools = factory.create_tools(tool_names=["tool1", "tool3"])

            assert len(tools) == 2
            tool_names = [getattr(t, '__name__', '') for t in tools]
            assert "tool1" in tool_names
            assert "tool3" in tool_names
            assert "tool2" not in tool_names

    def test_get_tool_names(self):
        """Test getting tool names."""
        factory = MCPToolFactory("http://localhost:8070")

        with patch.object(factory, 'get_tool_definitions', return_value=[
            ToolDefinition("tool1", "Tool 1", {}, "http://localhost:8070"),
            ToolDefinition("tool2", "Tool 2", {}, "http://localhost:8070")
        ]):
            names = factory.get_tool_names()
            assert names == ["tool1", "tool2"]

    def test_validation(self):
        """Test parameter validation."""
        definition = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters={
                "type": "object",
                "properties": {
                    "required_param": {"type": "string"},
                    "optional_param": {"type": "integer"}
                },
                "required": ["required_param"]
            },
            server_url="http://localhost:8070"
        )

        factory = MCPToolFactory("http://localhost:8070")

        with patch.object(factory.client, 'call_tool', return_value={"success": True}):
            tool = factory.create_callable_tool(definition)

            # Valid call
            result = tool(required_param="test", optional_param=42)
            assert result == {"success": True}

            # Missing required parameter
            with pytest.raises(ValueError, match="Missing required parameter"):
                tool()

            # Wrong type
            with pytest.raises(TypeError, match="must be a string"):
                tool(required_param=123)


class TestCreateTools:
    """Tests for the create_tools convenience function."""

    @patch('agentring.mcp.tool_factory.MCPToolFactory')
    def test_create_tools_function(self, mock_factory_class):
        """Test create_tools convenience function."""
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        mock_factory.create_tools.return_value = [Mock()]

        tools = create_tools("http://localhost:8070", tool_names=["reset_env"])

        mock_factory_class.assert_called_once_with("http://localhost:8070", None)
        mock_factory.create_tools.assert_called_once_with(["reset_env"])
        assert len(tools) == 1


class TestToolExecution:
    """Tests for tool execution and result formatting."""

    def test_result_formatting(self):
        """Test result formatting."""
        definition = ToolDefinition(
            name="test_tool",
            description="Test",
            parameters={},
            server_url="http://localhost:8070"
        )

        factory = MCPToolFactory("http://localhost:8070")

        with patch.object(factory.client, 'call_tool', return_value={
            "success": True,
            "result": "test_result"
        }):
            tool = factory.create_callable_tool(definition)
            result = tool()
            assert result == {"success": True, "result": "test_result"}

    def test_error_handling(self):
        """Test error handling in tool execution."""
        definition = ToolDefinition(
            name="test_tool",
            description="Test",
            parameters={},
            server_url="http://localhost:8070"
        )

        factory = MCPToolFactory("http://localhost:8070")

        with patch.object(factory.client, 'call_tool', side_effect=Exception("Server error")):
            tool = factory.create_callable_tool(definition)

            with pytest.raises(RuntimeError, match="Tool 'test_tool' failed"):
                tool()
