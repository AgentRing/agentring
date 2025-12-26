"""Tests for MCP tool discovery."""

import pytest
from unittest.mock import Mock, patch

from agentring.mcp.discovery import discover_tools, _discover_via_mcp, _discover_via_rest
from agentring.mcp.types import ToolDefinition


class TestToolDiscovery:
    """Tests for tool discovery functionality."""

    @patch('agentring.mcp.discovery._discover_via_mcp')
    @patch('agentring.mcp.discovery._discover_via_rest')
    def test_discover_tools_mcp_success(self, mock_rest, mock_mcp):
        """Test tool discovery with successful MCP."""
        mock_mcp.return_value = [
            ToolDefinition("reset_env", "Reset environment", {}, "http://localhost:8070")
        ]

        tools = discover_tools("http://localhost:8070")

        assert len(tools) == 1
        assert tools[0].name == "reset_env"
        mock_mcp.assert_called_once()
        mock_rest.assert_not_called()

    @patch('agentring.mcp.discovery._discover_via_mcp')
    @patch('agentring.mcp.discovery._discover_via_rest')
    def test_discover_tools_mcp_fallback(self, mock_rest, mock_mcp):
        """Test tool discovery with MCP failure and REST fallback."""
        mock_mcp.side_effect = Exception("MCP failed")
        mock_rest.return_value = [
            ToolDefinition("reset_env", "Reset environment", {}, "http://localhost:8070")
        ]

        tools = discover_tools("http://localhost:8070")

        assert len(tools) == 1
        assert tools[0].name == "reset_env"
        mock_mcp.assert_called_once()
        mock_rest.assert_called_once()

    @patch('httpx.Client.get')
    def test_discover_via_mcp_success(self, mock_get):
        """Test MCP protocol discovery success."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "tools": [
                {
                    "name": "reset_env",
                    "description": "Reset environment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "seed": {"type": "integer"}
                        }
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        from agentring.mcp.client import MCPServerClient
        client = MCPServerClient("http://localhost:8070")

        tools = _discover_via_mcp(client)

        assert len(tools) == 1
        assert tools[0].name == "reset_env"
        assert tools[0].description == "Reset environment"
        assert tools[0].server_url == "http://localhost:8070"
        assert "seed" in tools[0].parameters["properties"]

        client.close()

    @patch('httpx.Client.get')
    def test_discover_via_mcp_fallback(self, mock_get):
        """Test MCP protocol discovery fallback."""
        mock_get.side_effect = Exception("MCP endpoint not available")

        from agentring.mcp.client import MCPServerClient
        client = MCPServerClient("http://localhost:8070")

        tools = _discover_via_mcp(client)

        # Should return standard tools
        assert len(tools) == 5  # reset_env, step_env, get_env_info, render_env, close_env
        tool_names = [t.name for t in tools]
        assert "reset_env" in tool_names
        assert "step_env" in tool_names
        assert "get_env_info" in tool_names

        client.close()

    def test_discover_via_rest(self):
        """Test REST API discovery."""
        from agentring.mcp.client import MCPServerClient

        # Mock client with env info
        client = MCPServerClient("http://localhost:8070")

        with patch.object(client, 'get_server_info', return_value={
            "observation_space": {"type": "Box"},
            "action_space": {"type": "Discrete", "n": 4},
            "render_modes": ["rgb_array"]
        }):
            tools = _discover_via_rest(client)

            tool_names = [t.name for t in tools]
            assert "reset_env" in tool_names
            assert "step_env" in tool_names
            assert "get_env_info" in tool_names
            assert "render_env" in tool_names

        client.close()

    def test_infer_action_schema_discrete(self):
        """Test action schema inference for discrete spaces."""
        from agentring.mcp.discovery import _infer_action_schema

        schema = _infer_action_schema({"type": "Discrete", "n": 4, "start": 0})

        assert schema["type"] == "integer"
        assert schema["minimum"] == 0
        assert schema["maximum"] == 3

    def test_infer_action_schema_box(self):
        """Test action schema inference for Box spaces."""
        from agentring.mcp.discovery import _infer_action_schema

        schema = _infer_action_schema({"type": "Box", "shape": [2]})

        assert schema["type"] == "array"
        assert schema["items"] == {"type": "number"}

    def test_infer_action_schema_default(self):
        """Test action schema inference for unknown spaces."""
        from agentring.mcp.discovery import _infer_action_schema

        schema = _infer_action_schema({"type": "Unknown"})

        assert schema["type"] == "string"
        assert "description" in schema
