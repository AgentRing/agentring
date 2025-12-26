"""Tests for MCP server client."""

import pytest
from unittest.mock import Mock, patch
import httpx

from agentring.mcp.client import MCPServerClient


class TestMCPServerClient:
    """Tests for MCPServerClient."""

    def test_initialization(self):
        """Test basic client initialization."""
        client = MCPServerClient("http://localhost:8070", name="test")

        assert client.server_url == "http://localhost:8070"
        assert client.name == "test"
        assert client.timeout == 30.0
        assert client.max_retries == 3

        client.close()

    def test_context_manager(self):
        """Test context manager usage."""
        with MCPServerClient("http://localhost:8070") as client:
            assert client.server_url == "http://localhost:8070"

        # Client should be closed
        assert client._client is None

    @patch('httpx.Client.get')
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        # Mock successful health response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response

        client = MCPServerClient("http://localhost:8070")

        result = client.health_check()

        assert result is True
        assert client.server_info.is_healthy is True

        mock_get.assert_called_once_with("http://localhost:8070/health", timeout=5.0)

        client.close()

    @patch('httpx.Client.get')
    def test_health_check_failure(self, mock_get):
        """Test failed health check."""
        # Mock failed health response
        mock_get.side_effect = httpx.HTTPError("Connection failed")

        client = MCPServerClient("http://localhost:8070")

        result = client.health_check()

        assert result is False
        assert client.server_info.is_healthy is False

        client.close()

    @patch('httpx.Client.get')
    def test_health_check_fallback_to_info(self, mock_get):
        """Test health check fallback to info endpoint."""
        # Mock health endpoint failure, info endpoint success
        def mock_get_side_effect(url, **kwargs):
            if "health" in url:
                raise httpx.HTTPError("Health failed")
            else:  # info endpoint
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {"success": True}
                return mock_response

        mock_get.side_effect = mock_get_side_effect

        client = MCPServerClient("http://localhost:8070")

        result = client.health_check()

        assert result is True
        assert client.server_info.is_healthy is True

        client.close()

    @patch('httpx.Client.post')
    def test_call_tool_rest_success(self, mock_post):
        """Test successful REST tool call."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"success": True, "result": "test"}
        mock_post.return_value = mock_response

        client = MCPServerClient("http://localhost:8070")

        result = client.call_tool("reset_env", {"seed": 42}, use_mcp=False)

        assert result == {"success": True, "result": "test"}
        mock_post.assert_called_once_with(
            "http://localhost:8070/reset",
            json={"seed": 42}
        )

        client.close()

    @patch('httpx.Client.post')
    def test_call_tool_mcp_success(self, mock_post):
        """Test successful MCP tool call."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"success": True, "result": "test"}
        mock_post.return_value = mock_response

        client = MCPServerClient("http://localhost:8070")

        result = client.call_tool("custom_tool", {"param": "value"}, use_mcp=True)

        assert result == {"success": True, "result": "test"}
        mock_post.assert_called_once_with(
            "http://localhost:8070/mcp/v1/tools/custom_tool/call",
            json={"params": {"param": "value"}}
        )

        client.close()

    @patch('httpx.Client.post')
    def test_call_tool_mcp_fallback(self, mock_post):
        """Test MCP tool call with fallback to REST."""
        # MCP call fails, REST succeeds
        def mock_post_side_effect(url, **kwargs):
            if "mcp" in url:
                raise httpx.HTTPError("MCP failed")
            else:  # REST endpoint
                mock_response = Mock()
                mock_response.raise_for_status.return_value = None
                mock_response.json.return_value = {"success": True}
                return mock_response

        mock_post.side_effect = mock_post_side_effect

        client = MCPServerClient("http://localhost:8070")

        result = client.call_tool("reset_env", {"seed": 42})

        assert result == {"success": True}
        # Should have tried MCP first, then fallen back to REST
        assert mock_post.call_count == 2

        client.close()

    def test_is_available(self):
        """Test is_available method."""
        client = MCPServerClient("http://localhost:8070")

        with patch.object(client, 'health_check', return_value=True):
            assert client.is_available() is True

        with patch.object(client, 'health_check', return_value=False):
            assert client.is_available() is False

        client.close()

    def test_get_server_info(self):
        """Test get_server_info method."""
        client = MCPServerClient("http://localhost:8070")

        with patch.object(client, 'call_tool', return_value={"env_info": {"version": "1.0"}}):
            info = client.get_server_info()
            assert info == {"version": "1.0"}

        client.close()

    def test_repr(self):
        """Test string representation."""
        client = MCPServerClient("http://localhost:8070", name="test")
        repr_str = repr(client)
        assert "MCPServerClient" in repr_str
        assert "http://localhost:8070" in repr_str
        assert "test" in repr_str

        client.close()
