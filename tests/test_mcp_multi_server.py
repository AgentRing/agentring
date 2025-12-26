"""Tests for MCP multi-server client."""

import pytest
from unittest.mock import Mock, patch

from agentring.mcp.multi_server import MultiServerClient


class TestMultiServerClient:
    """Tests for MultiServerClient."""

    def test_initialization(self):
        """Test multi-server client initialization."""
        client = MultiServerClient()
        assert len(client) == 0
        assert "server1" not in client

    def test_add_server(self):
        """Test adding servers."""
        client = MultiServerClient()

        client.add_server("server1", "http://localhost:8070")

        assert len(client) == 1
        assert "server1" in client
        assert client.list_servers() == ["server1"]

    def test_add_duplicate_server(self):
        """Test adding duplicate server names."""
        client = MultiServerClient()

        client.add_server("server1", "http://localhost:8070")

        with pytest.raises(ValueError, match="already exists"):
            client.add_server("server1", "http://localhost:8080")

    def test_get_server(self):
        """Test getting server clients."""
        client = MultiServerClient()

        client.add_server("server1", "http://localhost:8070")

        server = client.get_server("server1")
        assert server.server_url == "http://localhost:8070"

    def test_get_nonexistent_server(self):
        """Test getting nonexistent server."""
        client = MultiServerClient()

        with pytest.raises(ValueError, match="not found"):
            client.get_server("nonexistent")

    def test_remove_server(self):
        """Test removing servers."""
        client = MultiServerClient()

        client.add_server("server1", "http://localhost:8070")
        assert len(client) == 1

        client.remove_server("server1")
        assert len(client) == 0
        assert "server1" not in client

    def test_remove_nonexistent_server(self):
        """Test removing nonexistent server."""
        client = MultiServerClient()

        with pytest.raises(ValueError, match="not found"):
            client.remove_server("nonexistent")

    @patch('agentring.mcp.tool_factory.create_tools')
    def test_get_tools(self, mock_create_tools):
        """Test getting tools from specific server."""
        mock_create_tools.return_value = [Mock(__name__="tool1")]

        client = MultiServerClient()
        client.add_server("server1", "http://localhost:8070")

        tools = client.get_tools("server1")

        assert len(tools) == 1
        mock_create_tools.assert_called_once_with("http://localhost:8070", client=client.servers["server1"])

    @patch('agentring.mcp.tool_factory.create_tools')
    def test_get_all_tools(self, mock_create_tools):
        """Test getting tools from all servers."""
        mock_create_tools.side_effect = [
            [Mock(__name__="tool1")],  # server1
            [Mock(__name__="tool2")]   # server2
        ]

        client = MultiServerClient()
        client.add_server("server1", "http://localhost:8070")
        client.add_server("server2", "http://localhost:8080")

        all_tools = client.get_all_tools()

        assert len(all_tools) == 2
        assert mock_create_tools.call_count == 2

        # Test caching
        all_tools2 = client.get_all_tools()
        assert len(all_tools2) == 2
        assert mock_create_tools.call_count == 2  # Should not call again

    def test_get_server_for_tool(self):
        """Test finding server for tool."""
        client = MultiServerClient()

        with patch.object(client, 'get_all_tools') as mock_get_all:
            mock_get_all.return_value = [
                Mock(__name__="tool1"),
                Mock(__name__="tool2")
            ]

            # Set up server mapping
            client._tool_name_to_server = {
                "tool1": "server1",
                "tool2": "server2"
            }

            assert client.get_server_for_tool("tool1") == "server1"
            assert client.get_server_for_tool("tool2") == "server2"
            assert client.get_server_for_tool("nonexistent") is None

    def test_health_check_all(self):
        """Test health checking all servers."""
        client = MultiServerClient()

        client.add_server("server1", "http://localhost:8070")
        client.add_server("server2", "http://localhost:8080")

        with patch.object(client.servers["server1"], 'health_check', return_value=True), \
             patch.object(client.servers["server2"], 'health_check', return_value=False):

            health = client.health_check_all()

            assert health["server1"] is True
            assert health["server2"] is False

    def test_get_healthy_servers(self):
        """Test getting healthy servers."""
        client = MultiServerClient()

        client.add_server("server1", "http://localhost:8070")
        client.add_server("server2", "http://localhost:8080")

        with patch.object(client.servers["server1"], 'is_available', return_value=True), \
             patch.object(client.servers["server2"], 'is_available', return_value=False):

            healthy = client.get_healthy_servers()
            unhealthy = client.get_unhealthy_servers()

            assert healthy == ["server1"]
            assert unhealthy == ["server2"]

    def test_call_tool_on_server(self):
        """Test calling tools on specific servers."""
        client = MultiServerClient()

        client.add_server("server1", "http://localhost:8070")

        with patch.object(client.servers["server1"], 'call_tool', return_value={"result": "success"}):
            result = client.call_tool_on_server("server1", "test_tool", {"param": "value"})

            assert result == {"result": "success"}

    def test_context_manager(self):
        """Test context manager usage."""
        with MultiServerClient() as client:
            client.add_server("server1", "http://localhost:8070")
            assert len(client) == 1

        # Should be cleaned up
        assert len(client) == 0

    def test_repr(self):
        """Test string representation."""
        client = MultiServerClient()

        # Empty client
        repr_str = repr(client)
        assert "MultiServerClient" in repr_str
        assert "servers=0" in repr_str

        # Client with servers
        client.add_server("server1", "http://localhost:8070")
        repr_str = repr(client)
        assert "servers=1" in repr_str
        assert "healthy=" in repr_str
