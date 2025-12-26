"""Multi-server client for managing multiple MCP servers."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from agentring.mcp.client import MCPServerClient
from agentring.mcp.tool_factory import create_tools
from agentring.mcp.types import ToolCallable, ToolDefinition


class MultiServerClient:
    """
    Client for managing multiple MCP servers and their tools.

    This class allows you to register multiple MCP servers and work with
    their combined tool sets as if they were a single server.
    """

    def __init__(self):
        """Initialize multi-server client."""
        self.servers: Dict[str, MCPServerClient] = {}
        self._cached_tools: Optional[Dict[str, List[ToolCallable]]] = None
        self._tool_name_to_server: Optional[Dict[str, str]] = None

    def add_server(
        self,
        name: str,
        url: str,
        client: Optional[MCPServerClient] = None,
        **client_kwargs
    ) -> None:
        """
        Add an MCP server.

        Args:
            name: Unique name for this server
            url: Server URL
            client: Optional pre-configured MCPServerClient instance
            **client_kwargs: Additional arguments for MCPServerClient
        """
        if name in self.servers:
            raise ValueError(f"Server '{name}' already exists")

        if client is None:
            client = MCPServerClient(url, name=name, **client_kwargs)

        self.servers[name] = client

        # Invalidate caches
        self._cached_tools = None
        self._tool_name_to_server = None

    def remove_server(self, name: str) -> None:
        """
        Remove an MCP server.

        Args:
            name: Server name to remove
        """
        if name not in self.servers:
            raise ValueError(f"Server '{name}' not found")

        # Close the client
        self.servers[name].close()
        del self.servers[name]

        # Invalidate caches
        self._cached_tools = None
        self._tool_name_to_server = None

    def get_server(self, name: str) -> MCPServerClient:
        """
        Get a server client by name.

        Args:
            name: Server name

        Returns:
            MCPServerClient instance

        Raises:
            ValueError: If server not found
        """
        if name not in self.servers:
            raise ValueError(f"Server '{name}' not found")
        return self.servers[name]

    def list_servers(self) -> List[str]:
        """
        List all registered server names.

        Returns:
            List of server names
        """
        return list(self.servers.keys())

    def get_tools(self, server_name: str) -> List[ToolCallable]:
        """
        Get tools from a specific server.

        Args:
            server_name: Name of the server

        Returns:
            List of callable tools
        """
        server = self.get_server(server_name)
        return create_tools(server.server_url, client=server)

    def get_all_tools(self) -> List[ToolCallable]:
        """
        Get tools from all registered servers.

        Returns:
            Combined list of callable tools from all servers
        """
        if self._cached_tools is None:
            self._cached_tools = {}
            self._tool_name_to_server = {}

            for server_name, server in self.servers.items():
                tools = create_tools(server.server_url, client=server)
                self._cached_tools[server_name] = tools

                # Map tool names to server names
                for tool in tools:
                    tool_name = getattr(tool, '__name__', str(id(tool)))
                    self._tool_name_to_server[tool_name] = server_name

        # Combine all tools
        all_tools = []
        for tools in self._cached_tools.values():
            all_tools.extend(tools)

        return all_tools

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """
        Get the server name that provides a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Server name, or None if not found
        """
        if self._tool_name_to_server is None:
            # Trigger cache population
            self.get_all_tools()

        return self._tool_name_to_server.get(tool_name)

    def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health checks on all servers.

        Returns:
            Dictionary mapping server names to health status
        """
        results = {}
        for name, server in self.servers.items():
            results[name] = server.health_check()
        return results

    def get_healthy_servers(self) -> List[str]:
        """
        Get names of all healthy servers.

        Returns:
            List of healthy server names
        """
        return [name for name, server in self.servers.items() if server.is_available()]

    def get_unhealthy_servers(self) -> List[str]:
        """
        Get names of all unhealthy servers.

        Returns:
            List of unhealthy server names
        """
        return [name for name, server in self.servers.items() if not server.is_available()]

    def call_tool_on_server(
        self,
        server_name: str,
        tool_name: str,
        params: Optional[Dict] = None
    ):
        """
        Call a tool on a specific server.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool to call
            params: Tool parameters

        Returns:
            Tool execution result
        """
        server = self.get_server(server_name)
        return server.call_tool(tool_name, params)

    def close(self) -> None:
        """Close all server connections."""
        for server in self.servers.values():
            server.close()
        self.servers.clear()

        # Clear caches
        self._cached_tools = None
        self._tool_name_to_server = None

    def __enter__(self) -> MultiServerClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __len__(self) -> int:
        """Return number of registered servers."""
        return len(self.servers)

    def __contains__(self, name: str) -> bool:
        """Check if a server is registered."""
        return name in self.servers

    def __repr__(self) -> str:
        """String representation."""
        healthy = len(self.get_healthy_servers())
        total = len(self.servers)
        return f"MultiServerClient(servers={total}, healthy={healthy})"
