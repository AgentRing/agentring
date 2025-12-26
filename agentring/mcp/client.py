"""MCP Server Client - Enhanced connection management for MCP servers."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx

from agentring.mcp.types import ServerInfo


class MCPServerClient:
    """
    Enhanced MCP server client with connection management and health checks.

    This class provides a higher-level interface for interacting with MCP servers,
    building on the patterns established in AgentRingClient but focused specifically
    on MCP protocol interactions.
    """

    def __init__(
        self,
        server_url: str,
        name: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        health_check_interval: float = 60.0,
    ):
        """
        Initialize MCP server client.

        Args:
            server_url: Base URL of the MCP server (e.g., "http://localhost:8070")
            name: Optional name for this server instance
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            health_check_interval: How often to perform health checks (seconds)
        """
        self.server_url = server_url.rstrip("/")
        self.name = name or f"mcp-server-{hash(server_url) % 1000}"

        # Connection settings
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.health_check_interval = health_check_interval

        # HTTP client (lazy initialization)
        self._client: Optional[httpx.Client] = None
        self._server_info: Optional[ServerInfo] = None
        self._last_health_check: float = 0

        # Initialize server info
        self._server_info = ServerInfo(
            url=self.server_url,
            name=self.name,
            is_healthy=False
        )

    @property
    def client(self) -> httpx.Client:
        """Get HTTP client (lazy initialization)."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    @property
    def server_info(self) -> ServerInfo:
        """Get current server information."""
        return self._server_info or ServerInfo(url=self.server_url, name=self.name)

    def close(self) -> None:
        """Close HTTP client and clean up resources."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> MCPServerClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def _should_retry_health_check(self) -> bool:
        """Check if we should perform a health check based on interval."""
        return time.time() - self._last_health_check >= self.health_check_interval

    def health_check(self, force: bool = False) -> bool:
        """
        Perform health check on the MCP server.

        Args:
            force: Force health check even if interval hasn't passed

        Returns:
            True if server is healthy, False otherwise
        """
        if not force and not self._should_retry_health_check():
            return self.server_info.is_healthy

        try:
            # Try health endpoint first
            response = self.client.get(f"{self.server_url}/health", timeout=5.0)
            response.raise_for_status()
            is_healthy = response.json().get("status") == "healthy"
        except (httpx.HTTPError, ValueError):
            # Fallback to info endpoint
            try:
                response = self.client.get(f"{self.server_url}/info", timeout=5.0)
                response.raise_for_status()
                is_healthy = response.json().get("success", False)
            except (httpx.HTTPError, ValueError, KeyError):
                is_healthy = False

        # Update server info
        self._last_health_check = time.time()
        if self._server_info:
            self._server_info.is_healthy = is_healthy
            self._server_info.last_health_check = self._last_health_check

        return is_healthy

    def call_tool(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        use_mcp: bool = True
    ) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            params: Parameters to pass to the tool
            use_mcp: Whether to use MCP protocol (True) or REST API (False)

        Returns:
            Tool execution result

        Raises:
            RuntimeError: If the tool call fails
        """
        params = params or {}

        # Try MCP protocol first if requested
        if use_mcp:
            try:
                return self._call_mcp_tool(tool_name, params)
            except (httpx.HTTPError, RuntimeError):
                # Fall back to REST if MCP fails
                if not use_mcp:
                    raise

        # Fall back to REST API
        return self._call_rest_tool(tool_name, params)

    def _call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool using MCP protocol."""
        url = f"{self.server_url}/mcp/v1/tools/{tool_name}/call"
        payload = {"params": params}

        response = self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def _call_rest_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool using REST API."""
        # Map tool names to REST endpoints
        endpoint_map = {
            "get_env_info": "/info",
            "reset_env": "/reset",
            "step_env": "/step",
            "render_env": "/render",
            "close_env": "/close",
        }

        if tool_name not in endpoint_map:
            raise ValueError(f"Unknown tool: {tool_name}")

        url = f"{self.server_url}{endpoint_map[tool_name]}"
        method = "GET" if tool_name == "get_env_info" else "POST"

        if method == "GET":
            response = self.client.get(url)
        else:
            response = self.client.post(url, json=params)

        response.raise_for_status()
        return response.json()

    def get_server_info(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Get server information.

        Args:
            refresh: Force refresh of cached information

        Returns:
            Server information dictionary
        """
        if refresh or not self._server_info:
            try:
                result = self.call_tool("get_env_info", use_mcp=False)
                if result.get("success"):
                    env_info = result.get("env_info", {})
                    self._server_info = ServerInfo(
                        url=self.server_url,
                        name=self.name,
                        version=env_info.get("version"),
                        tools_available=list(env_info.keys()) if isinstance(env_info, dict) else None,
                        is_healthy=True,
                        last_health_check=time.time()
                    )
                    return env_info
            except Exception:
                pass

        return {}

    def is_available(self) -> bool:
        """
        Check if the MCP server is available and responding.

        Returns:
            True if server is available, False otherwise
        """
        return self.health_check()

    def __repr__(self) -> str:
        return f"MCPServerClient(url='{self.server_url}', name='{self.name}', healthy={self.server_info.is_healthy})"
