"""AgentRing Client Library

A client library for connecting to AgentRing Server to run Gymnasium environments
with session management and episode tracking.

Example usage:
    ```python
    from agentring import AgentRingClient
    
    # Initialize client
    client = AgentRingClient(
        server_url="http://localhost:8000",
        api_key="your-api-key",
        session_key="my-experiment-1"
    )
    
    # Create a session with 10 episodes
    client.create_session(num_episodes=10)
    
    # Run episodes
    for episode in range(10):
        obs, info = client.reset()
        done = False
        
        while not done:
            action = client.sample_action()  # or your agent's action
            obs, reward, terminated, truncated, info = client.step(action)
            done = terminated or truncated
    
    # Get final metrics
    metrics = client.close()
    print(f"Average reward: {metrics['avg_reward']}")
    ```
"""

import logging
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class AgentRingError(Exception):
    """Base exception for AgentRing client errors."""
    pass


class SessionNotFoundError(AgentRingError):
    """Raised when session is not found."""
    pass


class SessionCompletedError(AgentRingError):
    """Raised when session has completed all episodes."""
    pass


class AgentRingClient:
    """Client for interacting with AgentRing Server.
    
    This client handles authentication, session management, and provides a clean
    interface for running Gymnasium environments through the AgentRing server.
    
    Attributes:
        server_url: Base URL of the AgentRing server
        api_key: API key for authentication
        session_key: Session key for identifying this experiment/run
        session_id: Current session ID (set after create_session)
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        server_url: str,
        api_key: str,
        session_key: str,
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        """Initialize AgentRing client.
        
        Args:
            server_url: Base URL of the AgentRing server (e.g., "http://localhost:8000")
            api_key: API key for authentication
            session_key: Session key for identifying this experiment/run
            timeout: Request timeout in seconds (default: 30)
            verify_ssl: Whether to verify SSL certificates (default: True)
        """
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.session_key = session_key
        self.session_id: Optional[str] = None
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        # Session state
        self._num_episodes: Optional[int] = None
        self._current_episode: int = 0
        self._session_completed: bool = False
        
        logger.info(f"Initialized AgentRing client for server: {self.server_url}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {
            "X-API-Key": self.api_key,
            "X-Session-Key": self.session_key,
            "Content-Type": "application/json",
        }
        if self.session_id:
            headers["X-Session-ID"] = self.session_id
        return headers
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and raise appropriate errors.
        
        Args:
            response: Response object from requests
            
        Returns:
            Parsed JSON response
            
        Raises:
            SessionNotFoundError: If session not found
            SessionCompletedError: If session completed
            AgentRingError: For other errors
        """
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json().get("detail", str(e))
            except:
                error_detail = str(e)
            
            if response.status_code == 404:
                raise SessionNotFoundError(f"Session not found: {error_detail}")
            elif response.status_code == 400:
                if "completed" in error_detail.lower() or "limit reached" in error_detail.lower():
                    self._session_completed = True
                    raise SessionCompletedError(error_detail)
                raise AgentRingError(f"Bad request: {error_detail}")
            else:
                raise AgentRingError(f"Server error: {error_detail}")
        except requests.exceptions.RequestException as e:
            raise AgentRingError(f"Request failed: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health status.
        
        Returns:
            Health status information
            
        Raises:
            AgentRingError: If health check fails
        """
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            return self._handle_response(response)
        except Exception as e:
            raise AgentRingError(f"Health check failed: {str(e)}")
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information and available endpoints.
        
        Returns:
            API information
            
        Raises:
            AgentRingError: If request fails
        """
        response = requests.get(
            f"{self.server_url}/api",
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        return self._handle_response(response)
    
    def get_openai_agents_mcp_server(self, name: str = "AgentRing Environment") -> Any:
        """Get an authenticated MCP server instance for use with OpenAI Agents SDK.
        
        This method returns a pre-configured MCPServerStreamableHttp instance that
        includes the necessary authentication headers (X-API-Key, X-Session-Key) to
        connect to the AgentRing server's MCP endpoint.
        
        The returned MCP server can be used directly with the OpenAI Agents SDK:
        
        Example:
            ```python
            client = AgentRingClient(...)
            client.create_session(num_episodes=5)
            
            mcp_server = client.get_openai_agents_mcp_server()
            
            async with mcp_server:
                agent = Agent(
                    name="My Agent",
                    instructions="...",
                    mcp_servers=[mcp_server]
                )
                result = await Runner.run(agent, "task")
            ```
        
        Args:
            name: Name for the MCP server (default: "AgentRing Environment")
            
        Returns:
            MCPServerStreamableHttp instance configured with authentication
            
        Raises:
            ImportError: If openai-agents package is not installed
        """
        try:
            from agents.mcp import MCPServerStreamableHttp
        except ImportError:
            raise ImportError(
                "OpenAI Agents SDK is required to use get_openai_agents_mcp_server(). "
                "Install it with: pip install agentring[openai-agents] or pip install openai-agents"
            )
        
        mcp_url = f"{self.server_url}/mcp"
        
        # Create MCP server with authentication headers
        return MCPServerStreamableHttp(
            name=name,
            params={
                "url": mcp_url,
                "timeout": self.timeout,
                "headers": {
                    "X-API-Key": self.api_key,
                    "X-Session-Key": self.session_key,
                }
            },
        )
    
    def get_pydanticai_mcp_server(self) -> Any:
        """Get an authenticated MCP server instance for use with Pydantic AI.
        
        This method returns a pre-configured MCPServerStreamableHTTP instance that
        includes the necessary authentication headers (X-API-Key, X-Session-Key) to
        connect to the AgentRing server's MCP endpoint.
        
        The returned MCP server can be used directly with Pydantic AI:
        
        Example:
            ```python
            from pydantic_ai import Agent
            
            client = AgentRingClient(...)
            client.create_session(num_episodes=5)
            
            mcp_server = client.get_pydanticai_mcp_server()
            
            agent = Agent(
                "openai:gpt-4o",
                mcp_servers=[mcp_server]
            )
            
            async with mcp_server:
                result = await agent.run("task")
            ```
        
        Returns:
            MCPServerStreamableHTTP instance configured with authentication
            
        Raises:
            ImportError: If pydantic-ai package is not installed
        """
        try:
            from pydantic_ai.mcp import MCPServerStreamableHTTP
        except ImportError:
            raise ImportError(
                "Pydantic AI is required to use get_pydanticai_mcp_server(). "
                "Install it with: pip install agentring[pydantic-ai] or pip install pydantic-ai"
            )
        
        mcp_url = f"{self.server_url}/mcp"
        
        # Create MCP server with authentication headers
        return MCPServerStreamableHTTP(
            url=mcp_url,
            headers={
                "X-API-Key": self.api_key,
                "X-Session-Key": self.session_key,
            },
        )
    
    def create_session(self, num_episodes: int) -> str:
        """Create a new session with fixed number of episodes.
        
        This will close any existing session for this api_key + session_key combination.
        
        Args:
            num_episodes: Number of episodes for this session
            
        Returns:
            Session ID
            
        Raises:
            AgentRingError: If session creation fails
        """
        if num_episodes <= 0:
            raise ValueError("num_episodes must be greater than 0")
        
        response = requests.post(
            f"{self.server_url}/sessions/create",
            headers=self._get_headers(),
            json={"num_episodes": num_episodes},
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        
        result = self._handle_response(response)
        self.session_id = result["session_id"]
        self._num_episodes = num_episodes
        self._current_episode = 0
        self._session_completed = False
        
        logger.info(f"Created session {self.session_id} with {num_episodes} episodes")
        return self.session_id
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and start a new episode.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            Tuple of (observation, info)
            
        Raises:
            SessionNotFoundError: If no session exists
            SessionCompletedError: If all episodes completed
            AgentRingError: If reset fails
        """
        if not self.session_id:
            raise SessionNotFoundError("No active session. Call create_session() first.")
        
        if self._session_completed:
            raise SessionCompletedError("Session has completed all episodes")
        
        payload = {}
        if seed is not None:
            payload["seed"] = seed
        
        response = requests.post(
            f"{self.server_url}/reset",
            headers=self._get_headers(),
            json=payload,
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        
        result = self._handle_response(response)
        
        # Update episode tracking
        self._current_episode = result.get("current_episode", self._current_episode)
        
        logger.debug(
            f"Reset environment (episode {self._current_episode}/{self._num_episodes})"
        )
        
        return result["observation"], result.get("info", {})
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            
        Raises:
            SessionNotFoundError: If no session exists
            SessionCompletedError: If all episodes completed
            AgentRingError: If step fails
        """
        if not self.session_id:
            raise SessionNotFoundError("No active session. Call create_session() first.")
        
        if self._session_completed:
            raise SessionCompletedError("Session has completed all episodes")
        
        response = requests.post(
            f"{self.server_url}/step",
            headers=self._get_headers(),
            json={"action": action},
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        
        result = self._handle_response(response)
        
        # Check if session auto-completed
        if result.get("session_completed", False):
            self._session_completed = True
            logger.info("Session completed all episodes")
        
        return (
            result["observation"],
            result["reward"],
            result["terminated"],
            result["truncated"],
            result.get("info", {}),
        )
    
    def render(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Render the current state of the environment.
        
        Args:
            mode: Render mode (if None, uses environment default)
            
        Returns:
            Render result
            
        Raises:
            SessionNotFoundError: If no session exists
            AgentRingError: If render fails
        """
        if not self.session_id:
            raise SessionNotFoundError("No active session. Call create_session() first.")
        
        payload = {}
        if mode is not None:
            payload["mode"] = mode
        
        response = requests.post(
            f"{self.server_url}/render",
            headers=self._get_headers(),
            json=payload,
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        
        return self._handle_response(response)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the environment.
        
        Returns:
            Environment information including action/observation spaces
            
        Raises:
            SessionNotFoundError: If no session exists
            AgentRingError: If request fails
        """
        if not self.session_id:
            raise SessionNotFoundError("No active session. Call create_session() first.")
        
        response = requests.get(
            f"{self.server_url}/info",
            headers=self._get_headers(),
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        
        result = self._handle_response(response)
        return result.get("env_info", {})
    
    def sample_action(self) -> Any:
        """Sample a random action from the environment's action space.
        
        Returns:
            Random action
            
        Raises:
            SessionNotFoundError: If no session exists
            AgentRingError: If request fails
        """
        if not self.session_id:
            raise SessionNotFoundError("No active session. Call create_session() first.")
        
        response = requests.get(
            f"{self.server_url}/sample",
            headers=self._get_headers(),
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        
        result = self._handle_response(response)
        return result["action"]
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics and metrics.
        
        Returns:
            Session statistics including metrics for completed episodes
            
        Raises:
            SessionNotFoundError: If no session exists
            AgentRingError: If request fails
        """
        if not self.session_id:
            raise SessionNotFoundError("No active session. Call create_session() first.")
        
        response = requests.get(
            f"{self.server_url}/sessions/current/stats",
            headers=self._get_headers(),
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        
        result = self._handle_response(response)
        return result.get("metrics", {})
    
    def end_session(self) -> Dict[str, Any]:
        """Explicitly end the current session and get final metrics.
        
        Returns:
            Final session metrics
            
        Raises:
            SessionNotFoundError: If no session exists
            AgentRingError: If request fails
        """
        if not self.session_id:
            raise SessionNotFoundError("No active session. Call create_session() first.")
        
        response = requests.post(
            f"{self.server_url}/sessions/current/end",
            headers=self._get_headers(),
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        
        result = self._handle_response(response)
        metrics = result.get("metrics", {})
        
        # Clear session state
        self.session_id = None
        self._num_episodes = None
        self._current_episode = 0
        self._session_completed = False
        
        logger.info("Session ended")
        return metrics
    
    def close(self) -> Dict[str, Any]:
        """Close the environment session and return final metrics.
        
        This is an alias for end_session() for compatibility with Gymnasium API.
        
        Returns:
            Final session metrics
            
        Raises:
            SessionNotFoundError: If no session exists
            AgentRingError: If request fails
        """
        if not self.session_id:
            raise SessionNotFoundError("No active session. Call create_session() first.")
        
        response = requests.post(
            f"{self.server_url}/close",
            headers=self._get_headers(),
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        
        result = self._handle_response(response)
        metrics = result.get("metrics", {})
        
        # Clear session state
        self.session_id = None
        self._num_episodes = None
        self._current_episode = 0
        self._session_completed = False
        
        logger.info("Session closed")
        return metrics
    
    @property
    def is_session_active(self) -> bool:
        """Check if there is an active session."""
        return self.session_id is not None and not self._session_completed
    
    @property
    def current_episode(self) -> int:
        """Get current episode number (1-based)."""
        return self._current_episode
    
    @property
    def num_episodes(self) -> Optional[int]:
        """Get total number of episodes for this session."""
        return self._num_episodes
    
    def __enter__(self) -> "AgentRingClient":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit - automatically close session."""
        if self.is_session_active:
            try:
                self.close()
            except Exception as e:
                logger.warning(f"Error closing session on exit: {e}")
        return False
    
    def __repr__(self) -> str:
        """String representation."""
        status = "active" if self.is_session_active else "inactive"
        return (
            f"AgentRingClient(server_url='{self.server_url}', "
            f"session_id='{self.session_id}', status='{status}')"
        )


def connect(
    server_url: str,
    api_key: str,
    session_key: str,
    num_episodes: int,
    **kwargs: Any
) -> AgentRingClient:
    """Connect to AgentRing server and create a session in one step.
    
    Args:
        server_url: Base URL of the AgentRing server
        api_key: API key for authentication
        session_key: Session key for identifying this experiment/run
        num_episodes: Number of episodes for this session
        **kwargs: Additional arguments passed to AgentRingClient
        
    Returns:
        Initialized AgentRingClient with active session
        
    Example:
        ```python
        client = connect(
            server_url="http://localhost:8000",
            api_key="my-key",
            session_key="experiment-1",
            num_episodes=10
        )
        ```
    """
    client = AgentRingClient(server_url, api_key, session_key, **kwargs)
    client.create_session(num_episodes)
    return client
