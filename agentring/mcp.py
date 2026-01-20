"""MCP Server Factory Functions for AgentRing.

This module provides factory functions to create MCP server instances
for use with various agent frameworks. Each framework integration is optional
and will raise ImportError if the required dependencies are not installed.

Example usage with OpenAI Agents SDK:
    ```python
    from agentring.mcp import get_openai_agents_mcp_server
    from agents import Agent, Runner
    
    mcp_server = get_openai_agents_mcp_server(
        server_url="http://localhost:8000",
        api_key="your-api-key",
        session_key="my-experiment",
    )
    
    async with mcp_server:
        agent = Agent(
            name="My Agent",
            instructions="...",
            mcp_servers=[mcp_server]
        )
        result = await Runner.run(agent, "task")
    ```

Example usage with Pydantic AI:
    ```python
    from agentring.mcp import get_pydanticai_mcp_server
    from pydantic_ai import Agent
    
    mcp_server = get_pydanticai_mcp_server(
        server_url="http://localhost:8000",
        api_key="your-api-key",
        session_key="my-experiment",
    )
    
    agent = Agent("openai:gpt-4o", mcp_servers=[mcp_server])
    
    async with mcp_server:
        result = await agent.run("task")
    ```
"""

from typing import Any, Optional


def get_openai_agents_mcp_server(
    server_url: str,
    api_key: str,
    session_key: str,
    name: str = "AgentRing Environment",
    timeout: int = 30,
) -> Any:
    """Create an authenticated MCP server instance for use with OpenAI Agents SDK.
    
    This function returns a pre-configured MCPServerStreamableHttp instance that
    includes the necessary authentication headers (X-API-Key, X-Session-Key) to
    connect to the AgentRing server's MCP endpoint.
    
    Args:
        server_url: Base URL of the AgentRing server (e.g., "http://localhost:8000")
        api_key: API key for authentication
        session_key: Session key for identifying this experiment/run
        name: Name for the MCP server (default: "AgentRing Environment")
        timeout: Request timeout in seconds (default: 30)
        
    Returns:
        MCPServerStreamableHttp instance configured with authentication
        
    Raises:
        ImportError: If openai-agents package is not installed
        
    Example:
        ```python
        from agentring.mcp import get_openai_agents_mcp_server
        from agents import Agent, Runner
        
        mcp_server = get_openai_agents_mcp_server(
            server_url="http://localhost:8000",
            api_key="my-key",
            session_key="experiment-1",
        )
        
        async with mcp_server:
            agent = Agent(
                name="My Agent",
                instructions="Navigate the environment",
                mcp_servers=[mcp_server]
            )
            result = await Runner.run(agent, "task")
        ```
    """
    try:
        from agents.mcp import MCPServerStreamableHttp
    except ImportError:
        raise ImportError(
            "OpenAI Agents SDK is required to use get_openai_agents_mcp_server(). "
            "Install it with: pip install agentring[openai-agents] or pip install openai-agents"
        )
    
    mcp_url = f"{server_url.rstrip('/')}/mcp"
    
    return MCPServerStreamableHttp(
        name=name,
        params={
            "url": mcp_url,
            "timeout": timeout,
            "headers": {
                "X-API-Key": api_key,
                "X-Session-Key": session_key,
            }
        },
    )


def get_pydanticai_mcp_server(
    server_url: str,
    api_key: str,
    session_key: str,
    timeout: Optional[float] = None,
) -> Any:
    """Create an authenticated MCP server instance for use with Pydantic AI.
    
    This function returns a pre-configured MCPServerStreamableHTTP instance that
    includes the necessary authentication headers (X-API-Key, X-Session-Key) to
    connect to the AgentRing server's MCP endpoint.
    
    Args:
        server_url: Base URL of the AgentRing server (e.g., "http://localhost:8000")
        api_key: API key for authentication
        session_key: Session key for identifying this experiment/run
        timeout: Optional request timeout in seconds
        
    Returns:
        MCPServerStreamableHTTP instance configured with authentication
        
    Raises:
        ImportError: If pydantic-ai package is not installed
        
    Example:
        ```python
        from agentring.mcp import get_pydanticai_mcp_server
        from pydantic_ai import Agent
        
        mcp_server = get_pydanticai_mcp_server(
            server_url="http://localhost:8000",
            api_key="my-key",
            session_key="experiment-1",
        )
        
        agent = Agent("openai:gpt-4o", mcp_servers=[mcp_server])
        
        async with mcp_server:
            result = await agent.run("Navigate the environment")
        ```
    """
    try:
        from pydantic_ai.mcp import MCPServerStreamableHTTP
    except ImportError:
        raise ImportError(
            "Pydantic AI is required to use get_pydanticai_mcp_server(). "
            "Install it with: pip install agentring[pydantic-ai] or pip install pydantic-ai"
        )
    
    mcp_url = f"{server_url.rstrip('/')}/mcp"
    
    kwargs: dict[str, Any] = {
        "url": mcp_url,
        "headers": {
            "X-API-Key": api_key,
            "X-Session-Key": session_key,
        },
    }
    
    if timeout is not None:
        kwargs["timeout"] = timeout
    
    return MCPServerStreamableHTTP(**kwargs)
