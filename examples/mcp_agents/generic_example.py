#!/usr/bin/env python3
"""
Generic MCP Tool Usage Example

This example demonstrates the basic usage of AgentRing's MCP extensions
to create and use tools from MCP servers in a generic, SDK-agnostic way.
"""

import agentring.mcp as gym_mcp


def main():
    """Demonstrate generic MCP tool usage."""

    # Server URL (change this to your running gym-mcp-server)
    server_url = "http://localhost:8070"

    print("AgentRing MCP Generic Tool Example")
    print("=" * 40)
    print(f"Server URL: {server_url}")
    print()

    try:
        # 1. Create tools from MCP server
        print("1. Discovering and creating tools...")
        tools = gym_mcp.create_tools(server_url)

        print(f"✓ Found {len(tools)} tools:")
        for i, tool in enumerate(tools):
            print(f"  {i+1}. {tool.__name__}: {tool.__doc__ or 'No description'}")
        print()

        # 2. Test individual tool calls
        print("2. Testing individual tool calls...")

        # Find reset tool
        reset_tool = None
        for tool in tools:
            if "reset" in tool.__name__.lower():
                reset_tool = tool
                break

        if reset_tool:
            print("✓ Found reset tool, calling it...")
            result = reset_tool(seed=42)
            print(f"Reset result: {result}")
            print()

        # Find step tool
        step_tool = None
        for tool in tools:
            if "step" in tool.__name__.lower():
                step_tool = tool
                break

        if step_tool:
            print("✓ Found step tool, calling it with a test action...")
            # Use a simple action that works with most environments
            result = step_tool(action="look")  # TextWorld action
            print(f"Step result: {result}")
            print()

        # 3. Demonstrate MCP server client
        print("3. Using MCP server client directly...")
        client = gym_mcp.MCPServerClient(server_url)

        print(f"Server health: {client.health_check()}")
        print(f"Server info: {client.get_server_info()}")
        print()

        # 4. Show format conversion
        print("4. Demonstrating format conversion...")
        if tools:
            from agentring.mcp import formats

            tool_def = gym_mcp.discovery.discover_tools(server_url)[0]

            print("Original tool definition:")
            print(f"  Name: {tool_def.name}")
            print(f"  Description: {tool_def.description}")
            print(f"  Parameters: {tool_def.parameters}")
            print()

            print("Converted to JSON Schema:")
            json_schema = formats.to_json_schema(tool_def)
            print(f"  {json_schema}")
            print()

            print("Converted to OpenAPI:")
            openapi = formats.to_openapi_spec(tool_def)
            print(f"  Summary: {openapi.get('summary', 'N/A')}")
            print(f"  Parameters: {len(openapi.get('parameters', []))}")
            print()

        print("✓ Example completed successfully!")

    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        print("Make sure you have a gym-mcp-server running at the specified URL.")
        print("Example servers to try:")
        print("  TextWorld: python -m gym_mcp_server --env textworld --port 8070")
        print("  ALFWorld: python -m gym_mcp_server --env alfworld --port 8090")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
