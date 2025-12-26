#!/usr/bin/env python3
"""
Multi-Server MCP Example

This example demonstrates using AgentRing's MultiServerClient to work with
multiple MCP servers simultaneously, combining tools from different environments.
"""

import agentring.mcp as gym_mcp


def main():
    """Demonstrate multi-server MCP usage."""

    print("AgentRing Multi-Server MCP Example")
    print("=" * 38)
    print()

    try:
        # 1. Create multi-server client
        print("1. Creating multi-server client...")
        multi_client = gym_mcp.MultiServerClient()
        print("✓ Multi-server client created")
        print()

        # 2. Add multiple servers
        print("2. Adding MCP servers...")

        # Add TextWorld server (example)
        try:
            multi_client.add_server("textworld", "http://localhost:8070")
            print("✓ Added TextWorld server (port 8070)")
        except Exception as e:
            print(f"⚠ Could not add TextWorld server: {e}")

        # Add ALFWorld server (example)
        try:
            multi_client.add_server("alfworld", "http://localhost:8090")
            print("✓ Added ALFWorld server (port 8090)")
        except Exception as e:
            print(f"⚠ Could not add ALFWorld server: {e}")

        # Add WebShop server (example)
        try:
            multi_client.add_server("webshop", "http://localhost:8002")
            print("✓ Added WebShop server (port 8002)")
        except Exception as e:
            print(f"⚠ Could not add WebShop server: {e}")

        print(f"✓ Total servers registered: {len(multi_client)}")
        print()

        # 3. Health check all servers
        print("3. Checking server health...")
        health_status = multi_client.health_check_all()

        for server_name, is_healthy in health_status.items():
            status = "✓ Healthy" if is_healthy else "✗ Unhealthy"
            print(f"  {server_name}: {status}")

        healthy_servers = multi_client.get_healthy_servers()
        unhealthy_servers = multi_client.get_unhealthy_servers()

        print(f"✓ Healthy servers: {healthy_servers}")
        print(f"⚠ Unhealthy servers: {unhealthy_servers}")
        print()

        if not healthy_servers:
            print("No healthy servers found. Make sure you have gym-mcp-servers running:")
            print("  TextWorld: python -m gym_mcp_server --env textworld --port 8070")
            print("  ALFWorld: python -m gym_mcp_server --env alfworld --port 8090")
            print("  WebShop: python -m gym_mcp_server --env webshop --port 8002")
            return 1

        # 4. Get tools from individual servers
        print("4. Getting tools from individual servers...")
        for server_name in healthy_servers:
            try:
                tools = multi_client.get_tools(server_name)
                print(f"✓ {server_name}: {len(tools)} tools available")
                if tools:
                    tool_names = [getattr(t, '__name__', 'unknown') for t in tools[:3]]  # Show first 3
                    print(f"    Tools: {', '.join(tool_names)}{'...' if len(tools) > 3 else ''}")
            except Exception as e:
                print(f"✗ {server_name}: Error getting tools - {e}")

        print()

        # 5. Get all tools combined
        print("5. Getting all tools combined...")
        try:
            all_tools = multi_client.get_all_tools()
            print(f"✓ Total tools from all servers: {len(all_tools)}")

            # Group by server
            from agentring.mcp.utils import group_tools_by_server
            grouped = group_tools_by_server(all_tools)

            for server_name, server_tools in grouped.items():
                print(f"  {server_name}: {len(server_tools)} tools")

        except Exception as e:
            print(f"✗ Error getting combined tools: {e}")
            all_tools = []
        print()

        # 6. Demonstrate server-specific tool calls
        if healthy_servers and all_tools:
            print("6. Demonstrating server-specific tool calls...")

            # Find a server with tools
            test_server = healthy_servers[0]
            try:
                # Call a tool on a specific server
                result = multi_client.call_tool_on_server(
                    test_server, "get_env_info", {}
                )
                print(f"✓ Called get_env_info on {test_server}")
                if isinstance(result, dict) and result.get("success"):
                    print("  ✓ Server responded successfully")
                else:
                    print(f"  ⚠ Server response: {result}")

            except Exception as e:
                print(f"✗ Error calling tool on {test_server}: {e}")

        print()
        print("✓ Multi-server example completed!")

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
