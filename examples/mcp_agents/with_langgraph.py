#!/usr/bin/env python3
"""
MCP Tools with LangGraph Example

This example shows how to use AgentRing's generic MCP tools with LangGraph.
Note: This requires installing LangGraph and LangChain separately.
"""

# import agentring.mcp as gym_mcp

# Uncomment the lines below if you have LangGraph installed
# from langchain_core.tools import tool
# from langgraph import StateGraph, START, END
# from langgraph.prebuilt import ToolNode
# from typing import TypedDict


def create_langchain_tools_from_mcp(server_url: str):
    """
    Create LangChain-compatible tools from MCP server.

    This function shows how to wrap generic MCP tools for LangGraph/LangChain usage.
    """
    # This would be uncommented if LangGraph was available
    # tools = gym_mcp.create_tools(server_url)

    # Mock tools for demonstration
    tools = [
        type('MockTool', (), {
            '__name__': 'reset_env',
            '__doc__': 'Reset the environment',
            '__call__': lambda self, seed=None: {'success': True, 'observation': 'Environment reset'}
        })(),
        type('MockTool', (), {
            '__name__': 'step_env',
            '__doc__': 'Take an action in the environment',
            '__call__': lambda self, action: {'success': True, 'reward': 0.1, 'done': False}
        })(),
    ]

    # Wrap tools for LangChain
    # @tool
    # def reset_env(seed: Optional[int] = None) -> str:
    #     """Reset the environment to start a new episode."""
    #     result = tools[0](seed=seed)
    #     return f"Environment reset: {result}"

    # @tool
    # def step_env(action: str) -> str:
    #     """Take an action in the environment."""
    #     result = tools[1](action=action)
    #     return f"Action result: {result}"

    # return [reset_env, step_env]

    # For demonstration, return mock wrapped tools
    return tools


def main():
    """Demonstrate MCP tools with LangGraph."""
    print("AgentRing MCP Tools with LangGraph Example")
    print("=" * 46)
    print()

    # Server URL (change this to your running gym-mcp-server)
    server_url = "http://localhost:8070"

    try:
        print("This example shows how to integrate AgentRing MCP tools with LangGraph.")
        print("To run this example, you need to install LangGraph:")
        print("  pip install langgraph langchain-core")
        print()
        print("Then uncomment the LangGraph imports at the top of this file.")
        print()

        # Create LangChain-compatible tools
        print("1. Creating LangChain-compatible tools from MCP server...")
        langchain_tools = create_langchain_tools_from_mcp(server_url)
        print(f"✓ Created {len(langchain_tools)} tools")
        print()

        # Example LangGraph workflow (commented out since not installed)
        print("2. LangGraph Workflow Definition (uncomment when LangGraph is installed):")
        print("""
        # Define state
        class AgentState(TypedDict):
            messages: list
            step_count: int
            total_reward: float
            done: bool

        # Create tools
        tools = create_langchain_tools_from_mcp("http://localhost:8070")
        tool_node = ToolNode(tools)

        # Define workflow
        def agent_node(state: AgentState):
            # Agent logic here (LLM call with tools)
            messages = state["messages"]
            # ... LLM call with tool calling ...
            return {"messages": messages, "step_count": state["step_count"] + 1}

        def should_continue(state: AgentState) -> str:
            if state["done"] or state["step_count"] >= 50:
                return END
            return "tools"

        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges("agent", should_continue)

        app = workflow.compile()

        # Run workflow
        initial_state = {
            "messages": [{"role": "user", "content": "Complete the text adventure quest"}],
            "step_count": 0,
            "total_reward": 0.0,
            "done": False
        }

        result = app.invoke(initial_state)
        print(f"Workflow completed with {result['step_count']} steps")
        """)

        print("✓ Example completed (LangGraph integration pattern shown)")

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
