#!/usr/bin/env python3
"""
MCP Tools with CrewAI Example

This example shows how to use AgentRing's generic MCP tools with CrewAI.
Note: This requires installing CrewAI separately (not included in AgentRing).
"""

# import agentring.mcp as gym_mcp

# Uncomment the lines below if you have CrewAI installed
# from crewai import Agent, Task, Crew
# from crewai.tools import tool


def create_crewai_tools_from_mcp(server_url: str):
    """
    Create CrewAI-compatible tools from MCP server.

    This function shows how to wrap generic MCP tools for CrewAI usage.
    """
    # This would be uncommented if CrewAI was available
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

    # Wrap tools for CrewAI
    # @tool
    # def reset_env(seed=None):
    #     """Reset the environment to start a new episode."""
    #     return tools[0](seed=seed)

    # @tool
    # def step_env(action):
    #     """Take an action in the environment."""
    #     return tools[1](action=action)

    # return [reset_env, step_env]

    # For demonstration, return mock wrapped tools
    return tools


def main():
    """Demonstrate MCP tools with CrewAI."""
    print("AgentRing MCP Tools with CrewAI Example")
    print("=" * 45)
    print()

    # Server URL (change this to your running gym-mcp-server)
    server_url = "http://localhost:8070"

    try:
        print("This example shows how to integrate AgentRing MCP tools with CrewAI.")
        print("To run this example, you need to install CrewAI:")
        print("  pip install crewai")
        print()
        print("Then uncomment the CrewAI imports at the top of this file.")
        print()

        # Create CrewAI-compatible tools
        print("1. Creating CrewAI-compatible tools from MCP server...")
        crewai_tools = create_crewai_tools_from_mcp(server_url)
        print(f"✓ Created {len(crewai_tools)} tools")
        print()

        # Example CrewAI agent (commented out since CrewAI not installed)
        print("2. CrewAI Agent Definition (uncomment when CrewAI is installed):")
        print("""
        # Create CrewAI agent
        agent = Agent(
            role="Text Adventure Agent",
            goal="Complete quests and solve puzzles in text-based environments",
            backstory=\"\"\"You are an expert at playing text adventure games.
            You carefully read descriptions, make logical decisions, and
            systematically explore environments to achieve objectives.\"\"\",
            tools=crewai_tools,
            verbose=True
        )

        # Create task
        task = Task(
            description=\"\"\"Navigate the environment, find the treasure,
            and return it to the starting location. Be methodical and
            examine objects before using them.\"\"\",
            agent=agent,
            expected_output="A summary of the completed quest"
        )

        # Create and run crew
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )

        result = crew.kickoff()
        print(f"Quest completed: {result}")
        """)

        print("✓ Example completed (CrewAI integration pattern shown)")

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
