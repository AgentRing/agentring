# AgentRing

A unified Python client for working with both local and remote Gymnasium environments via [gym-mcp-server](https://github.com/haggaishachar/gym-mcp-server).

**Goal**: Write code once, seamlessly switch between local development and remote execution.

## Features

- 🎮 **Unified API**: Same Gymnasium interface for both local and remote environments
- 🔄 **Seamless Switching**: Change modes with a single parameter
- 🌐 **Remote Execution**: Connect to gym-mcp-server instances over HTTP
- 🤖 **MCP Extensions**: Generic MCP tool generation and agent utilities (SDK-agnostic)
- 🔧 **Full Compatibility**: Supports all Gymnasium environment types (Box, Discrete, MultiBinary, etc.)
- 🐍 **Modern Python**: Python 3.10+ with complete type hints
- 📦 **Easy Setup**: Managed with uv for fast dependency management
- 📊 **Result Analysis**: Comprehensive episode statistics and export capabilities
- ✅ **Well Tested**: 19 core tests + 7 MCP extension test suites

## Installation

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Complete Example

Here's a complete example showing both modes with multiple episodes:

```python
import agentring as gym

# Choose mode: "local" or "remote"
MODE = "local"  # Change to "remote" to use gym-mcp-server
SERVER_URL = "http://localhost:8000" if MODE == "remote" else None

# Create environment with context manager for automatic cleanup
with gym.make(
    "CartPole-v1",
    mode=MODE,
    gym_server_url=SERVER_URL,
    render_mode="rgb_array"
) as env:
    print(f"Environment: {env}")
    print(f"Mode: {MODE}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()

    # Run 3 episodes
    for episode in range(3):
        observation, info = env.reset(seed=42 + episode)
        total_reward = 0
        steps = 0
        terminated = truncated = False

        print(f"Episode {episode + 1}:")
        print(f"  Initial observation: {observation}")

        while not (terminated or truncated):
            # Sample random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            if steps >= 200:  # Safety limit
                break

        print(f"  Total reward: {total_reward}")
        print(f"  Steps: {steps}")
        print()

print("Done!")
```

## Architecture

```
┌─────────────────┐
│  Your Code      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ gym.make    │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│ Local  │ │ Remote       │
│ Gym    │ │ HTTP Client  │
└────────┘ └──────┬───────┘
              │
              ▼
         ┌──────────────┐
         │ gym-mcp-     │
         │ server       │
         └──────┬───────┘
                │
                ▼
           ┌────────┐
           │ Gym    │
           │ Env    │
           └────────┘
```

### Supported Space Types

| Space Type | Local | Remote | Serialization |
|------------|-------|--------|---------------|
| Box | ✅ | ✅ | array ↔ list |
| Discrete | ✅ | ✅ | int ↔ int |
| MultiBinary | ✅ | ✅ | array ↔ list |
| MultiDiscrete | ✅ | ✅ | array ↔ list |
| Tuple | ✅ | ✅ | recursive |
| Dict | ✅ | ✅ | recursive |

## Examples

See `quickstart.py` for a complete working example showing both local and remote modes.

Run the example:

```bash
# Local mode (default)
uv run python quickstart.py

# Remote mode (start server first!)
python -m gym_mcp_server --env CartPole-v1 --transport streamable-http --port 8000
# Then edit quickstart.py to set REMOTE_MODE = True and run:
uv run python quickstart.py
```

## Development

### Setup

```bash
git clone <your-repo-url>
cd agentring
make install
```

### Available Commands

```bash
make help       # Show all commands
make test       # Run test suite (14 tests)
make lint       # Run ruff linter
make format     # Format code with ruff
make typecheck  # Run mypy type checker
make check      # Run all checks (lint + typecheck + test)
make all        # Format, then run all checks
make demo       # Run local demo
make clean      # Clean build artifacts
```

### Running Tests

```bash
make test
# Or: uv run pytest tests/ -v
# 19 tests (18 passing, 1 failing due to missing pygame) ✅
```

## Use Cases

1. **Development → Production**: Develop locally, deploy remotely
2. **Distributed Training**: Multiple processes connecting to remote environments
3. **Resource Management**: Run expensive simulations on dedicated servers
4. **Testing**: Test locally before remote deployment

## Performance

### Local Mode
- **Overhead**: Minimal (thin wrapper)
- **Best for**: Development, testing, lightweight environments

### Remote Mode
- **Overhead**: HTTP round-trip (1-10ms on localhost)
- **Best for**: Expensive environments, distributed training, resource sharing

## Error Handling

```python
import agentring as gym
import httpx

try:
    env = gym.make(
        "CartPole-v1",
        mode="remote",
        gym_server_url="http://localhost:8000"
    )
    observation, info = env.reset()
    # ... your code ...
except ValueError:
    # Invalid mode, missing URL, etc.
    pass
except RuntimeError:
    # Environment initialization failed, remote call failed
    pass
except httpx.HTTPError:
    # Network error (remote mode only)
    pass
finally:
    if 'env' in locals():
        env.close()
```

## Troubleshooting

### Remote Connection Issues

1. Ensure the gym-mcp-server is running and accessible
2. Check URL is correct (including protocol: `http://` or `https://`)
3. Verify firewall/network settings
4. Check server logs for errors

### Environment Not Found

```bash
# For Atari environments
uv add "gymnasium[atari]"

# For Box2D environments
uv add "gymnasium[box2d]"

# For MuJoCo environments
uv add "gymnasium[mujoco]"
```

## Requirements

- Python 3.10+
- gymnasium >= 1.2.1
- httpx >= 0.28.1
- numpy >= 2.0.0
- gym-mcp-server (from GitHub)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make check` to verify all tests pass
5. Submit a Pull Request

See `CONTRIBUTING.md` for detailed guidelines.

## License

MIT License - see LICENSE file for details.

## MCP Extensions for Agent Development

AgentRing now includes powerful MCP (Model Context Protocol) extensions that dramatically simplify agent development with MCP servers. These extensions provide generic, SDK-agnostic tools and utilities.

### Features

- 🤖 **Generic Tool Factory**: Auto-generate callable tools from MCP servers (works with any agent SDK)
- 🎯 **SDK Agnostic**: No SDK dependencies - tools are standard Python callables
- 🚀 **Episode Runner**: Unified episode execution and result collection
- 🔧 **Format Converters**: Convert tool definitions to JSON Schema, OpenAPI, and SDK-specific formats
- 🌐 **Multi-Server Support**: Work with multiple MCP servers simultaneously
- 📊 **Result Analysis**: Comprehensive episode result statistics and export capabilities

### Quick Start

```python
import agentring.mcp as gym_mcp

# 1. Generate tools from MCP server
tools = gym_mcp.create_tools("http://localhost:8070")
# Returns: List of callable Python functions

# 2. Use with any agent SDK
# Example with CrewAI:
from crewai import Agent, Task, Crew
from crewai.tools import tool

@tool
def reset_env(seed=None):
    return tools[0](seed=seed)

@tool
def step_env(action):
    return tools[1](action=action)

agent = Agent(tools=[reset_env, step_env], ...)
task = Task(description="Complete the household task", agent=agent)
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()

# 3. Or use generic episode runner
runner = gym_mcp.MCPAgentRunner(tools, your_agent_callable)
results = runner.run_episodes(episodes=5)
print(results.summary())
```

### MCP Tool Factory

The `create_tools()` function automatically discovers available tools from MCP servers and generates callable Python functions:

```python
# Generate all tools from server
tools = gym_mcp.create_tools("http://localhost:8070")

# Generate specific tools
tools = gym_mcp.create_tools("http://localhost:8070", ["reset_env", "step_env"])

# Each tool is a standard callable
result = tools[0](seed=42)  # Call reset_env
```

### Episode Runner

Run episodes with any agent interface using the generic runner:

```python
# Works with any callable agent (sync or async)
runner = gym_mcp.MCPAgentRunner(tools, agent_callable)

# Run single episode
result = runner.run_episode(episode_num=1, seed=42)

# Run multiple episodes
results = runner.run_episodes(num_episodes=10)
print(f"Success rate: {results.success_percentage:.1f}%")
```

### Format Converters

Convert tool definitions to various formats for SDK integration:

```python
from agentring.mcp import formats

# Convert to JSON Schema (OpenAI style)
json_schema = formats.to_json_schema(tool_definition)

# Convert to OpenAPI spec
openapi_spec = formats.to_openapi_spec(tool_definition)

# Convert to SDK-specific formats
crewai_format = formats.to_crewai_tool(tool_definition)
langchain_format = formats.to_langchain_tool(tool_definition)
```

### Multi-Server Support

Work with multiple MCP servers:

```python
multi_client = gym_mcp.MultiServerClient()
multi_client.add_server("textworld", "http://localhost:8070")
multi_client.add_server("alfworld", "http://localhost:8090")

# Get tools from all servers
all_tools = multi_client.get_all_tools()

# Health check all servers
health = multi_client.health_check_all()
print(f"Healthy servers: {multi_client.get_healthy_servers()}")
```

### Agent Templates

Pre-built instruction templates for common agent patterns:

```python
from agentring.mcp import templates

# Get templates
text_adventure_prompt = templates.TEXT_ADVENTURE_INSTRUCTIONS
shopping_prompt = templates.SHOPPING_INSTRUCTIONS
household_prompt = templates.HOUSEHOLD_INSTRUCTIONS

# Create complete agent configurations
config = templates.create_text_adventure_config(
    max_steps=50,
    custom_instructions="Always examine objects before using them."
)
```

### Result Analysis

Comprehensive episode result analysis and export:

```python
results = runner.run_episodes(episodes=20)

# Statistics
print(f"Success rate: {results.success_percentage:.1f}%")
print(f"Average reward: {results.average_reward:.2f}")
print(f"Average steps: {results.average_steps:.1f}")

# Export results
results.save_json("results.json")
results.save_csv("results.csv")

# Filter and analyze
successful_episodes = results.filter_by_success(successful_only=True)
high_reward_episodes = results.filter_by_reward(min_reward=1.0)
```

### SDK Integration Examples

#### With CrewAI

```python
import agentring.mcp as gym_mcp
from crewai import Agent, Task, Crew
from crewai.tools import tool

# Generate tools
tools = gym_mcp.create_tools("http://localhost:8070")

# Wrap for CrewAI
@tool
def reset_env(seed=None):
    return tools[0](seed=seed)

@tool
def step_env(action):
    return tools[1](action=action)

agent = Agent(
    role="Text Adventure Agent",
    goal="Complete quests in text worlds",
    backstory="You excel at solving puzzles and exploring environments.",
    tools=[reset_env, step_env]
)

task = Task(description="Find the treasure and escape the dungeon", agent=agent)
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

#### With LangGraph

```python
import agentring.mcp as gym_mcp
from langchain_core.tools import tool
from langgraph import StateGraph

# Generate and wrap tools
tools = gym_mcp.create_tools("http://localhost:8070")

@tool
def reset_env(seed=None):
    return tools[0](seed=seed)

@tool
def step_env(action):
    return tools[1](action=action)

# Use in LangGraph workflow
# ... workflow definition ...
```

#### With Generic Agent

```python
import agentring.mcp as gym_mcp

# Generate tools
tools = gym_mcp.create_tools("http://localhost:8070")

# Define agent as callable
def my_agent(prompt: str) -> str:
    # Your agent logic here (LLM call, etc.)
    return "I'll reset the environment first by calling reset_env()"

# Run episodes
runner = gym_mcp.MCPAgentRunner(tools, my_agent)
results = runner.run_episodes(episodes=5)
```

## Related Projects

- [gym-mcp-server](https://github.com/haggaishachar/gym-mcp-server) - MCP server for Gymnasium environments
- [Gymnasium](https://gymnasium.farama.org/) - RL environment standard
- [Model Context Protocol](https://modelcontextprotocol.io/) - Tool integration protocol

## Support

- **Issues**: Open a GitHub issue
- **Questions**: Start a GitHub discussion
- **Documentation**: See examples/ directory

---

**Status**: ✅ Production Ready | **Version**: 0.2.0 | **Python**: 3.10+ | **Tests**: 19 core + 7 MCP extension suites
