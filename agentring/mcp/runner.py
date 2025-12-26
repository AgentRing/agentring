"""Generic MCP Agent Runner - Run episodes with any agent interface."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from agentring.mcp.types import AgentInterface, AsyncAgentInterface, EpisodeResult, ToolCallable

logger = logging.getLogger(__name__)


class MCPAgentRunner:
    """
    Generic agent runner that works with any agent interface.

    This runner can execute episodes using agents from different SDKs by providing
    a simple callable interface. It handles the episode lifecycle, tool orchestration,
    and result collection in a unified way.
    """

    def __init__(
        self,
        tools: List[ToolCallable],
        agent: Union[AgentInterface, AsyncAgentInterface],
        max_steps: int = 50,
        reset_tool_name: str = "reset_env",
        step_tool_name: str = "step_env",
        get_info_tool_name: str = "get_env_info",
    ):
        """
        Initialize the agent runner.

        Args:
            tools: List of callable tools from MCP server
            agent: Agent callable that takes a prompt string and returns a response string
            max_steps: Maximum steps per episode
            reset_tool_name: Name of the reset tool
            step_tool_name: Name of the step tool
            get_info_tool_name: Name of the get_info tool
        """
        self.tools = tools
        self.agent = agent
        self.max_steps = max_steps

        # Find specific tools by name
        self.tool_map = {getattr(tool, '__name__', str(i)): tool for i, tool in enumerate(tools)}
        self.reset_tool = self.tool_map.get(reset_tool_name)
        self.step_tool = self.tool_map.get(step_tool_name)
        self.get_info_tool = self.tool_map.get(get_info_tool_name)

        if not self.reset_tool:
            logger.warning(f"Reset tool '{reset_tool_name}' not found")
        if not self.step_tool:
            logger.warning(f"Step tool '{step_tool_name}' not found")

    def run_episode(
        self,
        episode_num: int = 1,
        seed: Optional[int] = None,
        instructions: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> EpisodeResult:
        """
        Run a single episode synchronously.

        Args:
            episode_num: Episode number for tracking
            seed: Optional random seed
            instructions: Instructions for the agent
            initial_prompt: Initial prompt to start the episode

        Returns:
            EpisodeResult with episode statistics
        """
        return asyncio.run(self.run_episode_async(
            episode_num=episode_num,
            seed=seed,
            instructions=instructions,
            initial_prompt=initial_prompt
        ))

    async def run_episode_async(
        self,
        episode_num: int = 1,
        seed: Optional[int] = None,
        instructions: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> EpisodeResult:
        """
        Run a single episode asynchronously.

        Args:
            episode_num: Episode number for tracking
            seed: Optional random seed
            instructions: Instructions for the agent
            initial_prompt: Initial prompt to start the episode

        Returns:
            EpisodeResult with episode statistics
        """
        logger.info(f"Starting episode {episode_num}")

        total_reward = 0.0
        step_count = 0
        success = False
        error = None
        observation = None

        try:
            # Reset the environment
            if self.reset_tool:
                logger.debug("Resetting environment")
                reset_result = self.reset_tool(seed=seed)
                observation = reset_result
                logger.debug(f"Reset result: {reset_result}")
            else:
                logger.warning("No reset tool available")

            # Build initial prompt
            prompt = self._build_initial_prompt(
                instructions=instructions,
                initial_prompt=initial_prompt,
                observation=observation
            )

            # Run the episode loop
            done = False
            while not done and step_count < self.max_steps:
                step_count += 1
                logger.debug(f"Step {step_count}: Getting agent response")

                try:
                    # Get agent response
                    agent_response = await self._call_agent_async(prompt)
                    logger.debug(f"Agent response: {agent_response}")

                    # Parse and execute tool calls from agent response
                    tool_results = await self._execute_tool_calls_async(agent_response)

                    if tool_results:
                        # Use tool results to build next prompt
                        prompt = self._build_next_prompt(
                            agent_response, tool_results, step_count
                        )

                        # Extract reward and done status from step results
                        for tool_result in tool_results:
                            if isinstance(tool_result, dict):
                                if "reward" in tool_result:
                                    total_reward += float(tool_result["reward"])
                                if tool_result.get("terminated") or tool_result.get("truncated"):
                                    done = True
                                    success = tool_result.get("reward", 0) > 0
                                    break
                    else:
                        # No tool calls found, episode might be complete
                        done = True
                        success = total_reward > 0

                except Exception as e:
                    logger.error(f"Error in step {step_count}: {e}")
                    error = str(e)
                    break

        except Exception as e:
            logger.error(f"Error in episode {episode_num}: {e}")
            error = str(e)

        result = EpisodeResult(
            episode_num=episode_num,
            total_reward=total_reward,
            num_steps=step_count,
            success=success,
            observation=observation,
            error=error
        )

        logger.info(f"Episode {episode_num} completed: {step_count} steps, reward={total_reward:.2f}, success={success}")
        return result

    def run_episodes(
        self,
        num_episodes: int,
        seeds: Optional[List[Optional[int]]] = None,
        instructions: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> List[EpisodeResult]:
        """
        Run multiple episodes synchronously.

        Args:
            num_episodes: Number of episodes to run
            seeds: Optional list of seeds (one per episode)
            instructions: Instructions for the agent
            initial_prompt: Initial prompt for each episode

        Returns:
            List of EpisodeResult objects
        """
        results = []
        for episode_num in range(1, num_episodes + 1):
            seed = seeds[episode_num - 1] if seeds and episode_num <= len(seeds) else None
            result = self.run_episode(
                episode_num=episode_num,
                seed=seed,
                instructions=instructions,
                initial_prompt=initial_prompt
            )
            results.append(result)

        return results

    async def run_episodes_async(
        self,
        num_episodes: int,
        seeds: Optional[List[Optional[int]]] = None,
        instructions: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> List[EpisodeResult]:
        """
        Run multiple episodes asynchronously.

        Args:
            num_episodes: Number of episodes to run
            seeds: Optional list of seeds (one per episode)
            instructions: Instructions for the agent
            initial_prompt: Initial prompt for each episode

        Returns:
            List of EpisodeResult objects
        """
        tasks = []
        for episode_num in range(1, num_episodes + 1):
            seed = seeds[episode_num - 1] if seeds and episode_num <= len(seeds) else None
            task = self.run_episode_async(
                episode_num=episode_num,
                seed=seed,
                instructions=instructions,
                initial_prompt=initial_prompt
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def _call_agent_async(self, prompt: str) -> str:
        """Call the agent, handling both sync and async interfaces."""
        if asyncio.iscoroutinefunction(self.agent):
            return await self.agent(prompt)
        else:
            # Run sync function in thread pool
            return await asyncio.get_event_loop().run_in_executor(None, self.agent, prompt)

    async def _execute_tool_calls_async(self, agent_response: str) -> List[Dict[str, Any]]:
        """Parse agent response and execute any tool calls found."""
        tool_results = []

        # Try to extract tool calls from agent response
        # This is a simplified implementation - real agents would have structured outputs
        tool_calls = self._parse_tool_calls_from_response(agent_response)

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})

            if tool_name in self.tool_map:
                tool = self.tool_map[tool_name]
                try:
                    logger.debug(f"Executing tool: {tool_name} with args: {tool_args}")
                    if asyncio.iscoroutinefunction(tool):
                        result = await tool(**tool_args)
                    else:
                        result = tool(**tool_args)
                    tool_results.append({"tool": tool_name, "result": result})
                    logger.debug(f"Tool result: {result}")
                except Exception as e:
                    logger.error(f"Tool execution failed: {tool_name} - {e}")
                    tool_results.append({"tool": tool_name, "error": str(e)})
            else:
                logger.warning(f"Unknown tool: {tool_name}")

        return tool_results

    def _parse_tool_calls_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from agent response text."""
        # This is a very basic parser - real implementations would use structured outputs
        # from the agent SDK (like OpenAI's tool_calls format)
        tool_calls = []

        # Look for patterns like "reset_env(seed=42)" or "step_env(action='north')"
        import re

        # Simple regex to find function calls
        pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(pattern, response)

        for func_name, args_str in matches:
            if func_name in self.tool_map:
                try:
                    # Parse arguments (very basic)
                    args = {}
                    if args_str.strip():
                        # Split by comma and try to parse
                        for arg_pair in args_str.split(','):
                            if '=' in arg_pair:
                                key, value = arg_pair.split('=', 1)
                                key = key.strip()
                                value = value.strip()

                                # Remove quotes if present
                                if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                                    value = value[1:-1]
                                elif value.isdigit():
                                    value = int(value)
                                elif value.replace('.', '').isdigit():
                                    value = float(value)
                                elif value.lower() in ('true', 'false'):
                                    value = value.lower() == 'true'

                                args[key] = value

                    tool_calls.append({
                        "name": func_name,
                        "arguments": args
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse tool call '{func_name}({args_str})': {e}")

        return tool_calls

    def _build_initial_prompt(
        self,
        instructions: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        observation: Any = None
    ) -> str:
        """Build the initial prompt for the agent."""
        prompt_parts = []

        if instructions:
            prompt_parts.append(f"Instructions: {instructions}")

        if initial_prompt:
            prompt_parts.append(initial_prompt)

        if observation:
            prompt_parts.append(f"Initial observation: {observation}")

        # Add available tools
        tool_names = list(self.tool_map.keys())
        prompt_parts.append(f"Available tools: {', '.join(tool_names)}")

        # Add guidance
        prompt_parts.append(
            "Start by resetting the environment if needed, then take actions to complete the task. "
            "Use the available tools to interact with the environment. "
            f"You have a maximum of {self.max_steps} steps."
        )

        return "\n\n".join(prompt_parts)

    def _build_next_prompt(
        self,
        agent_response: str,
        tool_results: List[Dict[str, Any]],
        step_count: int
    ) -> str:
        """Build the next prompt based on tool results."""
        prompt_parts = []

        # Add previous agent response
        prompt_parts.append(f"Previous response: {agent_response}")

        # Add tool results
        for i, tool_result in enumerate(tool_results):
            tool_name = tool_result.get("tool", f"tool_{i}")
            if "result" in tool_result:
                prompt_parts.append(f"Tool result ({tool_name}): {tool_result['result']}")
            elif "error" in tool_result:
                prompt_parts.append(f"Tool error ({tool_name}): {tool_result['error']}")

        # Add continuation guidance
        prompt_parts.append(
            f"Step {step_count}/{self.max_steps}: Based on the tool results above, "
            "what should you do next? Use tools to continue the task."
        )

        return "\n\n".join(prompt_parts)
