"""Tests for MCP agent runner."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from agentring.mcp.runner import MCPAgentRunner
from agentring.mcp.types import EpisodeResult


class TestMCPAgentRunner:
    """Tests for MCPAgentRunner."""

    def test_initialization(self):
        """Test runner initialization."""
        tools = [Mock(__name__="reset_env"), Mock(__name__="step_env")]
        agent = Mock()

        runner = MCPAgentRunner(tools, agent)

        assert runner.tools == tools
        assert runner.agent == agent
        assert runner.max_steps == 50
        assert runner.reset_tool == tools[0]
        assert runner.step_tool == tools[1]

    def test_initialization_missing_tools(self):
        """Test initialization with missing tools."""
        tools = [Mock(__name__="other_tool")]
        agent = Mock()

        with patch('agentring.mcp.runner.logger') as mock_logger:
            runner = MCPAgentRunner(tools, agent)

            mock_logger.warning.assert_any_call("Reset tool 'reset_env' not found")
            mock_logger.warning.assert_any_call("Step tool 'step_env' not found")

    @patch('asyncio.run')
    def test_run_episode_sync(self, mock_asyncio_run):
        """Test synchronous episode running."""
        mock_asyncio_run.return_value = EpisodeResult(
            episode_num=1, total_reward=1.0, num_steps=5, success=True
        )

        tools = [Mock(__name__="reset_env"), Mock(__name__="step_env")]
        agent = Mock()

        runner = MCPAgentRunner(tools, agent)
        result = runner.run_episode(episode_num=1, seed=42)

        assert result.episode_num == 1
        assert result.success is True
        mock_asyncio_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_episode_async(self):
        """Test asynchronous episode running."""
        tools = [Mock(__name__="reset_env"), Mock(__name__="step_env")]
        agent = AsyncMock()

        # Mock tool calls
        tools[0].return_value = {"observation": "reset done"}
        tools[1].return_value = {"reward": 1.0, "terminated": True}

        runner = MCPAgentRunner(tools, agent)

        # Mock agent to return a simple response that doesn't trigger tool calls
        agent.return_value = "episode complete"

        result = await runner.run_episode_async(episode_num=1)

        assert result.episode_num == 1
        assert isinstance(result, EpisodeResult)

    def test_run_episodes(self):
        """Test running multiple episodes."""
        tools = [Mock(__name__="reset_env"), Mock(__name__="step_env")]
        agent = Mock()

        runner = MCPAgentRunner(tools, agent)

        with patch.object(runner, 'run_episode') as mock_run:
            mock_run.side_effect = [
                EpisodeResult(1, 1.0, 5, True),
                EpisodeResult(2, 0.5, 3, False)
            ]

            results = runner.run_episodes(2)

            assert len(results) == 2
            assert results[0].episode_num == 1
            assert results[1].episode_num == 2

    @pytest.mark.asyncio
    async def test_run_episodes_async(self):
        """Test running multiple episodes asynchronously."""
        tools = [Mock(__name__="reset_env"), Mock(__name__="step_env")]
        agent = Mock()

        runner = MCPAgentRunner(tools, agent)

        with patch.object(runner, 'run_episode_async') as mock_run:
            mock_run.side_effect = [
                EpisodeResult(1, 1.0, 5, True),
                EpisodeResult(2, 0.5, 3, False)
            ]

            results = await runner.run_episodes_async(2)

            assert len(results) == 2
            assert results[0].episode_num == 1
            assert results[1].episode_num == 2

    def test_build_initial_prompt(self):
        """Test initial prompt building."""
        tools = [Mock(__name__="reset_env"), Mock(__name__="step_env")]
        agent = Mock()

        runner = MCPAgentRunner(tools, agent)

        prompt = runner._build_initial_prompt(
            instructions="Test instructions",
            initial_prompt="Test prompt",
            observation="Test observation"
        )

        assert "Test instructions" in prompt
        assert "Test prompt" in prompt
        assert "Test observation" in prompt
        assert "reset_env" in prompt
        assert "step_env" in prompt

    def test_parse_tool_calls_simple(self):
        """Test simple tool call parsing."""
        tools = [Mock(__name__="reset_env"), Mock(__name__="step_env")]
        agent = Mock()

        runner = MCPAgentRunner(tools, agent)

        # Test parsing tool calls from text
        response = "I will reset_env(seed=42) and then step_env(action='north')"
        tool_calls = runner._parse_tool_calls_from_response(response)

        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "reset_env"
        assert tool_calls[0]["arguments"] == {"seed": 42}
        assert tool_calls[1]["name"] == "step_env"
        assert tool_calls[1]["arguments"] == {"action": "north"}

    def test_parse_tool_calls_invalid(self):
        """Test parsing invalid tool calls."""
        tools = [Mock(__name__="reset_env")]
        agent = Mock()

        runner = MCPAgentRunner(tools, agent)

        # Test parsing malformed tool calls
        response = "I will reset_env(seed=) and unknown_tool()"
        tool_calls = runner._parse_tool_calls_from_response(response)

        # Should only return valid tool calls
        assert len(tool_calls) == 0  # malformed calls are skipped
