"""Tests for MCP episode results."""

from agentring.mcp.results import EpisodeResults
from agentring.mcp.types import EpisodeResult


class TestEpisodeResults:
    """Tests for EpisodeResults."""

    def test_initialization(self):
        """Test results initialization."""
        results = [
            EpisodeResult(1, 1.0, 5, True),
            EpisodeResult(2, 0.5, 3, False)
        ]

        episode_results = EpisodeResults(results)

        assert len(episode_results) == 2
        assert episode_results.total_episodes == 2
        assert episode_results.successful_episodes == 1
        assert episode_results.failed_episodes == 1

    def test_statistics(self):
        """Test statistical calculations."""
        results = [
            EpisodeResult(1, 1.0, 5, True),
            EpisodeResult(2, 0.5, 3, False),
            EpisodeResult(3, 2.0, 8, True)
        ]

        episode_results = EpisodeResults(results)

        assert episode_results.success_rate == 2/3
        assert episode_results.total_reward == 3.5
        assert episode_results.average_reward == 3.5 / 3
        assert episode_results.average_steps == 16 / 3
        assert episode_results.min_reward == 0.5
        assert episode_results.max_reward == 2.0

    def test_filtering(self):
        """Test result filtering."""
        results = [
            EpisodeResult(1, 1.0, 5, True),
            EpisodeResult(2, 0.5, 3, False),
            EpisodeResult(3, 2.0, 8, True)
        ]

        episode_results = EpisodeResults(results)

        # Filter successful only
        successful = episode_results.filter_by_success(successful_only=True)
        assert len(successful) == 2
        assert all(r.success for r in successful)

        # Filter by reward
        high_reward = episode_results.filter_by_reward(min_reward=1.0)
        assert len(high_reward) == 2

        # Filter by steps
        short_episodes = episode_results.filter_by_steps(max_steps=5)
        assert len(short_episodes) == 2

    def test_export_formats(self):
        """Test export functionality."""
        results = [
            EpisodeResult(1, 1.0, 5, True, error="test error"),
            EpisodeResult(2, 0.5, 3, False)
        ]

        episode_results = EpisodeResults(results)

        # Test dict export
        data = episode_results.to_dict()
        assert len(data) == 2
        assert data[0]["episode_num"] == 1
        assert data[0]["total_reward"] == 1.0

        # Test JSON export
        json_str = episode_results.to_json()
        assert "summary" in json_str
        assert "episodes" in json_str

        # Test CSV export
        csv_str = episode_results.to_csv()
        assert "episode_num" in csv_str
        assert "1.0" in csv_str

    def test_summary(self):
        """Test summary generation."""
        results = [
            EpisodeResult(1, 1.0, 5, True),
            EpisodeResult(2, 0.5, 3, False)
        ]

        episode_results = EpisodeResults(results)

        summary = episode_results.summary()

        assert summary["total_episodes"] == 2
        assert summary["successful_episodes"] == 1
        assert summary["success_rate"] == 0.5

        # Test with individual results
        summary_with_individual = episode_results.summary(include_individual=True)
        assert "episodes" in summary_with_individual
        assert len(summary_with_individual["episodes"]) == 2

    def test_repr(self):
        """Test string representation."""
        results = [
            EpisodeResult(1, 1.0, 5, True),
            EpisodeResult(2, 0.5, 3, False)
        ]

        episode_results = EpisodeResults(results)

        repr_str = repr(episode_results)
        assert "EpisodeResults" in repr_str
        assert "episodes=2" in repr_str
        assert "success_rate=50.0%" in repr_str

    def test_empty_results(self):
        """Test behavior with empty results."""
        episode_results = EpisodeResults([])

        assert len(episode_results) == 0
        assert episode_results.success_rate == 0.0
        assert episode_results.average_reward == 0.0
        assert episode_results.average_steps == 0.0
