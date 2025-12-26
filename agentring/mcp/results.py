"""Episode results collection and analysis."""

from __future__ import annotations

import csv
import json
import statistics
from io import StringIO
from typing import Any, Dict, List, Optional, TextIO, Union

from agentring.mcp.types import EpisodeResult


class EpisodeResults:
    """
    Collection of episode results with analysis and export capabilities.

    This class provides comprehensive analysis of agent performance across multiple
    episodes, including statistics, summaries, and export functionality.
    """

    def __init__(self, results: List[EpisodeResult]):
        """
        Initialize with episode results.

        Args:
            results: List of EpisodeResult objects
        """
        self.results = results.copy()

    def __len__(self) -> int:
        """Return number of episodes."""
        return len(self.results)

    def __getitem__(self, index: int) -> EpisodeResult:
        """Get episode result by index."""
        return self.results[index]

    def __iter__(self):
        """Iterate over episode results."""
        return iter(self.results)

    @property
    def total_episodes(self) -> int:
        """Total number of episodes."""
        return len(self.results)

    @property
    def successful_episodes(self) -> int:
        """Number of successful episodes."""
        return sum(1 for r in self.results if r.success)

    @property
    def failed_episodes(self) -> int:
        """Number of failed episodes."""
        return self.total_episodes - self.successful_episodes

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        if not self.results:
            return 0.0
        return self.successful_episodes / self.total_episodes

    @property
    def success_percentage(self) -> float:
        """Success rate as a percentage (0.0 to 100.0)."""
        return self.success_rate * 100.0

    @property
    def total_reward(self) -> float:
        """Sum of all episode rewards."""
        return sum(r.total_reward for r in self.results)

    @property
    def average_reward(self) -> float:
        """Average reward per episode."""
        if not self.results:
            return 0.0
        return self.total_reward / self.total_episodes

    @property
    def total_steps(self) -> int:
        """Sum of all episode steps."""
        return sum(r.num_steps for r in self.results)

    @property
    def average_steps(self) -> float:
        """Average steps per episode."""
        if not self.results:
            return 0.0
        return self.total_steps / self.total_episodes

    @property
    def min_reward(self) -> float:
        """Minimum reward across all episodes."""
        if not self.results:
            return 0.0
        return min(r.total_reward for r in self.results)

    @property
    def max_reward(self) -> float:
        """Maximum reward across all episodes."""
        if not self.results:
            return 0.0
        return max(r.total_reward for r in self.results)

    @property
    def reward_stddev(self) -> float:
        """Standard deviation of rewards."""
        if len(self.results) < 2:
            return 0.0
        rewards = [r.total_reward for r in self.results]
        return statistics.stdev(rewards)

    @property
    def steps_stddev(self) -> float:
        """Standard deviation of steps."""
        if len(self.results) < 2:
            return 0.0
        steps = [r.num_steps for r in self.results]
        return statistics.stdev(steps)

    def summary(self, include_individual: bool = False) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of episode results.

        Args:
            include_individual: Whether to include individual episode results

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "failed_episodes": self.failed_episodes,
            "success_rate": self.success_rate,
            "success_percentage": self.success_percentage,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "total_steps": self.total_steps,
            "average_steps": self.average_steps,
        }

        # Add standard deviations if we have enough data
        if len(self.results) >= 2:
            summary["reward_stddev"] = self.reward_stddev
            summary["steps_stddev"] = self.steps_stddev

        if include_individual:
            summary["episodes"] = [
                {
                    "episode_num": r.episode_num,
                    "total_reward": r.total_reward,
                    "num_steps": r.num_steps,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ]

        return summary

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Convert results to a list of dictionaries.

        Returns:
            List of episode result dictionaries
        """
        return [
            {
                "episode_num": r.episode_num,
                "total_reward": r.total_reward,
                "num_steps": r.num_steps,
                "success": r.success,
                "error": r.error,
                "observation": r.observation,
            }
            for r in self.results
        ]

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Export results to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        data = {
            "summary": self.summary(include_individual=False),
            "episodes": self.to_dict()
        }
        return json.dumps(data, indent=indent, default=str)

    def to_csv(self) -> str:
        """
        Export results to CSV string.

        Returns:
            CSV string representation
        """
        if not self.results:
            return ""

        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "episode_num",
                "total_reward",
                "num_steps",
                "success",
                "error"
            ]
        )

        writer.writeheader()
        for result in self.results:
            writer.writerow({
                "episode_num": result.episode_num,
                "total_reward": result.total_reward,
                "num_steps": result.num_steps,
                "success": result.success,
                "error": result.error or "",
            })

        return output.getvalue()

    def save_json(self, filepath: str, indent: Optional[int] = 2) -> None:
        """
        Save results to JSON file.

        Args:
            filepath: Path to save the file
            indent: JSON indentation level
        """
        with open(filepath, 'w') as f:
            f.write(self.to_json(indent=indent))

    def save_csv(self, filepath: str) -> None:
        """
        Save results to CSV file.

        Args:
            filepath: Path to save the CSV file
        """
        with open(filepath, 'w') as f:
            f.write(self.to_csv())

    def filter_by_success(self, successful_only: bool = True) -> EpisodeResults:
        """
        Filter results by success status.

        Args:
            successful_only: If True, return only successful episodes;
                           if False, return only failed episodes

        Returns:
            New EpisodeResults instance with filtered results
        """
        filtered = [
            r for r in self.results
            if r.success == successful_only
        ]
        return EpisodeResults(filtered)

    def filter_by_reward(self, min_reward: Optional[float] = None, max_reward: Optional[float] = None) -> EpisodeResults:
        """
        Filter results by reward range.

        Args:
            min_reward: Minimum reward (inclusive)
            max_reward: Maximum reward (inclusive)

        Returns:
            New EpisodeResults instance with filtered results
        """
        filtered = []
        for r in self.results:
            if min_reward is not None and r.total_reward < min_reward:
                continue
            if max_reward is not None and r.total_reward > max_reward:
                continue
            filtered.append(r)

        return EpisodeResults(filtered)

    def filter_by_steps(self, min_steps: Optional[int] = None, max_steps: Optional[int] = None) -> EpisodeResults:
        """
        Filter results by step count range.

        Args:
            min_steps: Minimum steps (inclusive)
            max_steps: Maximum steps (inclusive)

        Returns:
            New EpisodeResults instance with filtered results
        """
        filtered = []
        for r in self.results:
            if min_steps is not None and r.num_steps < min_steps:
                continue
            if max_steps is not None and r.num_steps > max_steps:
                continue
            filtered.append(r)

        return EpisodeResults(filtered)

    def __repr__(self) -> str:
        """String representation of the results."""
        return (
            f"EpisodeResults("
            f"episodes={self.total_episodes}, "
            f"success_rate={self.success_percentage:.1f}%, "
            f"avg_reward={self.average_reward:.2f}, "
            f"avg_steps={self.average_steps:.1f}"
            f")"
        )

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        summary = self.summary()

        print("=" * 60)
        print("EPISODE RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Episodes:     {summary['total_episodes']}")
        print(f"Successful:         {summary['successful_episodes']}")
        print(f"Failed:            {summary['failed_episodes']}")
        print(f"Success Rate:       {summary['success_percentage']:.1f}%")
        print()
        print(f"Total Reward:       {summary['total_reward']:.2f}")
        print(f"Average Reward:     {summary['average_reward']:.2f}")
        print(f"Min Reward:         {summary['min_reward']:.2f}")
        print(f"Max Reward:         {summary['max_reward']:.2f}")

        if 'reward_stddev' in summary:
            print(f"Reward StdDev:      {summary['reward_stddev']:.2f}")

        print()
        print(f"Total Steps:        {summary['total_steps']}")
        print(f"Average Steps:      {summary['average_steps']:.1f}")

        if 'steps_stddev' in summary:
            print(f"Steps StdDev:       {summary['steps_stddev']:.1f}")

        print("=" * 60)
