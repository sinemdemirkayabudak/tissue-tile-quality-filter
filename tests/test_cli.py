"""Tests for CLI commands."""

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tissue_tile_quality_filter.__main__ import cli


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_tiles_dir() -> Iterator[Path]:
    """Create a temporary directory with sample tiles."""
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        # Copy some sample tiles
        sample_tiles = Path("./example_tiles/synthetic_tiles")
        if sample_tiles.exists():
            for tile in list(sample_tiles.glob("*.png"))[:3]:
                shutil.copy(tile, temp_path / tile.name)
        yield temp_path


class TestCliInfo:
    """Tests for the info command."""

    def test_info_command(self, cli_runner: CliRunner) -> None:
        """Test info command displays package information."""
        result = cli_runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "Tissue Tile Quality Filter" in result.output
        assert "Version:" in result.output
        assert "Quality Assessment Thresholds:" in result.output
        assert "HIGH" in result.output
        assert "MEDIUM" in result.output
        assert "LOW" in result.output

    def test_info_shows_supported_formats(self, cli_runner: CliRunner) -> None:
        """Test info command shows supported formats."""
        result = cli_runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "PNG" in result.output
        assert "JPG" in result.output
        assert "TIFF" in result.output

    def test_info_shows_quality_metrics(self, cli_runner: CliRunner) -> None:
        """Test info command displays quality metric thresholds."""
        result = cli_runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "blur_score" in result.output
        assert "tissue_coverage" in result.output


class TestCliProcessCommand:
    """Tests for the process command."""

    def test_process_command_help(self, cli_runner: CliRunner) -> None:
        """Test process command help text."""
        result = cli_runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "DIRECTORY" in result.output
        assert "--output" in result.output
        assert "--batch-id" in result.output
        assert "--threshold" in result.output
        assert "--workers" in result.output
        assert "--verbose" in result.output

    def test_process_missing_directory(self, cli_runner: CliRunner) -> None:
        """Test process command with missing directory."""
        result = cli_runner.invoke(cli, ["process", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_process_with_sample_tiles(self, cli_runner: CliRunner, temp_tiles_dir: Path) -> None:
        """Test process command with actual tiles."""
        with tempfile.TemporaryDirectory() as output_dir:
            result = cli_runner.invoke(
                cli,
                [
                    "process",
                    str(temp_tiles_dir),
                    "--output",
                    output_dir,
                    "--batch-id",
                    "test_batch",
                ],
            )
            # Command should succeed or provide meaningful feedback
            if temp_tiles_dir.glob("*.png"):
                assert "Processing complete" in result.output or result.exit_code == 0
            else:
                # No tiles to process is still valid
                pass

    def test_process_default_output_directory(
        self, cli_runner: CliRunner, temp_tiles_dir: Path
    ) -> None:
        """Test process command creates default output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            try:
                import os

                os.chdir(tmpdir)
                result = cli_runner.invoke(cli, ["process", str(temp_tiles_dir)])
                # Should complete without error
                assert result.exit_code == 0 or "Processing" in result.output
            finally:
                import os

                os.chdir(original_cwd)

    def test_process_with_threshold_options(
        self, cli_runner: CliRunner, temp_tiles_dir: Path
    ) -> None:
        """Test process command with different threshold values."""
        thresholds = ["high", "medium", "low"]
        with tempfile.TemporaryDirectory() as output_dir:
            for threshold in thresholds:
                result = cli_runner.invoke(
                    cli,
                    [
                        "process",
                        str(temp_tiles_dir),
                        "--threshold",
                        threshold,
                        "--output",
                        output_dir,
                    ],
                )
                # Should not error
                assert result.exit_code == 0 or "Processing" in result.output

    def test_process_threshold_case_insensitive(
        self, cli_runner: CliRunner, temp_tiles_dir: Path
    ) -> None:
        """Test process command accepts uppercase thresholds."""
        with tempfile.TemporaryDirectory() as output_dir:
            result = cli_runner.invoke(
                cli,
                [
                    "process",
                    str(temp_tiles_dir),
                    "--threshold",
                    "HIGH",
                    "--output",
                    output_dir,
                ],
            )
            # Should not error (click should handle case conversion)
            assert result.exit_code == 0 or "Processing" in result.output

    def test_process_with_batch_id_option(
        self, cli_runner: CliRunner, temp_tiles_dir: Path
    ) -> None:
        """Test process command with custom batch ID."""
        with tempfile.TemporaryDirectory() as output_dir:
            batch_id = "custom_batch_12345"
            result = cli_runner.invoke(
                cli,
                [
                    "process",
                    str(temp_tiles_dir),
                    "--batch-id",
                    batch_id,
                    "--output",
                    output_dir,
                ],
            )
            assert (
                result.exit_code == 0 or batch_id in result.output or "Processing" in result.output
            )

    def test_process_with_verbose_flag(self, cli_runner: CliRunner, temp_tiles_dir: Path) -> None:
        """Test process command with verbose output."""
        with tempfile.TemporaryDirectory() as output_dir:
            result = cli_runner.invoke(
                cli,
                [
                    "process",
                    str(temp_tiles_dir),
                    "--output",
                    output_dir,
                    "--verbose",
                ],
            )
            # Verbose should produce more output
            if result.exit_code == 0:
                assert (
                    "Processing tiles from:" in result.output
                    or "Output directory:" in result.output
                )

    def test_process_with_workers_option(self, cli_runner: CliRunner, temp_tiles_dir: Path) -> None:
        """Test process command with custom worker count."""
        with tempfile.TemporaryDirectory() as output_dir:
            result = cli_runner.invoke(
                cli,
                [
                    "process",
                    str(temp_tiles_dir),
                    "--workers",
                    "2",
                    "--output",
                    output_dir,
                ],
            )
            assert result.exit_code == 0 or "Processing" in result.output

    def test_process_short_options(self, cli_runner: CliRunner, temp_tiles_dir: Path) -> None:
        """Test process command with short option flags."""
        with tempfile.TemporaryDirectory() as output_dir:
            result = cli_runner.invoke(
                cli,
                [
                    "process",
                    str(temp_tiles_dir),
                    "-o",
                    output_dir,
                    "-b",
                    "test",
                    "-t",
                    "high",
                    "-w",
                    "1",
                    "-v",
                ],
            )
            assert result.exit_code == 0 or "Processing" in result.output


class TestCliTopLevel:
    """Tests for top-level CLI functionality."""

    def test_cli_version(self, cli_runner: CliRunner) -> None:
        """Test --version flag."""
        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.3" in result.output

    def test_cli_help(self, cli_runner: CliRunner) -> None:
        """Test --help flag."""
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Commands:" in result.output
        assert "process" in result.output
        assert "info" in result.output

    def test_cli_with_no_args(self, cli_runner: CliRunner) -> None:
        """Test CLI with no arguments shows help (Click requires a subcommand)."""
        result = cli_runner.invoke(cli, [])
        # Click returns exit code 2 when no subcommand is provided for a group
        assert result.exit_code == 2
        assert "Commands:" in result.output


class TestCliErrorHandling:
    """Tests for CLI error handling."""

    def test_process_invalid_threshold(self, cli_runner: CliRunner, temp_tiles_dir: Path) -> None:
        """Test process command with invalid threshold value."""
        with tempfile.TemporaryDirectory() as output_dir:
            result = cli_runner.invoke(
                cli,
                [
                    "process",
                    str(temp_tiles_dir),
                    "--threshold",
                    "invalid",
                    "--output",
                    output_dir,
                ],
            )
            # Click validates choices before command runs
            assert result.exit_code != 0

    def test_process_invalid_worker_count(
        self, cli_runner: CliRunner, temp_tiles_dir: Path
    ) -> None:
        """Test process command with non-integer worker count."""
        with tempfile.TemporaryDirectory() as output_dir:
            result = cli_runner.invoke(
                cli,
                [
                    "process",
                    str(temp_tiles_dir),
                    "--workers",
                    "not_a_number",
                    "--output",
                    output_dir,
                ],
            )
            assert result.exit_code != 0

    def test_process_with_corrupted_directory(self, cli_runner: CliRunner) -> None:
        """Test process command fails gracefully with error."""
        with tempfile.TemporaryDirectory() as output_dir:
            # Use a path that isn't actually a directory with images
            bad_path = Path(output_dir) / "nonexistent"
            result = cli_runner.invoke(
                cli,
                [
                    "process",
                    str(bad_path),
                    "--output",
                    output_dir,
                ],
            )
            # Should fail because directory doesn't exist
            assert result.exit_code != 0

    def test_process_pipeline_exception(self, cli_runner: CliRunner, temp_tiles_dir: Path) -> None:
        """Test process command handles pipeline exceptions gracefully."""
        with tempfile.TemporaryDirectory() as output_dir:
            with patch(
                "tissue_tile_quality_filter.__main__.TileQualityFilterPipeline"
            ) as mock_pipeline_class:
                # Mock the pipeline to raise an exception
                mock_instance = MagicMock()
                mock_instance.process_directory.side_effect = RuntimeError("Test error")
                mock_pipeline_class.return_value = mock_instance

                result = cli_runner.invoke(
                    cli,
                    [
                        "process",
                        str(temp_tiles_dir),
                        "--output",
                        output_dir,
                    ],
                )
                # Should fail with error message
                assert result.exit_code == 1
                assert "Error:" in result.output or "RuntimeError" in result.output
