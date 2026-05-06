"""Extended tests for improved code coverage.

Tests for error scenarios and edge cases not covered by primary test suites.
"""

from pathlib import Path
from unittest import mock

import pytest

from tissue_tile_quality_filter.exporter import (
    export_summary,
    export_to_csv,
    export_to_parquet,
)
from tissue_tile_quality_filter.models import BatchQualityReport
from tissue_tile_quality_filter.pipeline import TileQualityFilterPipeline


class TestPipelineThresholdValidation:
    """Test pipeline threshold validation edge cases."""

    def test_invalid_threshold_string_raises_error(self) -> None:
        """Verify invalid threshold string raises ValueError."""
        with pytest.raises(ValueError, match="pass_threshold must be one of"):
            TileQualityFilterPipeline(batch_id="test", pass_threshold="invalid_threshold")

    def test_threshold_case_insensitive(self) -> None:
        """Verify threshold accepts lowercase strings."""
        # Should not raise
        pipeline = TileQualityFilterPipeline(batch_id="test", pass_threshold="medium")
        assert pipeline.pass_threshold.value == "medium"

    def test_threshold_uppercase_string(self) -> None:
        """Verify threshold normalizes uppercase strings."""
        # Should work with uppercase (converts to lowercase internally)
        pipeline = TileQualityFilterPipeline(batch_id="test", pass_threshold="HIGH")
        assert pipeline.pass_threshold.value == "high"


class TestExporterErrorHandling:
    """Test exporter error handling for uncovered paths."""

    def test_csv_export_permission_error(
        self, sample_batch_report_mixed: BatchQualityReport
    ) -> None:
        """Verify CSV export raises OSError when file cannot be written."""
        with mock.patch("pathlib.Path.exists", return_value=False):
            # Mock to_csv to raise OSError
            with mock.patch("pandas.DataFrame.to_csv", side_effect=OSError("Permission denied")):
                with pytest.raises(OSError, match="Permission denied"):
                    export_to_csv(sample_batch_report_mixed, Path("/invalid/path/tiles.csv"))

    def test_parquet_export_missing_dependency(
        self, sample_batch_report_mixed: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify Parquet export raises ImportError when dependency missing."""
        output_path = tmp_path / "tiles.parquet"
        with mock.patch(
            "pandas.DataFrame.to_parquet",
            side_effect=ImportError("No module named 'pyarrow'"),
        ):
            with pytest.raises(ImportError, match="No module named"):
                export_to_parquet(sample_batch_report_mixed, output_path)

    def test_parquet_export_permission_error(
        self, sample_batch_report_mixed: BatchQualityReport
    ) -> None:
        """Verify Parquet export handles OSError gracefully."""
        with mock.patch("pandas.DataFrame.to_parquet", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError, match="Permission denied"):
                export_to_parquet(sample_batch_report_mixed, Path("/invalid/path/tiles.parquet"))

    def test_summary_export_permission_error(
        self, sample_batch_report_mixed: BatchQualityReport
    ) -> None:
        """Verify summary export handles OSError gracefully."""
        with mock.patch("pandas.DataFrame.to_csv", side_effect=OSError("Permission denied")):
            with pytest.raises(OSError, match="Permission denied"):
                export_summary(sample_batch_report_mixed, Path("/invalid/path/summary.csv"))

    def test_csv_export_generic_exception(
        self, sample_batch_report_mixed: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify CSV export handles unexpected exceptions."""
        output_path = tmp_path / "tiles.csv"
        with mock.patch("pandas.DataFrame.to_csv", side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(RuntimeError, match="Unexpected error"):
                export_to_csv(sample_batch_report_mixed, output_path)

    def test_parquet_export_generic_exception(
        self, sample_batch_report_mixed: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify Parquet export handles unexpected exceptions."""
        output_path = tmp_path / "tiles.parquet"
        with mock.patch(
            "pandas.DataFrame.to_parquet",
            side_effect=RuntimeError("Unexpected error"),
        ):
            with pytest.raises(RuntimeError, match="Unexpected error"):
                export_to_parquet(sample_batch_report_mixed, output_path)

    def test_summary_export_generic_exception(
        self, sample_batch_report_mixed: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify summary export handles unexpected exceptions."""
        output_path = tmp_path / "summary.csv"
        with mock.patch("pandas.DataFrame.to_csv", side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(RuntimeError, match="Unexpected error"):
                export_summary(sample_batch_report_mixed, output_path)


class TestPipelineDirectoryHandling:
    """Test pipeline directory scanning error handling."""

    def test_process_directory_not_found(self) -> None:
        """Verify processing non-existent directory raises FileNotFoundError."""
        pipeline = TileQualityFilterPipeline(batch_id="test")
        non_existent = Path("/non/existent/directory")

        with pytest.raises(FileNotFoundError, match="Directory not found"):
            pipeline.process_directory(non_existent)

    def test_process_file_not_directory(self, temp_tile_file: Path) -> None:
        """Verify processing a file instead of directory raises NotADirectoryError."""
        pipeline = TileQualityFilterPipeline(batch_id="test")

        with pytest.raises(NotADirectoryError, match="Not a directory"):
            pipeline.process_directory(temp_tile_file)

    def test_process_empty_directory(self, tmp_path: Path) -> None:
        """Verify processing empty directory returns batch with zero tiles."""
        pipeline = TileQualityFilterPipeline(batch_id="test_empty")
        batch_report = pipeline.process_directory(tmp_path)

        assert batch_report.total_tiles == 0
        assert batch_report.passed_tiles == 0
        assert batch_report.failed_tiles == 0
        assert batch_report.pass_rate == 0.0

    def test_process_directory_with_corrupted_image(self, tmp_path: Path) -> None:
        """Verify processing directory with corrupted image creates failure report."""
        # Create a file with invalid image data
        bad_file = tmp_path / "corrupted.png"
        bad_file.write_bytes(b"not a valid png file")

        pipeline = TileQualityFilterPipeline(batch_id="test_corrupted")
        batch_report = pipeline.process_directory(tmp_path)

        assert batch_report.total_tiles == 1
        assert batch_report.failed_tiles == 1
        assert batch_report.passed_tiles == 0
        assert batch_report.tile_reports[0].is_passed is False


class TestPipelineLogging:
    """Test logging for better observability coverage."""

    def test_pipeline_initialization_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify pipeline initialization logs correctly."""
        with caplog.at_level("INFO"):
            TileQualityFilterPipeline(batch_id="test_log", pass_threshold="high")

        assert "Pipeline initialized" in caplog.text
        assert "batch_id=test_log" in caplog.text

    def test_process_directory_logging(
        self, sample_tiles_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify directory processing logs correctly."""
        pipeline = TileQualityFilterPipeline(batch_id="test_log")

        with caplog.at_level("INFO"):
            batch_report = pipeline.process_directory(sample_tiles_directory)

        assert "Scanning directory" in caplog.text
        assert str(sample_tiles_directory) in caplog.text
        assert batch_report.total_tiles > 0


class TestExporterOverwriteWarnings:
    """Test export overwrite warning scenarios."""

    def test_csv_export_overwrite_existing_file(
        self,
        sample_batch_report_mixed: BatchQualityReport,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify CSV export logs warning when overwriting existing file."""
        output_path = tmp_path / "tiles.csv"
        output_path.write_text("dummy,data\n1,2\n")

        with caplog.at_level("WARNING"):
            export_to_csv(sample_batch_report_mixed, output_path)

        assert "Overwriting existing file" in caplog.text
        assert str(output_path) in caplog.text

    def test_parquet_export_overwrite_existing_file(
        self,
        sample_batch_report_mixed: BatchQualityReport,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify Parquet export logs warning when overwriting existing file."""
        output_path = tmp_path / "tiles.parquet"
        output_path.write_bytes(b"dummy parquet content")

        with caplog.at_level("WARNING"):
            export_to_parquet(sample_batch_report_mixed, output_path)

        assert "Overwriting existing file" in caplog.text
        assert str(output_path) in caplog.text

    def test_summary_export_overwrite_existing_file(
        self,
        sample_batch_report_mixed: BatchQualityReport,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify summary export logs warning when overwriting existing file."""
        output_path = tmp_path / "summary.csv"
        output_path.write_text("batch_id,total_tiles\ntest,10\n")

        with caplog.at_level("WARNING"):
            export_summary(sample_batch_report_mixed, output_path)

        assert "Overwriting existing file" in caplog.text
        assert str(output_path) in caplog.text


class TestPipelineProcessingEdgeCases:
    """Test edge cases in pipeline processing."""

    def test_process_directory_with_mixed_file_types(
        self, sample_tiles_directory: Path, tmp_path: Path
    ) -> None:
        """Verify pipeline ignores non-image files."""
        import shutil

        mixed_dir = tmp_path / "mixed"
        mixed_dir.mkdir()

        valid_images = list(sample_tiles_directory.glob("*.png"))
        if valid_images:
            shutil.copy(valid_images[0], mixed_dir / "valid.png")

        (mixed_dir / "readme.txt").write_text("This is a text file")
        (mixed_dir / "config.json").write_text('{"key": "value"}')

        pipeline = TileQualityFilterPipeline(batch_id="test_mixed_files")
        batch_report = pipeline.process_directory(mixed_dir)

        assert batch_report.total_tiles >= 1
        for report in batch_report.tile_reports:
            assert report.metadata.tile_id != "readme"
            assert report.metadata.tile_id != "config"

    def test_process_directory_large_batch(
        self, sample_tiles_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify pipeline logs batch processing information."""
        pipeline = TileQualityFilterPipeline(batch_id="test_large_batch")

        with caplog.at_level("INFO"):
            _ = pipeline.process_directory(sample_tiles_directory)

        assert "Starting batch processing" in caplog.text
        assert "Found" in caplog.text
        assert "valid image files" in caplog.text


class TestThreadPoolExceptionHandling:
    """Test ThreadPoolExecutor exception handling in pipeline."""

    def test_executor_future_exception_from_process_tile(
        self, sample_tiles_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test pipeline handles exceptions from _process_tile via executor."""
        pipeline = TileQualityFilterPipeline(batch_id="test_executor_exception")

        # Get a real tile to work with
        image_paths = list(pipeline._scan_image_files(sample_tiles_directory))
        if not image_paths:
            pytest.skip("No image files found in sample directory")

        # Mock _process_tile to raise an exception (not ValueError, to bypass its catch)
        def raising_process(path):
            raise RuntimeError("Simulated unhandled exception in _process_tile")

        with mock.patch.object(pipeline, "_process_tile", side_effect=raising_process):
            with caplog.at_level("ERROR"):
                batch_report = pipeline.process_directory(sample_tiles_directory)

        # Verify error was logged and failure report was created
        assert "Unexpected error processing" in caplog.text
        assert batch_report.total_tiles == len(image_paths)
        # All should be failures since _process_tile raised exceptions
        assert batch_report.failed_tiles == len(image_paths)

    def test_process_tile_value_error_exception(
        self, sample_tiles_directory: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test pipeline handles ValueError from quality metrics computation."""

        pipeline = TileQualityFilterPipeline(batch_id="test_value_error")

        # Get a real tile to work with
        image_paths = list(pipeline._scan_image_files(sample_tiles_directory))
        if not image_paths:
            pytest.skip("No image files found in sample directory")

        # Mock compute_tile_quality_metrics to raise ValueError
        with mock.patch(
            "tissue_tile_quality_filter.pipeline.compute_tile_quality_metrics",
            side_effect=ValueError("Invalid quality metrics"),
        ):
            with caplog.at_level("WARNING"):
                batch_report = pipeline.process_directory(sample_tiles_directory)

        # Verify ValueError exception handler was triggered
        assert "Failed to process tile" in caplog.text
        assert "Invalid quality metrics" in caplog.text
        # All tiles should be failures due to ValueError
        assert batch_report.failed_tiles == len(image_paths)

    def test_pipeline_reports_all_failures_on_no_images_found(
        self, tmp_path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify pipeline handles empty directory with proper logging."""
        empty_dir = tmp_path / "empty_images"
        empty_dir.mkdir()

        pipeline = TileQualityFilterPipeline(batch_id="test_empty")

        with caplog.at_level("WARNING"):
            batch_report = pipeline.process_directory(empty_dir)

        assert batch_report.total_tiles == 0
        assert batch_report.pass_rate == 0.0
        assert "No valid tiles found" in caplog.text
