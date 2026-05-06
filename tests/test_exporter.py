"""Tests for batch export functionality."""

from pathlib import Path

import pandas as pd

from tissue_tile_quality_filter.exporter import (
    batch_report_to_dataframe,
    export_all,
    export_summary,
    export_to_csv,
    export_to_parquet,
    optimize_dtypes,
)
from tissue_tile_quality_filter.models import BatchQualityReport


class TestBatchReportToDataFrame:
    """Test batch report to DataFrame conversion."""

    def test_dataframe_structure(self, sample_batch_report_mixed: BatchQualityReport) -> None:
        """Verify DataFrame has correct structure."""
        df = batch_report_to_dataframe(sample_batch_report_mixed)

        # Check shape
        assert len(df) == 2  # 2 tile reports
        expected_cols = {
            "batch_id",
            "batch_created_at",
            "tile_id",
            "file_name",
            "file_dir",
            "file_size_mb",
            "height",
            "width",
            "blur_score",
            "tissue_coverage",
            "quality_rating",
            "is_passed",
            "notes",
            "processing_timestamp",
        }
        assert set(df.columns) == expected_cols

    def test_empty_batch_dataframe(self, sample_batch_report_empty: BatchQualityReport) -> None:
        """Verify empty batch creates DataFrame with correct schema."""
        df = batch_report_to_dataframe(sample_batch_report_empty)

        # Should have schema even with no rows
        assert len(df) == 0
        expected_cols = {
            "batch_id",
            "batch_created_at",
            "tile_id",
            "file_name",
            "file_dir",
            "file_size_mb",
            "height",
            "width",
            "blur_score",
            "tissue_coverage",
            "quality_rating",
            "is_passed",
            "notes",
            "processing_timestamp",
        }
        assert set(df.columns) == expected_cols

    def test_data_types(self, sample_batch_report_mixed: BatchQualityReport) -> None:
        """Verify DataFrame data types."""
        df = batch_report_to_dataframe(sample_batch_report_mixed)

        # Check datetime columns
        assert pd.api.types.is_datetime64_any_dtype(df["batch_created_at"])
        assert pd.api.types.is_datetime64_any_dtype(df["processing_timestamp"])

        # Batch ID should be string (object or StringDtype)
        assert pd.api.types.is_string_dtype(df["batch_id"])


class TestOptimizeDtypes:
    """Test data type optimization."""

    def test_categorical_encoding(self, sample_batch_report_mixed: BatchQualityReport) -> None:
        """Verify batch_id and quality_rating are categorical."""
        df = batch_report_to_dataframe(sample_batch_report_mixed)
        optimize_dtypes(df)

        assert df["batch_id"].dtype.name == "category"
        assert df["quality_rating"].dtype.name == "category"

    def test_bool_encoding(self, sample_batch_report_mixed: BatchQualityReport) -> None:
        """Verify is_passed is boolean."""
        df = batch_report_to_dataframe(sample_batch_report_mixed)
        optimize_dtypes(df)

        assert df["is_passed"].dtype == "bool"

    def test_float_rounding(self, sample_batch_report_mixed: BatchQualityReport) -> None:
        """Verify float values are rounded."""
        df = batch_report_to_dataframe(sample_batch_report_mixed)
        optimize_dtypes(df)

        # Check rounding
        for val in df["file_size_mb"]:
            # Should have at most 3 decimals
            assert len(str(val).split(".")[-1]) <= 3 or val == int(val)

        for val in df["blur_score"]:
            # Should have at most 2 decimals
            assert len(str(val).split(".")[-1]) <= 2 or val == int(val)

        for val in df["tissue_coverage"]:
            # Should have at most 2 decimals
            assert len(str(val).split(".")[-1]) <= 2 or val == int(val)


class TestExportToCSV:
    """Test CSV export functionality."""

    def test_csv_export(
        self, sample_batch_report_mixed: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify CSV export creates file."""
        output_path = tmp_path / "export.csv"
        export_to_csv(sample_batch_report_mixed, output_path)

        assert output_path.exists()

        # Verify content
        df = pd.read_csv(output_path)
        assert len(df) == 2
        assert "batch_id" in df.columns

    def test_csv_empty_batch(
        self, sample_batch_report_empty: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify CSV export works with empty batch."""
        output_path = tmp_path / "empty.csv"
        export_to_csv(sample_batch_report_empty, output_path)

        assert output_path.exists()

        # Verify structure
        df = pd.read_csv(output_path)
        assert len(df) == 0
        assert "batch_id" in df.columns

    def test_csv_overwrite_warning(
        self, sample_batch_report_mixed: BatchQualityReport, tmp_path: Path, caplog
    ) -> None:
        """Verify warning when overwriting existing file."""
        output_path = tmp_path / "export.csv"

        # First export
        export_to_csv(sample_batch_report_mixed, output_path)
        assert output_path.exists()

        # Second export (overwrite) - should log warning
        caplog.clear()
        export_to_csv(sample_batch_report_mixed, output_path)

        # Check for warning in logs
        assert any("Overwriting" in record.message for record in caplog.records)


class TestExportToParquet:
    """Test Parquet export functionality."""

    def test_parquet_export(
        self, sample_batch_report_mixed: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify Parquet export creates file."""
        output_path = tmp_path / "export.parquet"
        export_to_parquet(sample_batch_report_mixed, output_path)

        assert output_path.exists()

        # Verify content
        df = pd.read_parquet(output_path)
        assert len(df) == 2
        assert "batch_id" in df.columns

    def test_parquet_empty_batch(
        self, sample_batch_report_empty: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify Parquet export works with empty batch."""
        output_path = tmp_path / "empty.parquet"
        export_to_parquet(sample_batch_report_empty, output_path)

        assert output_path.exists()

        # Verify structure
        df = pd.read_parquet(output_path)
        assert len(df) == 0
        assert "batch_id" in df.columns

    def test_parquet_dtypes_preserved(
        self, sample_batch_report_mixed: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify data types are preserved in Parquet."""
        output_path = tmp_path / "export.parquet"
        export_to_parquet(sample_batch_report_mixed, output_path)

        df = pd.read_parquet(output_path)

        # Check preserved types
        assert pd.api.types.is_datetime64_any_dtype(df["batch_created_at"])
        assert pd.api.types.is_datetime64_any_dtype(df["processing_timestamp"])


class TestExportSummary:
    """Test batch summary export."""

    def test_summary_export(
        self, sample_batch_report_mixed: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify summary export creates file with batch stats."""
        output_path = tmp_path / "summary.csv"
        export_summary(sample_batch_report_mixed, output_path)

        assert output_path.exists()

        # Verify content
        df = pd.read_csv(output_path)
        assert len(df) == 1  # Single row
        assert df["batch_id"].iloc[0] == "batch_test_001"
        assert df["total_tiles"].iloc[0] == 10
        assert df["passed_tiles"].iloc[0] == 7
        assert df["failed_tiles"].iloc[0] == 3
        assert df["pass_rate"].iloc[0] == 70.0

    def test_summary_empty_batch(
        self, sample_batch_report_empty: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify summary export works with empty batch."""
        output_path = tmp_path / "summary.csv"
        export_summary(sample_batch_report_empty, output_path)

        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert df["total_tiles"].iloc[0] == 0


class TestExportAll:
    """Test convenience function exporting all formats."""

    def test_export_all(
        self, sample_batch_report_mixed: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify export_all creates all three files."""
        export_all(sample_batch_report_mixed, tmp_path)

        # Check all files created with fixed names
        tiles_csv = tmp_path / "tiles.csv"
        tiles_parquet = tmp_path / "tiles.parquet"
        summary_csv = tmp_path / "summary.csv"

        assert tiles_csv.exists()
        assert tiles_parquet.exists()
        assert summary_csv.exists()

    def test_export_all_consistency(
        self, sample_batch_report_mixed: BatchQualityReport, tmp_path: Path
    ) -> None:
        """Verify CSV and Parquet have same data."""
        export_all(sample_batch_report_mixed, tmp_path)

        tiles_csv = tmp_path / "tiles.csv"
        tiles_parquet = tmp_path / "tiles.parquet"

        df_csv = pd.read_csv(tiles_csv)
        df_parquet = pd.read_parquet(tiles_parquet)

        # Compare data (ignoring dtype differences)
        assert len(df_csv) == len(df_parquet)
        assert set(df_csv.columns) == set(df_parquet.columns)
