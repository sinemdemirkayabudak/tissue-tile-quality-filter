"""Export batch quality reports to CSV and Parquet formats.

Provides serialization utilities for BatchQualityReport into pandas DataFrames
and file formats (CSV, Parquet) for downstream analysis and archival.
"""

import logging
from pathlib import Path

import pandas as pd

from .models import BatchQualityReport

logger = logging.getLogger(__name__)


def batch_report_to_dataframe(batch_report: BatchQualityReport) -> pd.DataFrame:
    """Convert BatchQualityReport to flattened pandas DataFrame.

    Each row represents a single tile with all associated metadata and metrics.
    Suitable for CSV/Parquet export and downstream analysis.

    Args:
        batch_report: Batch quality report to convert.

    Returns:
        DataFrame with columns:
        - batch_id: Batch identifier
        - batch_created_at: Datetime (UTC) when batch was created
        - tile_id: Tile identifier
        - file_name: Image filename
        - file_dir: Parent directory of image file
        - file_size_mb: File size in MB
        - height: Image height in pixels (nullable Int32; None if load failed)
        - width: Image width in pixels (nullable Int32; None if load failed)
        - blur_score: Laplacian variance (blur metric)
        - tissue_coverage: Tissue coverage percentage
        - quality_rating: Quality classification (high/medium/low)
        - is_passed: Whether tile passed quality filter
        - notes: Diagnostic notes or failure details
        - processing_timestamp: Datetime (UTC) when tile was processed
    """
    # Define canonical schema to ensure columns exist even for empty batches
    columns = [
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
    ]

    df = pd.DataFrame(
        [
            {
                "batch_id": batch_report.batch_id,
                "batch_created_at": batch_report.created_at,
                "tile_id": report.metadata.tile_id,
                "file_name": report.metadata.file_path.name,
                "file_dir": str(report.metadata.file_path.parent),
                "file_size_mb": report.metadata.file_size_mb,
                "height": (report.metadata.dimensions[0] if report.metadata.dimensions else None),
                "width": (report.metadata.dimensions[1] if report.metadata.dimensions else None),
                "blur_score": report.quality_metrics.blur_score,
                "tissue_coverage": report.quality_metrics.tissue_coverage,
                "quality_rating": report.quality_metrics.quality_rating,
                "is_passed": report.is_passed,
                "notes": report.notes,
                "processing_timestamp": report.metadata.processing_timestamp,
            }
            for report in batch_report.tile_reports
        ],
        columns=columns,
    )

    # Convert datetime columns to proper datetime type with UTC timezone
    # (defensive check avoids redundant conversion if already correct type)
    if df["batch_created_at"].dtype != "datetime64[ns, UTC]":
        df["batch_created_at"] = pd.to_datetime(df["batch_created_at"], utc=True)
    if df["processing_timestamp"].dtype != "datetime64[ns, UTC]":
        df["processing_timestamp"] = pd.to_datetime(df["processing_timestamp"], utc=True)

    # Optimize data types for clean schema and accurate storage
    df["height"] = df["height"].astype("Int32")
    df["width"] = df["width"].astype("Int32")

    logger.debug("DataFrame columns: %s", df.columns.tolist())
    logger.debug("DataFrame dtypes: %s", df.dtypes.to_dict())
    logger.debug("Converted batch report to DataFrame: %d rows", len(df))
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types for efficient storage and analytics.

    Applies consistent type optimizations across all export formats:
    - batch_id: categorical (typically one value per batch)
    - quality_rating: categorical (3 unique values)
    - is_passed: boolean
    - file_size_mb: round to 3 decimals
    - blur_score: round to 2 decimals
    - tissue_coverage: round to 2 decimals

    Args:
        df: DataFrame to optimize.

    Returns:
        DataFrame with optimized data types (modifies in-place and returns).
    """
    df["batch_id"] = df["batch_id"].astype("category")
    df["quality_rating"] = df["quality_rating"].astype("category")
    df["is_passed"] = df["is_passed"].astype("bool")
    df["file_size_mb"] = df["file_size_mb"].round(3)
    df["blur_score"] = df["blur_score"].round(2)
    df["tissue_coverage"] = df["tissue_coverage"].round(2)
    return df


def export_to_csv(batch_report: BatchQualityReport, output_path: Path) -> None:
    """Export batch quality report to CSV file.

    Always creates output file even if batch contains no tiles, ensuring
    downstream pipelines have consistent, predictable file presence.
    Overwrites existing file if present.

    Applies same optimized data types as Parquet via optimize_dtypes()
    for consistency across export formats.

    Args:
        batch_report: Batch quality report to export.
        output_path: Path where CSV file will be written.

    Raises:
        IOError: If file cannot be written.
    """
    try:
        if output_path.exists():
            logger.warning("Overwriting existing file: %s", output_path)

        df = batch_report_to_dataframe(batch_report)
        optimize_dtypes(df)
        df.to_csv(output_path, index=False)
        logger.info(
            "Exported batch report to CSV: %s (%d tiles)",
            output_path,
            len(df),
        )
    except OSError as e:
        logger.error("Failed to write CSV file %s: %s", output_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error exporting to CSV: %s", e)
        raise


def export_to_parquet(batch_report: BatchQualityReport, output_path: Path) -> None:
    """Export batch quality report to Parquet file.

    Parquet format provides compression and efficient columnar storage,
    ideal for large batches and distributed analysis.
    Always creates output file even if batch contains no tiles, ensuring
    downstream pipelines have consistent, predictable file presence.
    Overwrites existing file if present.

    Uses same optimized data types as CSV export via optimize_dtypes():
    - batch_id: categorical (typically one value per batch)
    - batch_created_at/processing_timestamp: datetime64[ns, UTC]
    - height/width: nullable Int32 (preserves integers, supports missing values)
    - quality_rating: categorical (3 unique values: high/medium/low)
    - is_passed: boolean

    Args:
        batch_report: Batch quality report to export.
        output_path: Path where Parquet file will be written.

    Raises:
        ImportError: If pyarrow or fastparquet is not installed.
        IOError: If file cannot be written.
    """
    try:
        if output_path.exists():
            logger.warning("Overwriting existing file: %s", output_path)

        df = batch_report_to_dataframe(batch_report)
        optimize_dtypes(df)
        df.to_parquet(output_path, index=False)
        logger.info(
            "Exported batch report to Parquet: %s (%d tiles)",
            output_path,
            len(df),
        )
    except ImportError as e:
        logger.error("Parquet export requires pyarrow or fastparquet: %s", e)
        raise
    except OSError as e:
        logger.error("Failed to write Parquet file %s: %s", output_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error exporting to Parquet: %s", e)
        raise


def export_summary(batch_report: BatchQualityReport, output_path: Path) -> None:
    """Export batch summary statistics to CSV.

    Useful for quick overview of batch quality metrics.
    Overwrites existing file if present.

    Args:
        batch_report: Batch quality report to summarize.
        output_path: Path where summary CSV will be written.

    Raises:
        IOError: If file cannot be written.
    """
    try:
        if output_path.exists():
            logger.warning("Overwriting existing file: %s", output_path)

        summary_data = {
            "batch_id": [batch_report.batch_id],
            "total_tiles": [batch_report.total_tiles],
            "passed_tiles": [batch_report.passed_tiles],
            "failed_tiles": [batch_report.failed_tiles],
            "pass_rate": [batch_report.pass_rate],
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df["batch_id"] = summary_df["batch_id"].astype("category")
        summary_df.to_csv(output_path, index=False)
        logger.info("Exported batch summary to CSV: %s", output_path)
    except OSError as e:
        logger.error("Failed to write summary CSV file %s: %s", output_path, e)
        raise
    except Exception as e:
        logger.error("Unexpected error exporting summary: %s", e)
        raise


def export_all(batch_report: BatchQualityReport, output_dir: Path) -> None:
    """Export batch quality report to all formats (CSV, Parquet, summary).

    Convenience function that exports to all formats in a single call,
    creating a consistent output directory structure suitable for
    CLI tools and pipeline automation.

    Creates files:
    - output_dir/tiles.csv — Full tile details
    - output_dir/tiles.parquet — Optimized columnar format
    - output_dir/summary.csv — Batch summary statistics

    Args:
        batch_report: Batch quality report to export.
        output_dir: Directory where export files will be written.

    Raises:
        IOError: If directory cannot be created or files cannot be written.
        ImportError: If pyarrow or fastparquet is not installed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    export_to_csv(batch_report, output_dir / "tiles.csv")
    export_to_parquet(batch_report, output_dir / "tiles.parquet")
    export_summary(batch_report, output_dir / "summary.csv")

    logger.info("Exported batch report to all formats in directory: %s", output_dir)
