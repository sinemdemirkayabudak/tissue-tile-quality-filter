"""Pydantic models for tile metadata and quality assessment.

This module defines the core data structures for the tissue tile quality filter pipeline.
It uses Pydantic for runtime validation and serialization of tile metadata and quality metrics.
"""

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TissueQualityScore(StrEnum):
    """Enumeration of tissue quality ratings.

    Attributes:
        HIGH: Indicates high-quality tissue with good clarity and coverage.
        MEDIUM: Indicates acceptable quality tissue with minor issues.
        LOW: Indicates poor quality tissue, likely to be filtered out.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImageQualityMetrics(BaseModel):
    """Quality metrics computed from image analysis.

    Stores the quantitative measurements used to assess tissue tile quality,
    including blur detection and tissue coverage analysis.

    Attributes:
        blur_score: Laplacian variance indicating image sharpness.
                   Higher values indicate sharper (less blurred) images.
        tissue_coverage: Percentage of image containing tissue (0-100).
                        Lower values indicate more background noise.
        quality_rating: Overall quality classification (HIGH/MEDIUM/LOW).
    """

    blur_score: float = Field(..., ge=0.0, description="Laplacian variance (blur score)")
    tissue_coverage: float = Field(..., ge=0.0, le=100.0, description="Tissue % coverage")
    quality_rating: TissueQualityScore = Field(..., description="Overall quality rating")

    model_config = ConfigDict(use_enum_values=True)


class TileMetadata(BaseModel):
    """Metadata describing a histology image tile.

    Captures file-level information about a tissue tile including its identifier,
    location, size, and processing timestamp.

    Attributes:
        tile_id: Unique identifier for the tile.
        file_path: Absolute path to the image file on disk.
        file_size_mb: File size in megabytes.
        dimensions: Image dimensions as (height, width) tuple in pixels.
                   None if image could not be loaded (failure case).
        processing_timestamp: ISO 8601 timestamp when the tile was processed.
                             Auto-generated if not provided.
    """

    tile_id: str = Field(..., description="Unique tile identifier")
    file_path: Path = Field(..., description="Path to tile image")
    file_size_mb: float = Field(..., gt=0.0, description="File size in MB")
    dimensions: tuple[int, int] | None = Field(
        None, description="Image dimensions (height, width) or None if load failed"
    )
    processing_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_all(self) -> "TileMetadata":
        """Validate tile metadata: dimensions and file path.

        Performs all validation checks in a single pass for clarity and
        guaranteed execution order.

        Checks:
        - Image dimensions must be positive (height > 0, width > 0) if present.
        - File path must exist on the filesystem.

        Raises:
            ValueError: If dimensions are present but not positive.
            FileNotFoundError: If file does not exist.

        Returns:
            The validated TileMetadata instance.
        """
        # Validate dimensions if present (None is acceptable for failures)
        if self.dimensions is not None:
            height, width = self.dimensions
            if height <= 0 or width <= 0:
                raise ValueError(f"Image dimensions must be positive, got: ({height}, {width})")

        # Validate file path exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {self.file_path.absolute()}")

        return self


class TileQualityReport(BaseModel):
    """Complete quality assessment report for a single tile.

    Combines metadata and quality metrics into a comprehensive report,
    including a pass/fail decision and optional notes.

    Attributes:
        metadata: File and image metadata for the tile.
        quality_metrics: Computed quality measurements and ratings.
        is_passed: Boolean indicating whether tile meets quality thresholds.
        notes: Optional free-form text for additional context or issues.
    """

    metadata: TileMetadata
    quality_metrics: ImageQualityMetrics
    is_passed: bool = Field(..., description="Whether tile passes quality filter")
    notes: str | None = None


class BatchQualityReport(BaseModel):
    """Aggregated quality report for a batch of tiles.

    Provides summary statistics and a collection of individual tile reports.
    Useful for generating batch-level insights and export to CSV/Parquet.

    Attributes:
        batch_id: Unique identifier for the processing batch.
        total_tiles: Total number of tiles processed in the batch.
        passed_tiles: Number of tiles that passed quality filters.
        failed_tiles: Number of tiles that failed quality filters.
        pass_rate: Percentage of tiles that passed (0-100).
        created_at: ISO 8601 timestamp when the report was generated.
        tile_reports: List of individual TileQualityReport objects for each tile.
    """

    batch_id: str = Field(..., description="Unique batch identifier")
    total_tiles: int = Field(..., ge=0)
    passed_tiles: int = Field(..., ge=0)
    failed_tiles: int = Field(..., ge=0)
    pass_rate: float = Field(..., ge=0.0, le=100.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tile_reports: list[TileQualityReport] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tile_counts(self) -> "BatchQualityReport":
        """Validate consistency between tile counts and pass rate.

        Ensures that:
        - total_tiles == passed_tiles + failed_tiles
        - pass_rate matches passed_tiles / total_tiles (if total > 0)

        Raises:
            ValueError: If counts or pass_rate are inconsistent.

        Returns:
            The validated BatchQualityReport instance.
        """
        if self.total_tiles != self.passed_tiles + self.failed_tiles:
            raise ValueError(
                f"total_tiles ({self.total_tiles}) must equal "
                f"passed_tiles ({self.passed_tiles}) + failed_tiles ({self.failed_tiles})"
            )

        if self.total_tiles > 0:
            expected_pass_rate = (self.passed_tiles / self.total_tiles) * 100
            # Allow small floating-point differences (0.01%)
            if abs(self.pass_rate - expected_pass_rate) > 0.01:
                raise ValueError(
                    f"pass_rate ({self.pass_rate}) does not match "
                    f"calculated rate ({expected_pass_rate:.2f})"
                )
        elif self.pass_rate != 0.0:
            raise ValueError("pass_rate must be 0.0 when total_tiles is 0")

        return self
