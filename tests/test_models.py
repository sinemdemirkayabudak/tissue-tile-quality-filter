"""Tests for data models and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tissue_tile_quality_filter.models import (
    BatchQualityReport,
    ImageQualityMetrics,
    TileMetadata,
    TileQualityReport,
    TissueQualityScore,
)


class TestTissueQualityScore:
    """Test TissueQualityScore enum."""

    def test_enum_values(self) -> None:
        """Verify enum has correct values."""
        assert TissueQualityScore.HIGH.value == "high"
        assert TissueQualityScore.MEDIUM.value == "medium"
        assert TissueQualityScore.LOW.value == "low"

    def test_enum_from_string(self) -> None:
        """Verify enum can be constructed from string."""
        assert TissueQualityScore("high") == TissueQualityScore.HIGH
        assert TissueQualityScore("medium") == TissueQualityScore.MEDIUM
        assert TissueQualityScore("low") == TissueQualityScore.LOW


class TestImageQualityMetrics:
    """Test ImageQualityMetrics validation."""

    def test_valid_metrics(self) -> None:
        """Verify valid metrics are accepted."""
        metrics = ImageQualityMetrics(
            blur_score=50.0,
            tissue_coverage=75.0,
            quality_rating=TissueQualityScore.HIGH,
        )
        assert metrics.blur_score == 50.0
        assert metrics.tissue_coverage == 75.0
        assert metrics.quality_rating == TissueQualityScore.HIGH

    def test_blur_score_nonnegative(self) -> None:
        """Verify negative blur scores are rejected."""
        with pytest.raises(ValidationError):
            ImageQualityMetrics(
                blur_score=-1.0,
                tissue_coverage=50.0,
                quality_rating=TissueQualityScore.LOW,
            )

    def test_tissue_coverage_range(self) -> None:
        """Verify tissue coverage must be 0-100."""
        with pytest.raises(ValidationError):
            ImageQualityMetrics(
                blur_score=50.0,
                tissue_coverage=150.0,
                quality_rating=TissueQualityScore.LOW,
            )


class TestTileMetadata:
    """Test TileMetadata validation."""

    def test_valid_metadata(self, temp_tile_file: Path) -> None:
        """Verify valid metadata is accepted."""
        metadata = TileMetadata(
            tile_id="test_tile_001",
            file_path=temp_tile_file,
            file_size_mb=1.5,
            dimensions=(512, 512),
        )
        assert metadata.tile_id == "test_tile_001"
        assert metadata.file_path == temp_tile_file
        assert metadata.file_size_mb == 1.5
        assert metadata.dimensions == (512, 512)

    def test_file_exists_validation(self) -> None:
        """Verify non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image file does not exist"):
            TileMetadata(
                tile_id="test_tile",
                file_path=Path("/nonexistent/path/tile.png"),
                file_size_mb=1.0,
            )

    def test_dimensions_validation(self, temp_tile_file: Path) -> None:
        """Verify positive dimension validation."""
        with pytest.raises(ValueError, match="Image dimensions must be positive"):
            TileMetadata(
                tile_id="test_tile",
                file_path=temp_tile_file,
                file_size_mb=1.0,
                dimensions=(-512, 512),
            )

    def test_file_size_positive(self, temp_tile_file: Path) -> None:
        """Verify file size must be positive."""
        with pytest.raises(ValidationError):
            TileMetadata(
                tile_id="test_tile",
                file_path=temp_tile_file,
                file_size_mb=-1.0,
            )


class TestTileQualityReport:
    """Test TileQualityReport."""

    def test_valid_report(
        self, sample_tile_metadata: TileMetadata, sample_quality_metrics_high: ImageQualityMetrics
    ) -> None:
        """Verify valid report creation."""
        report = TileQualityReport(
            metadata=sample_tile_metadata,
            quality_metrics=sample_quality_metrics_high,
            is_passed=True,
            notes="High quality tile",
        )
        assert report.is_passed is True
        assert report.notes == "High quality tile"

    def test_optional_notes(
        self, sample_tile_metadata: TileMetadata, sample_quality_metrics_high: ImageQualityMetrics
    ) -> None:
        """Verify notes field is optional."""
        report = TileQualityReport(
            metadata=sample_tile_metadata,
            quality_metrics=sample_quality_metrics_high,
            is_passed=True,
        )
        assert report.notes is None


class TestBatchQualityReport:
    """Test BatchQualityReport validation."""

    def test_valid_batch_report(self, sample_tile_report_high: TileQualityReport) -> None:
        """Verify valid batch report."""
        batch = BatchQualityReport(
            batch_id="batch_001",
            total_tiles=1,
            passed_tiles=1,
            failed_tiles=0,
            pass_rate=100.0,
            tile_reports=[sample_tile_report_high],
        )
        assert batch.total_tiles == 1
        assert batch.pass_rate == 100.0

    def test_tile_counts_validation(self, sample_tile_report_high: TileQualityReport) -> None:
        """Verify tile count consistency validation."""
        with pytest.raises(ValueError, match="total_tiles .* must equal"):
            BatchQualityReport(
                batch_id="batch_001",
                total_tiles=5,  # Inconsistent with passed + failed
                passed_tiles=1,
                failed_tiles=0,
                pass_rate=100.0,
                tile_reports=[sample_tile_report_high],
            )

    def test_pass_rate_validation(self, sample_tile_report_high: TileQualityReport) -> None:
        """Verify pass rate consistency validation."""
        with pytest.raises(ValueError, match="pass_rate .* does not match"):
            BatchQualityReport(
                batch_id="batch_001",
                total_tiles=1,
                passed_tiles=1,
                failed_tiles=0,
                pass_rate=50.0,  # Inconsistent with 1/1
                tile_reports=[sample_tile_report_high],
            )

    def test_empty_batch_zero_pass_rate(self) -> None:
        """Verify empty batch has 0% pass rate."""
        batch = BatchQualityReport(
            batch_id="batch_empty",
            total_tiles=0,
            passed_tiles=0,
            failed_tiles=0,
            pass_rate=0.0,
        )
        assert batch.total_tiles == 0
        assert batch.pass_rate == 0.0

    def test_empty_batch_nonzero_pass_rate_raises_error(self) -> None:
        """Verify empty batch cannot have non-zero pass rate."""
        with pytest.raises(ValueError, match="pass_rate must be 0.0 when total_tiles is 0"):
            BatchQualityReport(
                batch_id="batch_empty",
                total_tiles=0,
                passed_tiles=0,
                failed_tiles=0,
                pass_rate=50.0,  # Invalid for empty batch
            )

    def test_all_passed_scenario(self, sample_tile_report_high: TileQualityReport) -> None:
        """Verify scenario where all tiles pass."""
        batch = BatchQualityReport(
            batch_id="batch_all_pass",
            total_tiles=3,
            passed_tiles=3,
            failed_tiles=0,
            pass_rate=100.0,
            tile_reports=[sample_tile_report_high] * 3,
        )
        assert batch.passed_tiles == 3
        assert batch.failed_tiles == 0
        assert batch.pass_rate == 100.0

    def test_empty_batch_scenario(self) -> None:
        """Verify empty batch is valid."""
        batch = BatchQualityReport(
            batch_id="batch_empty",
            total_tiles=0,
            passed_tiles=0,
            failed_tiles=0,
            pass_rate=0.0,
        )
        assert len(batch.tile_reports) == 0
        assert batch.pass_rate == 0.0
