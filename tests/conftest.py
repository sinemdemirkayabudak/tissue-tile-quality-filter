"""Shared test fixtures for unit tests.

Provides reusable test data and utilities for all test modules.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from tissue_tile_quality_filter.models import (
    BatchQualityReport,
    ImageQualityMetrics,
    TileMetadata,
    TileQualityReport,
    TissueQualityScore,
)


@pytest.fixture
def temp_tile_file(tmp_path: Path) -> Path:
    """Create a temporary test image file."""
    import numpy as np
    from PIL import Image

    # Create simple test image (512x512 RGB)
    img_array = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    tile_path = tmp_path / "test_tile.png"
    img.save(tile_path)

    return tile_path


@pytest.fixture
def sample_tile_metadata(temp_tile_file: Path) -> TileMetadata:
    """Create sample tile metadata."""
    return TileMetadata(
        tile_id="tile_001",
        file_path=temp_tile_file,
        file_size_mb=1.5,
        dimensions=(512, 512),
    )


@pytest.fixture
def sample_tile_metadata_no_dims(temp_tile_file: Path) -> TileMetadata:
    """Create sample tile metadata with no dimensions (load failure case)."""
    return TileMetadata(
        tile_id="tile_002",
        file_path=temp_tile_file,
        file_size_mb=1.5,
        dimensions=None,
    )


@pytest.fixture
def sample_quality_metrics_high() -> ImageQualityMetrics:
    """Create high-quality metrics."""
    return ImageQualityMetrics(
        blur_score=150.5,
        tissue_coverage=85.5,
        quality_rating=TissueQualityScore.HIGH,
    )


@pytest.fixture
def sample_quality_metrics_medium() -> ImageQualityMetrics:
    """Create medium-quality metrics."""
    return ImageQualityMetrics(
        blur_score=80.3,
        tissue_coverage=65.2,
        quality_rating=TissueQualityScore.MEDIUM,
    )


@pytest.fixture
def sample_quality_metrics_low() -> ImageQualityMetrics:
    """Create low-quality metrics."""
    return ImageQualityMetrics(
        blur_score=20.1,
        tissue_coverage=30.0,
        quality_rating=TissueQualityScore.LOW,
    )


@pytest.fixture
def sample_tile_report_high(
    sample_tile_metadata: TileMetadata, sample_quality_metrics_high: ImageQualityMetrics
) -> TileQualityReport:
    """Create high-quality tile report."""
    return TileQualityReport(
        metadata=sample_tile_metadata,
        quality_metrics=sample_quality_metrics_high,
        is_passed=True,
        notes="High quality tile, passed all checks",
    )


@pytest.fixture
def sample_tile_report_low(
    sample_tile_metadata_no_dims: TileMetadata, sample_quality_metrics_low: ImageQualityMetrics
) -> TileQualityReport:
    """Create low-quality tile report."""
    return TileQualityReport(
        metadata=sample_tile_metadata_no_dims,
        quality_metrics=sample_quality_metrics_low,
        is_passed=False,
        notes="Low quality, insufficient tissue coverage",
    )


@pytest.fixture
def sample_batch_report_mixed(
    sample_tile_report_high: TileQualityReport, sample_tile_report_low: TileQualityReport
) -> BatchQualityReport:
    """Create batch report with mixed results."""
    return BatchQualityReport(
        batch_id="batch_test_001",
        total_tiles=10,
        passed_tiles=7,
        failed_tiles=3,
        pass_rate=70.0,
        created_at=datetime.now(UTC),
        tile_reports=[sample_tile_report_high, sample_tile_report_low],
    )


@pytest.fixture
def sample_batch_report_all_passed() -> BatchQualityReport:
    """Create batch report where all tiles passed."""
    return BatchQualityReport(
        batch_id="batch_all_pass",
        total_tiles=5,
        passed_tiles=5,
        failed_tiles=0,
        pass_rate=100.0,
        created_at=datetime.now(UTC),
        tile_reports=[],
    )


@pytest.fixture
def sample_batch_report_empty() -> BatchQualityReport:
    """Create empty batch report (no tiles)."""
    return BatchQualityReport(
        batch_id="batch_empty",
        total_tiles=0,
        passed_tiles=0,
        failed_tiles=0,
        pass_rate=0.0,
        created_at=datetime.now(UTC),
        tile_reports=[],
    )


@pytest.fixture
def sample_tiles_directory() -> Path:
    """Get path to synthetic sample tiles (generated earlier)."""
    sample_dir = Path("example_tiles/synthetic_tiles")
    if not sample_dir.exists():
        pytest.skip("example_tiles/synthetic_tiles directory not found")
    return sample_dir
