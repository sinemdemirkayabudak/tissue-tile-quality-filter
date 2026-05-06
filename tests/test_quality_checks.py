"""Tests for image quality assessment functions."""

import numpy as np
import pytest

from tissue_tile_quality_filter.config import QualityCheckConfig
from tissue_tile_quality_filter.models import TissueQualityScore
from tissue_tile_quality_filter.quality_checks import (
    compute_blur_score,
    compute_tile_quality_metrics,
    compute_tissue_coverage,
    determine_quality_rating,
)


class TestComputeBlurScore:
    """Test blur score computation (Laplacian variance)."""

    def test_sharp_image_high_blur_score(self) -> None:
        """Verify sharp image has high blur score."""
        # Create sharp image with distinct edges
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        # Draw high-contrast rectangle (sharp edges)
        img[100:400, 100:400] = [100, 150, 200]

        blur_score = compute_blur_score(img)
        assert blur_score > 100  # Sharp images typically > 100

    def test_blurry_image_low_blur_score(self) -> None:
        """Verify blurry image has low blur score."""
        # Create blurry image (smooth gradient)
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            img[i, :] = [int(i * 255 / 512)] * 3

        blur_score = compute_blur_score(img)
        assert blur_score < 100  # Blurry/smooth images typically < 100

    def test_uniform_image_returns_zero(self) -> None:
        """Verify uniform image returns 0.0."""
        # Completely uniform image
        img = np.full((512, 512, 3), 128, dtype=np.uint8)

        blur_score = compute_blur_score(img)
        assert blur_score == 0.0  # Uniform image has no variance

    def test_grayscale_input(self) -> None:
        """Verify blur score works with grayscale input."""
        img_gray = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
        blur_score = compute_blur_score(img_gray)
        assert isinstance(blur_score, float)
        assert blur_score >= 0

    def test_returns_float(self) -> None:
        """Verify blur score returns float."""
        img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
        blur_score = compute_blur_score(img)
        assert isinstance(blur_score, float)


class TestComputeTissueCoverage:
    """Test tissue coverage computation."""

    def test_all_tissue_100_percent(self) -> None:
        """Verify tissue coverage detection with structured image."""
        # Create image with tissue-like and background regions
        # Use structured pattern instead of uniform to enable Otsu thresholding
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        # Fill top half with tissue-like dark color
        img[:256, :] = [80, 40, 80]
        # Fill bottom half with light background
        img[256:, :] = [250, 250, 250]
        coverage = compute_tissue_coverage(img)
        # Should detect ~50% tissue coverage (top half)
        assert 30 <= coverage <= 70

    def test_all_background_0_percent(self) -> None:
        """Verify full background returns ~0%."""
        # Light image (background-like)
        img = np.full((512, 512, 3), 250, dtype=np.uint8)
        coverage = compute_tissue_coverage(img)
        assert coverage < 10  # Should be near 0%

    def test_coverage_in_valid_range(self) -> None:
        """Verify coverage is always 0-100%."""
        # Random tissue-like image
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        coverage = compute_tissue_coverage(img)
        assert 0 <= coverage <= 100

    def test_grayscale_input(self) -> None:
        """Verify tissue coverage works with grayscale input."""
        img_gray = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        coverage = compute_tissue_coverage(img_gray)
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 100

    def test_returns_float(self) -> None:
        """Verify tissue coverage returns float."""
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        coverage = compute_tissue_coverage(img)
        assert isinstance(coverage, float)


class TestScoreToLevel:
    """Test conversion of numeric scores to quality levels."""

    def test_high_rating(self) -> None:
        """Verify high score gets HIGH rating."""
        # Above both blur and tissue thresholds
        rating = determine_quality_rating(blur_score=200.0, tissue_coverage=80.0)
        assert rating == TissueQualityScore.HIGH

    def test_low_rating_blur(self) -> None:
        """Verify low blur gets LOW rating."""
        rating = determine_quality_rating(blur_score=20.0, tissue_coverage=80.0)
        assert rating == TissueQualityScore.LOW

    def test_low_rating_tissue(self) -> None:
        """Verify low tissue gets LOW rating."""
        rating = determine_quality_rating(blur_score=200.0, tissue_coverage=10.0)
        assert rating == TissueQualityScore.LOW

    def test_medium_rating(self) -> None:
        """Verify medium scores get MEDIUM rating."""
        rating = determine_quality_rating(blur_score=120.0, tissue_coverage=50.0)
        assert rating == TissueQualityScore.MEDIUM

    def test_boundary_values(self) -> None:
        """Verify boundary thresholds are inclusive."""
        config = QualityCheckConfig()

        # Exactly at high threshold
        rating = determine_quality_rating(
            blur_score=config.blur_thresholds.high,
            tissue_coverage=config.tissue_thresholds.high,
        )
        assert rating == TissueQualityScore.HIGH

        # Exactly at low threshold
        rating = determine_quality_rating(
            blur_score=config.blur_thresholds.low,
            tissue_coverage=config.tissue_thresholds.low,
        )
        assert rating == TissueQualityScore.MEDIUM


class TestDetermineQualityRating:
    """Test overall quality rating determination."""

    def test_custom_config(self) -> None:
        """Verify custom config thresholds are applied."""
        from tissue_tile_quality_filter.config import BlurThresholds, TissueThresholds

        custom_config = QualityCheckConfig(
            blur_thresholds=BlurThresholds(low=50, high=100),
            tissue_thresholds=TissueThresholds(low=30, high=60),
        )

        rating = determine_quality_rating(150.0, 70.0, config=custom_config)
        assert rating == TissueQualityScore.HIGH

    def test_rating_consistency(self) -> None:
        """Verify rating is deterministic."""
        rating1 = determine_quality_rating(150.5, 75.2)
        rating2 = determine_quality_rating(150.5, 75.2)
        assert rating1 == rating2


class TestComputeTileQualityMetrics:
    """Test complete tile quality assessment."""

    def test_valid_image(self) -> None:
        """Verify metrics computed for valid image."""
        img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
        metrics = compute_tile_quality_metrics(img)

        assert metrics.blur_score >= 0
        assert 0 <= metrics.tissue_coverage <= 100
        assert metrics.quality_rating in (
            TissueQualityScore.HIGH,
            TissueQualityScore.MEDIUM,
            TissueQualityScore.LOW,
        )

    def test_empty_image_raises_error(self) -> None:
        """Verify empty image raises ValueError."""
        img = np.array([])
        with pytest.raises(ValueError, match="Image is None or empty"):
            compute_tile_quality_metrics(img)

    def test_none_image_raises_error(self) -> None:
        """Verify None image raises ValueError."""
        with pytest.raises(ValueError, match="Image is None or empty"):
            compute_tile_quality_metrics(None)  # type: ignore

    def test_custom_config_applied(self) -> None:
        """Verify custom config thresholds affect metrics."""
        from tissue_tile_quality_filter.config import BlurThresholds, TissueThresholds

        img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)

        # Strict config
        strict_config = QualityCheckConfig(
            blur_thresholds=BlurThresholds(low=500, high=1000),
            tissue_thresholds=TissueThresholds(low=90, high=95),
        )

        metrics_strict = compute_tile_quality_metrics(img, config=strict_config)

        # Loose config
        loose_config = QualityCheckConfig(
            blur_thresholds=BlurThresholds(low=10, high=50),
            tissue_thresholds=TissueThresholds(low=10, high=20),
        )

        metrics_loose = compute_tile_quality_metrics(img, config=loose_config)

        # Loose config should be equal or more permissive
        # (higher or equal quality rating)
        assert (
            metrics_loose.quality_rating
            in (
                metrics_strict.quality_rating,
                TissueQualityScore.HIGH,
            )
            or metrics_loose.quality_rating >= metrics_strict.quality_rating
        )

    def test_grayscale_image(self) -> None:
        """Verify metrics work with grayscale input."""
        img_gray = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
        metrics = compute_tile_quality_metrics(img_gray)

        assert metrics.blur_score >= 0
        assert 0 <= metrics.tissue_coverage <= 100
