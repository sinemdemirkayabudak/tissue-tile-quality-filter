"""Image quality assessment functions for histology tiles.

This module provides functions to compute quality metrics including blur detection
(via Laplacian variance) and tissue coverage (via adaptive thresholding).
"""

import logging

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, QualityCheckConfig
from .models import ImageQualityMetrics, TissueQualityScore

logger = logging.getLogger(__name__)


def compute_blur_score(image: np.ndarray) -> float:
    """Compute blur score using Laplacian variance.

    The Laplacian variance is a measure of image sharpness. Higher values
    indicate sharper (less blurred) images.

    Args:
        image: Input image as numpy array (BGR or grayscale).

    Returns:
        Laplacian variance (blur score). Blurry images have low edge variance,
        sharp images have high variance. Values typically range from 10-1000+.
        Generally: <100 = blurry, 100-500 = acceptable, >500 = sharp.
        Returns 0.0 for uniform/blank/corrupted images.

    Note:
        - Grayscale conversion is applied if needed
        - Score is sensitive to image content and illumination
        - Edge case: uniform or corrupted images return 0.0
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Handle edge cases: NaN, infinite, or near-zero variance
    # (uniform/blank/corrupted images)
    if np.isnan(laplacian_var) or np.isinf(laplacian_var) or laplacian_var < 1e-6:
        logger.warning(
            "Blur score is invalid/near-zero (%.2e), "
            "likely uniform or corrupted image, returning 0.0",
            laplacian_var,
        )
        return 0.0

    logger.debug("Blur score computed: %.2f", laplacian_var)
    return float(laplacian_var)


def compute_tissue_coverage(image: np.ndarray) -> float:
    """Compute percentage of image containing tissue (non-background).

    Uses Otsu's adaptive thresholding to separate tissue from background.
    This approach is robust to varying staining intensities and lighting
    conditions common in histology imaging.

    Args:
        image: Input image as numpy array in BGR color space.

    Returns:
        Tissue coverage percentage (0-100).

    Note:
        - Uses Otsu's method for automatic threshold computation
        - Adapts to image characteristics (dataset-independent)
        - Inverts binary mask (THRESH_BINARY_INV) to identify tissue as foreground
        - Works with H&E and other histology stains
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Use Otsu's thresholding: automatically finds optimal threshold
    # THRESH_BINARY_INV inverts the result so tissue (darker) = 255 (foreground)
    _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Count tissue pixels (non-zero values in mask)
    tissue_pixels = np.count_nonzero(tissue_mask)
    total_pixels = tissue_mask.size

    coverage_percent = (tissue_pixels / total_pixels) * 100
    return float(coverage_percent)


def _score_to_level(
    value: float, low_threshold: float, high_threshold: float
) -> TissueQualityScore:
    """Convert a numeric score to a quality level based on thresholds.

    Helper function to reduce code duplication when classifying metrics.

    Args:
        value: The numeric score to classify.
        low_threshold: Below this = LOW, between thresholds = MEDIUM.
        high_threshold: At or above this = HIGH.

    Returns:
        TissueQualityScore: HIGH, MEDIUM, or LOW classification.

    Note:
        Boundaries are inclusive (>=), so values exactly at thresholds
        are classified in the higher category.
    """
    if value >= high_threshold:
        return TissueQualityScore.HIGH
    elif value >= low_threshold:
        return TissueQualityScore.MEDIUM
    else:
        return TissueQualityScore.LOW


def determine_quality_rating(
    blur_score: float,
    tissue_coverage: float,
    config: QualityCheckConfig = DEFAULT_CONFIG,
) -> TissueQualityScore:
    """Determine tissue quality rating based on metrics.

    Applies configurable heuristic rules combining blur and tissue coverage:
    - HIGH: Good sharpness AND sufficient tissue
    - MEDIUM: Acceptable metrics with minor issues
    - LOW: Poor blur OR inadequate tissue coverage

    Args:
        blur_score: Laplacian variance (higher = sharper).
        tissue_coverage: Tissue percentage (0-100).
        config: QualityCheckConfig instance with classification thresholds.

    Returns:
        TissueQualityScore enum value (HIGH, MEDIUM, or LOW).

    Example:
        >>> from .config import QualityCheckConfig, BlurThresholds
        >>> custom_config = QualityCheckConfig(
        ...     blur_thresholds=BlurThresholds(low=100, high=200)
        ... )
        >>> rating = determine_quality_rating(250, 85, config=custom_config)
    """
    bt = config.blur_thresholds
    tt = config.tissue_thresholds

    # Classify blur and tissue metrics using helper function
    blur_quality = _score_to_level(blur_score, bt.low, bt.high)
    tissue_quality = _score_to_level(tissue_coverage, tt.low, tt.high)

    # Combined scoring: both must be HIGH for overall HIGH rating
    if blur_quality == TissueQualityScore.HIGH and tissue_quality == TissueQualityScore.HIGH:
        return TissueQualityScore.HIGH
    elif blur_quality == TissueQualityScore.LOW or tissue_quality == TissueQualityScore.LOW:
        return TissueQualityScore.LOW
    else:
        return TissueQualityScore.MEDIUM


def compute_tile_quality_metrics(
    image: np.ndarray, config: QualityCheckConfig = DEFAULT_CONFIG
) -> ImageQualityMetrics:
    """Compute all quality metrics for a tile image.

    Computes blur score and tissue coverage, determines rating
    using the provided configuration.

    Args:
        image: Tile image as numpy array (BGR or grayscale).
        config: QualityCheckConfig instance with classification thresholds.

    Returns:
        ImageQualityMetrics object with all computed metrics.

    Raises:
        ValueError: If image is invalid or empty.
    """
    if image is None or image.size == 0:
        raise ValueError("Image is None or empty")

    blur_score = compute_blur_score(image)
    tissue_coverage = compute_tissue_coverage(image)
    quality_rating = determine_quality_rating(blur_score, tissue_coverage, config=config)

    logger.debug(
        "Tile - Blur: %.2f, Tissue: %.1f%%, Rating: %s",
        blur_score,
        tissue_coverage,
        quality_rating,
    )

    return ImageQualityMetrics(
        blur_score=blur_score,
        tissue_coverage=tissue_coverage,
        quality_rating=quality_rating,
    )
