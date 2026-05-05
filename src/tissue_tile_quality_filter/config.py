"""Configuration constants for tissue tile quality assessment.

This module centralizes all configurable thresholds and parameters used
throughout the quality filter pipeline, making it easy to tune and adjust
quality criteria without modifying core logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BlurThresholds:
    """Thresholds for blur score classification.

    Attributes:
        low: Below this value, image is classified as LOW quality (blurry).
        high: Above this value, image is classified as HIGH quality (sharp).
        Between low and high = MEDIUM quality.
    """

    low: float = 80.0
    high: float = 150.0


@dataclass(frozen=True)
class TissueThresholds:
    """Thresholds for tissue coverage classification.

    Attributes:
        low: Below this percentage, tissue coverage is classified as LOW.
        high: Above this percentage, tissue coverage is classified as HIGH.
        Between low and high = MEDIUM quality.
    """

    low: float = 30.0
    high: float = 70.0


@dataclass(frozen=True)
class QualityCheckConfig:
    """Complete configuration for quality checking pipeline.

    Attributes:
        blur_thresholds: Blur score classification thresholds.
        tissue_thresholds: Tissue coverage classification thresholds.
    """

    blur_thresholds: BlurThresholds = BlurThresholds()
    tissue_thresholds: TissueThresholds = TissueThresholds()


# Default global configuration instance
DEFAULT_CONFIG: QualityCheckConfig = QualityCheckConfig()
