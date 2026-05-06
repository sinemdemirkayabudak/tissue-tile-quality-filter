"""Batch tile processing orchestrator for quality assessment pipeline.

Coordinates scanning image directories, processing tiles individually,
and aggregating results into batch reports with error handling and logging.
Supports parallel tile processing via ThreadPoolExecutor for I/O-bound optimization.
"""

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from .config import DEFAULT_CONFIG, QualityCheckConfig
from .models import (
    BatchQualityReport,
    ImageQualityMetrics,
    TileMetadata,
    TileQualityReport,
    TissueQualityScore,
)
from .quality_checks import _score_to_level, compute_tile_quality_metrics

logger = logging.getLogger(__name__)

# Supported image file extensions
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Quality rating hierarchy for pass/fail determination
RATING_HIERARCHY = {
    TissueQualityScore.LOW: 0,
    TissueQualityScore.MEDIUM: 1,
    TissueQualityScore.HIGH: 2,
}


class TileQualityFilterPipeline:
    """Main orchestrator for batch tile quality assessment.

    Stateless processor: each call to process_directory() is independent.
    Results are returned directly; no state is retained between calls.
    Processes directories of histology tiles, applies quality filters,
    and generates batch reports with configurable thresholds and logging.

    Supports parallel processing via ThreadPoolExecutor for efficient
    handling of large image batches (I/O-bound optimization).
    """

    def __init__(
        self,
        batch_id: str,
        config: QualityCheckConfig = DEFAULT_CONFIG,
        pass_threshold: str | TissueQualityScore = "HIGH",
        max_workers: int | None = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            batch_id: Unique identifier for this batch run.
            config: QualityCheckConfig with classification thresholds.
            pass_threshold: Quality rating required to pass. Can be a string
                ("HIGH", "MEDIUM", "LOW") or TissueQualityScore enum value.
            max_workers: Number of worker threads for parallel processing.
                If None, uses default (typically # of CPUs). Set to 1 for sequential.

        Raises:
            ValueError: If pass_threshold is not a valid TissueQualityScore.
        """
        # Convert string or enum to TissueQualityScore enum
        try:
            if isinstance(pass_threshold, TissueQualityScore):
                self.pass_threshold = pass_threshold
            else:
                self.pass_threshold = TissueQualityScore(pass_threshold.lower())
        except ValueError:
            valid = ", ".join(s.value for s in TissueQualityScore)
            raise ValueError(
                f"pass_threshold must be one of {valid}, got {pass_threshold}"
            ) from None

        self.batch_id = batch_id
        self.config = config
        self.max_workers = max_workers

        logger.info(
            "Pipeline initialized: batch_id=%s, pass_threshold=%s",
            batch_id,
            self.pass_threshold.value,
        )

    def _should_pass(self, quality_rating: TissueQualityScore) -> bool:
        """Determine if a quality rating meets pass threshold.

        Args:
            quality_rating: TissueQualityScore enum value.

        Returns:
            True if rating meets or exceeds pass_threshold, False otherwise.
        """
        threshold_level = RATING_HIERARCHY[self.pass_threshold]
        rating_level = RATING_HIERARCHY[quality_rating]
        return rating_level >= threshold_level

    def _create_failure_report(
        self, image_path: Path, error_msg: str, file_size_mb: float
    ) -> TileQualityReport:
        """Create a failure report for tiles that could not be processed.

        Maintains observability by including failed tiles in results with
        failure details, rather than silently dropping them.

        Args:
            image_path: Path to the tile that failed.
            error_msg: Description of the failure.
            file_size_mb: File size in megabytes (pre-computed to avoid stat call).

        Returns:
            TileQualityReport with is_passed=False and failure details.
        """
        metadata = TileMetadata(
            tile_id=image_path.stem,
            file_path=image_path,
            file_size_mb=file_size_mb,
            dimensions=None,  # Unknown due to load failure
        )

        # Create minimal metrics (all zero, LOW rating)
        failure_metrics = ImageQualityMetrics(
            blur_score=0.0,
            tissue_coverage=0.0,
            quality_rating=TissueQualityScore.LOW,
        )

        return TileQualityReport(
            metadata=metadata,
            quality_metrics=failure_metrics,
            is_passed=False,
            notes=f"Processing failed: {error_msg}",
        )

    def _scan_image_files(self, directory: Path) -> Iterator[Path]:
        """Scan directory for supported image files.

        Args:
            directory: Path to directory to scan.

        Yields:
            Path objects for each supported image file found.

        Raises:
            FileNotFoundError: If directory does not exist.
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        logger.info("Scanning directory: %s", directory)
        # Yield sorted image files as generator (no intermediate list variable)
        yield from sorted(
            (
                f
                for f in directory.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
            ),
            key=lambda x: x.name,
        )

    def _process_tile(self, image_path: Path) -> TileQualityReport:
        """Process a single tile image.

        Always returns a TileQualityReport (passed or failed) to maintain
        observability of all processing attempts, not just successes.

        Args:
            image_path: Path to tile image.

        Returns:
            TileQualityReport with results or failure details.
        """
        # Get file stat once (used in both success and failure paths)
        file_stat = image_path.stat()
        file_size_mb = file_stat.st_size / (1024 * 1024)

        try:
            # Load image once (preserving format: grayscale or color)
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                file_size_kb = file_stat.st_size / 1024
                file_ext = image_path.suffix.lower()
                error_msg = f"Cannot read image (ext={file_ext}, size={file_size_kb:.1f}KB)"
                logger.warning("Failed to load tile: %s - %s", image_path, error_msg)
                return self._create_failure_report(image_path, error_msg, file_size_mb)

            # Extract dimensions (height, width)
            height, width = image.shape[:2]

            # Compute quality metrics (single image load - no re-reading)
            quality_metrics = compute_tile_quality_metrics(image, self.config)

            # Create metadata
            metadata = TileMetadata(
                tile_id=image_path.stem,
                file_path=image_path,
                file_size_mb=file_size_mb,
                dimensions=(height, width),
            )

            # Determine pass/fail
            is_passed = self._should_pass(quality_metrics.quality_rating)

            # Generate detailed notes for explainability (especially important in medical context)
            # Use same classification logic as quality rating for consistency
            blur_level = _score_to_level(
                quality_metrics.blur_score,
                self.config.blur_thresholds.low,
                self.config.blur_thresholds.high,
            )
            tissue_level = _score_to_level(
                quality_metrics.tissue_coverage,
                self.config.tissue_thresholds.low,
                self.config.tissue_thresholds.high,
            )
            blur_status = "✓" if blur_level == TissueQualityScore.HIGH else "✗"
            tissue_status = "✓" if tissue_level == TissueQualityScore.HIGH else "✗"
            notes = (
                f"Blur={quality_metrics.blur_score:.1f} {blur_status} "
                f"({self.config.blur_thresholds.low}-{self.config.blur_thresholds.high}), "
                f"Tissue={quality_metrics.tissue_coverage:.1f}% {tissue_status} "
                f"({self.config.tissue_thresholds.low}-{self.config.tissue_thresholds.high}%), "
                f"Rating={quality_metrics.quality_rating.upper()}"
            )

            # Create report
            report = TileQualityReport(
                metadata=metadata,
                quality_metrics=quality_metrics,
                is_passed=is_passed,
                notes=notes,
            )

            logger.debug(
                "Tile processed: %s (%.2f MB, %dx%d, %s)",
                image_path.name,
                file_size_mb,
                width,
                height,
                quality_metrics.quality_rating,
            )

            return report

        except ValueError as e:
            error_msg = str(e)
            logger.warning("Failed to process tile: %s - %s", image_path, error_msg)
            return self._create_failure_report(image_path, error_msg, file_size_mb)

    def process_directory(self, directory: Path) -> BatchQualityReport:
        """Process all image tiles in a directory.

        Scans directory for image files, processes each one (in parallel
        by default), and aggregates results into a batch report with
        statistics and logging. This method is stateless: results are
        returned directly, no instance state is modified or retained.

        Parallelization: Uses ThreadPoolExecutor for I/O-bound tile loading
        and processing. Override max_workers in __init__ for sequential mode
        (max_workers=1) or custom concurrency levels.

        Args:
            directory: Path to directory containing tile images.

        Returns:
            BatchQualityReport with aggregated results and metadata.

        Raises:
            FileNotFoundError: If directory does not exist.
            NotADirectoryError: If path is not a directory.
        """
        logger.info("Starting batch processing: %s", directory)

        # Collect image paths to process
        image_paths = list(self._scan_image_files(directory))

        # Validate directory is not empty
        if not image_paths:
            logger.warning("No valid tiles found in directory: %s", directory)
        else:
            logger.info("Found %d valid image files to process", len(image_paths))

        # Process all tiles in parallel (or sequentially if max_workers=1)
        # Uses as_completed for better error isolation and extensibility
        # (supports future additions: retries, progress tracking, etc.)
        tile_reports = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_tile, path): path for path in image_paths}
            for future in as_completed(futures):
                image_path = futures[future]
                try:
                    report = future.result()
                    tile_reports.append(report)
                except Exception as e:
                    # This is a safety net; _process_tile always returns a report
                    # even on failure, so we shouldn't reach here normally.
                    logger.error(
                        "Unexpected error processing %s (will create failure report): %s",
                        image_path,
                        e,
                    )
                    # Fall back to creating a failure report
                    file_stat = image_path.stat()
                    file_size_mb = file_stat.st_size / (1024 * 1024)
                    tile_reports.append(
                        self._create_failure_report(image_path, str(e), file_size_mb)
                    )

        # Compute statistics
        total_tiles = len(tile_reports)
        passed_tiles = sum(1 for r in tile_reports if r.is_passed)
        failed_tiles = total_tiles - passed_tiles
        pass_rate = (passed_tiles / total_tiles * 100) if total_tiles > 0 else 0.0

        # Create batch report
        batch_report = BatchQualityReport(
            batch_id=self.batch_id,
            total_tiles=total_tiles,
            passed_tiles=passed_tiles,
            failed_tiles=failed_tiles,
            pass_rate=pass_rate,
            tile_reports=tile_reports,
        )

        logger.info(
            "Batch processing complete: total=%d, passed=%d, failed=%d, pass_rate=%.1f%%",
            total_tiles,
            passed_tiles,
            failed_tiles,
            pass_rate,
        )

        return batch_report
