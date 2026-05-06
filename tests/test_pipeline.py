"""Tests for the tile quality filter pipeline."""

from pathlib import Path

from tissue_tile_quality_filter.models import TissueQualityScore
from tissue_tile_quality_filter.pipeline import TileQualityFilterPipeline


class TestPipelineInitialization:
    """Test pipeline setup and configuration."""

    def test_pipeline_init_defaults(self) -> None:
        """Verify pipeline initializes with defaults."""
        pipeline = TileQualityFilterPipeline(batch_id="test_batch")

        assert pipeline.batch_id == "test_batch"
        assert pipeline.pass_threshold == TissueQualityScore.HIGH

    def test_pipeline_init_enum_threshold(self) -> None:
        """Verify pipeline accepts enum threshold."""
        pipeline = TileQualityFilterPipeline(
            batch_id="test_batch",
            pass_threshold=TissueQualityScore.MEDIUM,
        )

        assert pipeline.pass_threshold == TissueQualityScore.MEDIUM

    def test_pipeline_init_string_threshold(self) -> None:
        """Verify pipeline accepts string threshold."""
        pipeline = TileQualityFilterPipeline(
            batch_id="test_batch",
            pass_threshold="MEDIUM",
        )

        assert pipeline.pass_threshold == TissueQualityScore.MEDIUM

    def test_pipeline_sequential_mode(self) -> None:
        """Verify sequential processing mode."""
        pipeline = TileQualityFilterPipeline(
            batch_id="test_batch",
            max_workers=1,
        )

        assert pipeline.max_workers == 1

    def test_pipeline_parallel_mode(self) -> None:
        """Verify parallel processing mode."""
        pipeline = TileQualityFilterPipeline(
            batch_id="test_batch",
            max_workers=4,
        )

        assert pipeline.max_workers == 4


class TestPipelineProcessing:
    """Test tile processing functionality."""

    def test_process_single_tile(self, temp_tile_file: Path, tmp_path: Path) -> None:
        """Verify single tile processing via directory."""
        import shutil

        # Create temp directory with single tile
        single_tile_dir = tmp_path / "single_tile"
        single_tile_dir.mkdir()
        shutil.copy(temp_tile_file, single_tile_dir / "tile.png")

        pipeline = TileQualityFilterPipeline(batch_id="test_single")
        batch_report = pipeline.process_directory(single_tile_dir)

        assert batch_report.total_tiles == 1
        assert len(batch_report.tile_reports) == 1
        tile_report = batch_report.tile_reports[0]
        assert tile_report.metadata.tile_id is not None
        assert tile_report.metadata.file_path.name == "tile.png"
        assert tile_report.quality_metrics.blur_score >= 0
        assert 0 <= tile_report.quality_metrics.tissue_coverage <= 100
        assert tile_report.is_passed is not None

    def test_process_directory(self, sample_tiles_directory: Path) -> None:
        """Verify batch directory processing."""
        pipeline = TileQualityFilterPipeline(
            batch_id="test_batch",
            pass_threshold=TissueQualityScore.HIGH,
        )

        batch_report = pipeline.process_directory(sample_tiles_directory)

        # Verify batch report
        assert batch_report.batch_id == "test_batch"
        assert batch_report.total_tiles > 0
        assert batch_report.passed_tiles >= 0
        assert batch_report.failed_tiles >= 0
        assert batch_report.total_tiles == (batch_report.passed_tiles + batch_report.failed_tiles)
        assert 0 <= batch_report.pass_rate <= 100

    def test_pass_rate_calculation(self, sample_tiles_directory: Path) -> None:
        """Verify pass rate is calculated correctly."""
        pipeline = TileQualityFilterPipeline(batch_id="test_rate")

        batch_report = pipeline.process_directory(sample_tiles_directory)

        if batch_report.total_tiles > 0:
            expected_rate = (batch_report.passed_tiles / batch_report.total_tiles) * 100
            # Allow small floating-point differences
            assert abs(batch_report.pass_rate - expected_rate) < 0.01

    def test_tile_reports_populated(self, sample_tiles_directory: Path) -> None:
        """Verify tile reports are populated in batch."""
        pipeline = TileQualityFilterPipeline(batch_id="test_tiles")

        batch_report = pipeline.process_directory(sample_tiles_directory)

        assert len(batch_report.tile_reports) > 0
        assert len(batch_report.tile_reports) <= batch_report.total_tiles

        # Each tile report should have valid structure
        for tile_report in batch_report.tile_reports:
            assert tile_report.metadata.tile_id is not None
            assert tile_report.quality_metrics.blur_score >= 0
            assert 0 <= tile_report.quality_metrics.tissue_coverage <= 100


class TestThresholdFiltering:
    """Test quality threshold filtering."""

    def test_high_threshold_filtering(self, sample_tiles_directory: Path) -> None:
        """Verify HIGH threshold filtering."""
        pipeline_high = TileQualityFilterPipeline(
            batch_id="test_high",
            pass_threshold=TissueQualityScore.HIGH,
        )

        batch_report = pipeline_high.process_directory(sample_tiles_directory)

        # All passed tiles should have HIGH quality rating
        for tile_report in batch_report.tile_reports:
            if tile_report.is_passed:
                assert tile_report.quality_metrics.quality_rating == TissueQualityScore.HIGH

    def test_medium_threshold_filtering(self, sample_tiles_directory: Path) -> None:
        """Verify MEDIUM threshold filtering."""
        pipeline_medium = TileQualityFilterPipeline(
            batch_id="test_medium",
            pass_threshold=TissueQualityScore.MEDIUM,
        )

        batch_report = pipeline_medium.process_directory(sample_tiles_directory)

        # All passed tiles should have MEDIUM or HIGH rating
        for tile_report in batch_report.tile_reports:
            if tile_report.is_passed:
                assert tile_report.quality_metrics.quality_rating in (
                    TissueQualityScore.HIGH,
                    TissueQualityScore.MEDIUM,
                )

    def test_threshold_affects_pass_count(self, sample_tiles_directory: Path) -> None:
        """Verify different thresholds affect pass count."""
        pipeline_high = TileQualityFilterPipeline(
            batch_id="test_high",
            pass_threshold=TissueQualityScore.HIGH,
        )
        pipeline_low = TileQualityFilterPipeline(
            batch_id="test_low",
            pass_threshold=TissueQualityScore.LOW,
        )

        batch_high = pipeline_high.process_directory(sample_tiles_directory)
        batch_low = pipeline_low.process_directory(sample_tiles_directory)

        # Lower threshold should pass at least as many tiles
        assert batch_low.passed_tiles >= batch_high.passed_tiles


class TestProcessingModes:
    """Test sequential vs parallel processing."""

    def test_sequential_vs_parallel_consistency(self, sample_tiles_directory: Path) -> None:
        """Verify sequential and parallel modes produce same results."""
        pipeline_seq = TileQualityFilterPipeline(
            batch_id="test_seq",
            max_workers=1,
        )
        pipeline_par = TileQualityFilterPipeline(
            batch_id="test_par",
            max_workers=4,
        )

        batch_seq = pipeline_seq.process_directory(sample_tiles_directory)
        batch_par = pipeline_par.process_directory(sample_tiles_directory)

        # Same batch stats (allowing for floating-point precision)
        assert batch_seq.total_tiles == batch_par.total_tiles
        assert batch_seq.passed_tiles == batch_par.passed_tiles
        assert batch_seq.failed_tiles == batch_par.failed_tiles
        assert abs(batch_seq.pass_rate - batch_par.pass_rate) < 0.01
