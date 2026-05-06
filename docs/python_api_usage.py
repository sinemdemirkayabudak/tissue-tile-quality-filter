"""Example usage of the Tissue Tile Quality Filter Pipeline.

Demonstrates:
- Initializing pipeline with custom config
- Processing a batch of tiles
- Exporting results to CSV and Parquet
- Analyzing batch statistics
"""

from pathlib import Path

from tissue_tile_quality_filter import (
    TileQualityFilterPipeline,
    TissueQualityScore,
    batch_report_to_dataframe,
    export_summary,
    export_to_csv,
    export_to_parquet,
)


def main() -> None:
    """Run example pipeline with exports."""
    # Example 1: Sequential vs Parallel Processing
    print("=" * 70)
    print("Example 1: Pipeline Initialization Options")
    print("=" * 70)

    print("\nOption A: Sequential processing (max_workers=1)")
    sequential_pipeline = TileQualityFilterPipeline(
        batch_id="batch_sequential_001",
        pass_threshold=TissueQualityScore.HIGH,  # Accept enum directly
        max_workers=1,  # Sequential
    )
    print(f"  Initialized: {sequential_pipeline.batch_id}")

    print("\nOption B: Parallel processing (default)")
    parallel_pipeline = TileQualityFilterPipeline(
        batch_id="batch_parallel_001",
        pass_threshold="MEDIUM",  # Also accept string
        # max_workers=None uses default (auto-detect CPU count)
    )
    print(f"  Initialized: {parallel_pipeline.batch_id}")

    # Example 2: Process directory and export results
    print("\n" + "=" * 70)
    print("Example 2: Processing and Exporting Results")
    print("=" * 70)

    # Replace with actual tile directory
    tile_directory = Path("./example_tiles/synthetic_tiles")

    if tile_directory.exists():
        # Process all tiles
        batch_report = parallel_pipeline.process_directory(tile_directory)

        # Print summary
        print("\nBatch Summary:")
        print(f"  Total tiles: {batch_report.total_tiles}")
        print(f"  Passed: {batch_report.passed_tiles}")
        print(f"  Failed: {batch_report.failed_tiles}")
        print(f"  Pass rate: {batch_report.pass_rate:.1f}%")

        # Export to files
        output_dir = Path("./results")
        output_dir.mkdir(exist_ok=True)

        csv_path = output_dir / f"{batch_report.batch_id}_tiles.csv"
        parquet_path = output_dir / f"{batch_report.batch_id}_tiles.parquet"
        summary_path = output_dir / f"{batch_report.batch_id}_summary.csv"

        export_to_csv(batch_report, csv_path)
        export_to_parquet(batch_report, parquet_path)
        export_summary(batch_report, summary_path)

        print("\nExports:")
        print(f"  CSV: {csv_path}")
        print(f"  Parquet: {parquet_path}")
        print(f"  Summary: {summary_path}")

        # Example 4: DataFrame manipulation with pandas
        print("\n" + "=" * 70)
        print("Example 4: DataFrame Analysis")
        print("=" * 70)

        df = batch_report_to_dataframe(batch_report)
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Example queries
        failed_tiles = df[~df["is_passed"]]
        print(f"\nFailed tiles: {len(failed_tiles)}")

        low_blur = df[df["blur_score"] < 80]
        print(f"Low blur quality: {len(low_blur)}")

        high_pass_rate_batches = df[df["is_passed"]]
        avg_coverage = high_pass_rate_batches["tissue_coverage"].mean()
        print(f"Avg tissue coverage (passed): {avg_coverage:.1f}%")

    else:
        print(f"\nNote: Sample tile directory '{tile_directory}' not found.")
        print("Create sample tiles in this directory to run the example.")

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
