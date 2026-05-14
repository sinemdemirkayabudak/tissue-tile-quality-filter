"""CLI entry point for Tissue Tile Quality Filter."""

from pathlib import Path

import click

from . import __version__
from .exporter import export_all
from .pipeline import TileQualityFilterPipeline


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """Tissue Tile Quality Filter - Histology image quality assessment pipeline."""
    pass


@cli.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for results (default: ./results)",
)
@click.option(
    "--batch-id",
    "-b",
    type=str,
    default="batch_001",
    help="Batch identifier (default: batch_001)",
)
@click.option(
    "--threshold",
    "-t",
    type=click.Choice(["high", "medium", "low"], case_sensitive=False),
    default="medium",
    help="Quality threshold for passing tiles (default: medium)",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
def process(
    directory: Path,
    output: Path | None,
    batch_id: str,
    threshold: str,
    workers: int | None,
    verbose: bool,
) -> None:
    """Process a directory of histology tiles and assess quality.

    DIRECTORY: Path to folder containing image tiles (.png, .jpg, .tiff)
    """
    output = output or Path("./results")
    output.mkdir(parents=True, exist_ok=True)

    if verbose:
        click.echo(f"Processing tiles from: {directory}")
        click.echo(f"Output directory: {output}")
        click.echo(f"Batch ID: {batch_id}")
        click.echo(f"Quality threshold: {threshold}")
        click.echo(f"Workers: {workers or 'auto-detect'}")
        click.echo()

    try:
        # Initialize pipeline
        pipeline = TileQualityFilterPipeline(
            batch_id=batch_id,
            pass_threshold=threshold.lower(),
            max_workers=workers,
        )

        # Process directory
        with click.progressbar(length=100, label="Processing tiles", show_pos=True) as bar:
            batch_report = pipeline.process_directory(directory)
            bar.update(100)

        click.echo()

        # Display summary
        click.secho("✓ Processing complete!", fg="green", bold=True)
        click.echo()
        click.echo("Batch Summary:")
        click.echo(f"  Total tiles: {batch_report.total_tiles}")
        click.echo(f"  Passed: {click.style(str(batch_report.passed_tiles), fg='green')}")
        click.echo(f"  Failed: {click.style(str(batch_report.failed_tiles), fg='red')}")
        click.echo(f"  Pass rate: {batch_report.pass_rate:.1f}%")
        click.echo()

        # Export results
        export_all(batch_report, output)
        click.echo(f"Results exported to: {output}")
        click.echo(f"  • {output / 'tiles.csv'}")
        click.echo(f"  • {output / 'tiles.parquet'}")
        click.echo(f"  • {output / 'summary.csv'}")

    except Exception as e:
        click.secho(f"✗ Error: {e}", fg="red", bold=True)
        raise click.Abort() from e


@cli.command()
def info() -> None:
    """Display package information."""
    from . import __version__

    click.echo("Tissue Tile Quality Filter")
    click.echo(f"Version: {__version__}")
    click.echo()
    click.echo("Quality Assessment Thresholds:")
    click.echo("  • HIGH:   blur_score ≥ 100 AND tissue_coverage ≥ 50%")
    click.echo("  • MEDIUM: blur_score ≥ 80 AND tissue_coverage ≥ 30%")
    click.echo("  • LOW:    blur_score ≥ 50 AND tissue_coverage ≥ 10%")
    click.echo()
    click.echo("Supported image formats: PNG, JPG, JPEG, TIFF")
    click.echo()
    click.echo("For help: tissue-tile-quality-filter --help")


if __name__ == "__main__":
    cli()
