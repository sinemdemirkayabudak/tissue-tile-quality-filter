# Tissue Tile Quality Filter Pipeline

A production-grade Python pipeline for automated quality assessment of histology image tiles. Detects blur, tissue coverage, and filters tiles based on configurable quality thresholds.

---

## Installation

### From PyPI (recommended)
```bash
pip install tissue-tile-quality-filter
```

### From source (development)
```bash
git clone https://github.com/sinemdemirkayabudak/tissue-tile-quality-filter.git
cd tissue-tile-quality-filter
uv sync
```

---

## Usage

There are two ways to use this package: **CLI** for quick batch processing, or **Python API** for integration into your own code.

### CLI Usage

Process a directory of tiles from the command line:

```bash
# Basic usage (installed via pip)
tissue-tile-quality-filter process ./tiles

# With options
tissue-tile-quality-filter process ./tiles \
  --batch-id my_batch \
  --threshold high \
  --output ./results \
  --workers 4 \
  --verbose

# View quality thresholds
tissue-tile-quality-filter info
```

If running from source (development):
```bash
# Development usage (with uv)
uv run tissue-tile-quality-filter process ./tiles --verbose
```

**Options:**
- `--output, -o`: Output directory (default: `./results`)
- `--batch-id, -b`: Batch identifier (default: `batch_001`)
- `--threshold, -t`: Quality threshold: `high`, `medium`, `low` (default: `medium`)
- `--workers, -w`: Number of parallel workers (default: auto-detect CPU count)
- `--verbose, -v`: Enable verbose output

### Python API Usage

Integrate the pipeline into your own Python code:

```python
from pathlib import Path
from tissue_tile_quality_filter import TileQualityFilterPipeline, export_to_csv

# Initialize pipeline
pipeline = TileQualityFilterPipeline(
    batch_id="my_batch",
    pass_threshold="medium",
    max_workers=4  # Parallel processing
)

# Process directory
batch_report = pipeline.process_directory(Path("./tiles"))

# Export results
export_to_csv(batch_report, Path("./results/tiles.csv"))

# Access batch statistics
print(f"Passed: {batch_report.passed_tiles}/{batch_report.total_tiles}")
print(f"Pass rate: {batch_report.pass_rate:.1f}%")
```

See [docs/python_api_usage.py](docs/python_api_usage.py) for complete examples.

---

## Docker

Run the pipeline in a containerized environment for reproducibility and portability.

### Using pre-built image (recommended)
```bash
# Pull from Docker registry
docker pull sinembudak/tissue-tile-quality-filter:v0.1.4

# Process tiles using volume mount
docker run --rm \
  -v ./example_tiles/synthetic_tiles:/input:ro \
  -v ./docker_results:/output:rw \
  sinembudak/tissue-tile-quality-filter:v0.1.4 \
  process /input --output /output --verbose

# View available commands
docker run --rm sinembudak/tissue-tile-quality-filter:v0.1.4 --help
```

### Building from source
```bash
# Build the image locally
docker build -t tissue-tile-quality-filter:v0.1.4 .

# Run the local build
docker run --rm \
  -v ./example_tiles/synthetic_tiles:/input:ro \
  -v ./docker_results:/output:rw \
  tissue-tile-quality-filter:v0.1.4 \
  process /input --output /output --verbose
```

See [DOCKER.md](DOCKER.md) for complete Docker documentation.
