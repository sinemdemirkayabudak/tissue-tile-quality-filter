# Docker Usage Guide

This document explains how to build and run the Tissue Tile Quality Filter using Docker.

## Why Docker?

- **Reproducibility**: Same environment across all machines
- **Isolation**: No dependency conflicts with your system
- **Portability**: Run on any machine with Docker installed
- **Scalability**: Easy to run multiple containers in parallel

## Quick Start

### 1. Build the image

```bash
docker build -t tissue-tile-quality-filter:v0.1.0 .
```

### 2. Run with Docker

Process tiles from the `example_tiles/synthetic_tiles` directory:

```bash
docker run --rm \
  -v ./example_tiles/synthetic_tiles:/input:ro \
  -v ./docker_results:/output:rw \
  tissue-tile-quality-filter:v0.1.0 \
  process /input --output /output --verbose
```

### 3. Check the results

Results are saved to `./docker_results/`:
```bash
ls -la docker_results/
```

## Docker Commands Reference

### Show help
```bash
docker run --rm tissue-tile-quality-filter:v0.1.0 --help
```

### Display package info
```bash
docker run --rm tissue-tile-quality-filter:v0.1.0 info
```

### Process with custom options
```bash
docker run --rm \
  -v ./example_tiles/kaggle_tiles:/input:ro \
  -v ./docker_results:/output:rw \
  tissue-tile-quality-filter:v0.1.0 \
  process /input \
    --batch-id my_batch \
    --threshold high \
    --output /output \
    --workers 4 \
    --verbose
```

## Using Docker Compose

For convenience, use `docker-compose.yml`:

### Build and run
```bash
docker-compose up --build
```

### Override the command
```bash
docker-compose run --rm tissue-tile-quality-filter \
  process /data/input/kaggle_tiles --threshold high --output /data/output --verbose
```

## Volume Mounting Explanation

- `-v ./example_tiles/synthetic_tiles:/input:ro` 
  - Host directory: `./example_tiles/synthetic_tiles` (your local tiles)
  - Container path: `/input` (where the container sees them)
  - `:ro` = read-only (prevents accidental modification)

- `-v ./docker_results:/output:rw`
  - Host directory: `./docker_results` (created if it doesn't exist)
  - Container path: `/output` (where results are written)
  - `:rw` = read-write

## Image Details

### Base Image
- `python:3.11-slim` - Official Python 3.11 slim image (~150MB)

### Multi-stage Build
- **Stage 1 (Builder)**: Compiles dependencies in isolation
- **Stage 2 (Runtime)**: Contains only runtime files, minimal size (~800MB)

### Security
- Non-root user (`appuser`, UID 1000)
- Read-only filesystem where possible
- No unnecessary packages

