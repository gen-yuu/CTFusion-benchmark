# CTFusion Benchmark Suite

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a benchmark suite for evaluating GPU and Host performance for deep learning tasks.
It is designed to be a reproducible and extensible tool for research purposes.

## Quick Start

### 1. Clone & prepare environment

```bash
git clone https://github.com/gen-yuu/ctfusion-benchmark.git
cd ctfusion-benchmark

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -e .
```

### 3. Run all benchmarks

```bash
python run_benchmark.py
```

Benchmark results are stored under `results/<timestamp>_<system_id>/`.

## Directory Structure

```bash
CTFusion_benchmark/
├── Dockerfile                  # Docker recipe for reproducible runs
├── configs/
│   └── benchmark_config.yaml   # YAML for benchmark parameters
├── src/
│   ├── __init__.py
│   ├── analysis/
│   │   └── curve.py            # Feature extraction from performance curves
│   ├── benchmarks/
│   │   ├── data_transfer/
│   │   │   └── core.py         # Manages H↔D transfer benchmarks
│   │   ├── gpu_compute/
│   │   │   ├── core.py         # Manages GPU micro-benchmarks
│   │   │   └── layers.py       # Generates PyTorch modules to benchmark
│   │   └── host_compute/
│   │       ├── core.py         # Manages CPU micro-benchmarks
│   │       ├── domains/
│   │       │   └── image_processing.py
│   │       └── factory.py
│   ├── config.py               # YAML loader
│   ├── logger.py               # Custom logger
│   └── utils.py                # Helper utilities (system ID, etc.)
├── run_benchmark.py            # Entry point script
└── pyproject.toml
```
