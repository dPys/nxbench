[![Python](https://img.shields.io/pypi/pyversions/nxbench.svg)](https://badge.fury.io/py/nxbench)
[![PyPI](https://badge.fury.io/py/nxbench.svg)](https://badge.fury.io/py/nxbench)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NxBench

<p align="center">
  <img src="doc/_static/assets/nxbench_logo.png" alt="NxBench Logo" width="150"/>
</p>

**nxbench** is a comprehensive benchmarking suite designed to facilitate comparative profiling of graph analytic algorithms across NetworkX and compatible backends. Built on top of [Airspeed Velocity (ASV)](https://github.com/airspeed-velocity/asv), nxbench places an emphasis on extensible and granular performance analysis, enabling developers and researchers to optimize their graph analysis workflows efficiently and reproducibly.

## Key Features

- **Cross-Backend Benchmarking**: Leverage NetworkX's backend system to profile algorithms across multiple implementations (NetworkX, nx-parallel, GraphBLAS, and CuGraph)
- **Configurable Suite**: YAML-based configuration for algorithms, datasets, and benchmarking parameters
- **Real-World Datasets**: Automated downloading and caching of networks and their metadata from NetworkRepository
- **Synthetic Graph Generation**: Support for generating benchmark graphs using any of NetworkX's built-in generators
- **Validation Framework**: Comprehensive result validation for correctness across implementations
- **Performance Monitoring**: Track execution time and memory usage with detailed metrics
- **Interactive Visualization**: Dynamic dashboard for exploring benchmark results using Plotly Dash
- **Flexible Storage**: SQLite-based result storage with pandas integration for analysis
- **CI Integration**: Support for automated benchmarking through ASV (Airspeed Velocity)

## Installation

PyPi:

```bash
pip install nxbench
```

From a local clone:

```bash
make install
```

Docker:

```bash
# CPU-only
docker-compose -f docker/docker-compose.cpu.yaml build

# With GPU
docker-compose -f docker/docker-compose.gpu.yaml build
```

## Quick Start

1. Configure your benchmarks in a yaml file (see `configs/example.yaml`):

```yaml
algorithms:
  - name: "pagerank"
    func: "networkx.pagerank"
    params:
      alpha: 0.85
    groups: ["centrality"]

datasets:
  - name: "karate"
    source: "networkrepository"
```

2. Run benchmarks based on the configuration:

```bash
nxbench --config 'nxbench/configs/example.yaml' benchmark run
```

3. Export results:

```bash
nxbench benchmark export 'results/results.csv' --output-format csv  # convert benchmarked results into csv format.
```

4. View results:

```bash
nxbench viz serve  # visualize results using parallel categories dashboard
```

<p align="center">
  <img src="doc/_static/assets/animation.gif" alt="Parallel Categories Animation" width="1000"/>
</p>


## Advanced Command Line Interface

The CLI provides comprehensive management of benchmarks, datasets, and visualization:

```bash
# Data Management
nxbench data download karate  # download specific dataset
nxbench data list --category social  # list available datasets

# Benchmarking
nxbench --config 'nxbench/configs/example.yaml' -vvv benchmark run  # debug benchmark runs
nxbench benchmark export 'results/benchmarks.sqlite' --output-format sql # export the results into a sql database

# Visualization
nxbench viz serve  # visualize results using parallel categories dashboard
nxbench viz publish  # generate static asv report
```

## Configuration

Benchmarks are configured through YAML files with the following structure:

```yaml
algorithms:
  - name: "algorithm_name"
    func: "fully.qualified.function.name"
    params: {}
    requires_directed: false
    groups: ["category"]
    validate_result: "validation.function"

datasets:
  - name: "dataset_name"
    source: "networkrepository"
    params: {}
```

## Supported Backends

- NetworkX (default)
- nx-CuGraph (requires separate CuGraph installation and supported GPU hardware)
- GraphBLAS Algorithms (optional)
- nx-parallel (optional)

## Reproducible benchmarking through containerization

```bash
# Run benchmarks with GPU
NUM_GPU=1 docker-compose -f docker/docker-compose.gpu.yaml up nxbench

# Run benchmarks CPU-only
docker-compose -f docker/docker-compose.cpu.yaml up nxbench

# Start visualization dashboard
docker-compose -f docker/docker-compose.cpu.yaml up dashboard

# Run specific backend
docker-compose -f docker/docker-compose.cpu.yaml run --rm nxbench benchmark run --backend networkx

# View results
docker-compose -f docker/docker-compose.cpu.yaml run --rm nxbench benchmark export results.csv
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code style guidelines
- Development setup
- Testing requirements
- Pull request process

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- NetworkX community for the core graph library and dispatching support
- NetworkRepository.com for harmonized dataset access
- ASV team for benchmark infrastructure

## Contact

For questions or suggestions:

- Open an issue on GitHub
- Email: <dpysalexander@gmail.com>
