[![Python](https://img.shields.io/pypi/pyversions/nxbench.svg)](https://badge.fury.io/py/nxbench)
[![PyPI](https://badge.fury.io/py/nxbench.svg)](https://badge.fury.io/py/nxbench)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NxBench

<p align="center">
  <img src="doc/_static/nxbench_logo.png" alt="NxBench Logo" width="150"/>
</p>

**nxbench** is a comprehensive benchmarking suite designed to facilitate comparative profiling of graph analytic algorithms across NetworkX and compatible backends. Built with an emphasis on extensibility and detailed performance analysis, nxbench aims to enable developers and researchers to optimize their graph analysis workflows efficiently and reproducibly.

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

```bash
git clone https://github.com/dpys/nxbench.git
cd nxbench
pip install -e .[cuda]  # CUDA support is needed for CuGraph benchmarking
```

For benchmarking using CUDA-based tools like [CuGraph](https://github.com/rapidsai/cugraph):

```bash
pip install -e .[cuda]
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
nxbench --config 'configs/example.yaml' benchmark run
```

3. Export results:

```bash
nxbench benchmark export 'results/results.csv' --output-format csv  # Convert benchmarked results into csv format.
```


4. View results:

```bash
nxbench viz serve  # Launch interactive dashboard
```

## Advanced Command Line Interface

The CLI provides comprehensive management of benchmarks, datasets, and visualization:

```bash
# Validating asv configuration
asv check

# Data Management
nxbench data download karate  # Download specific dataset
nxbench data list --category social  # List available datasets

# Benchmarking
nxbench --config 'configs/example.yaml' -vvv benchmark run  # Debug benchmark runs
nxbench benchmark export 'results/benchmarks.sqlite' --output-format sql # Export the results into a sql database
nxbench benchmark compare HEAD HEAD~1  # Compare with previous commit

# Visualization
nxbench viz serve  # Launch parallel categories dashboard
nxbench viz publish  # Generate static asv report
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
- CuGraph (requires separate CUDA installation and supported GPU hardware)
- GraphBLAS Algorithms (optional)
- nx-parallel (optional)

## Development

```bash
# Install development dependencies
pip install -e .[test,doc] # testing and documentation

# Run tests
make test
```

## Reproducible benchmarking through containerization

```bash
# Run benchmarks with GPU
docker-compose up nxbench

# Run benchmarks CPU-only
NUM_GPU=0 docker-compose up nxbench

# Start visualization dashboard
docker-compose up dashboard

# Run specific backend
docker-compose run --rm nxbench benchmark run --backend networkx

# View results
docker-compose run --rm nxbench benchmark export results.csv
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
