# nxbench

**nxbench** is a comprehensive benchmarking suite designed to facilitate comparative profiling of graph analytic algorithms across NetworkX and compatible backends. Built with an emphasis on reproducibility, extensibility, and detailed performance analysis, nxbench enables developers and researchers to optimize their graph analysis workflows efficiently.

## Features

- **Cross-Backend Benchmarking**: Leverage NetworkX's backend system to profile algorithms across multiple implementations (NetworkX, nx-parallel, GraphBLAS, and CuGraph)
- **Configurable Suite**: YAML-based configuration for algorithms, datasets, and benchmarking parameters
- **Real-World Datasets**: Automated downloading and caching of networks from NetworkRepository
- **Synthetic Graph Generation**: Support for generating benchmark graphs using NetworkX's built-in generators
- **Validation Framework**: Comprehensive result validation for correctness across implementations
- **Performance Monitoring**: Track execution time and memory usage with detailed metrics
- **Interactive Visualization**: Dynamic dashboard for exploring benchmark results using Plotly Dash
- **Flexible Storage**: SQLite-based result storage with pandas integration for analysis
- **CI Integration**: Support for automated benchmarking through ASV (Airspeed Velocity)

## Installation

```bash
git clone https://github.com/dpys/nxbench.git
cd nxbench
pip install -e .[cuda]  # Optional extras for CUDA support
```

For benchmarking using CUDA-based tools like [CuGraph](https://github.com/rapidsai/cugraph):

```bash
pip install -e .[cuda]
```

## Quick Start

1. Configure your benchmarks in `configs/example.yaml`:

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

2. Run benchmarks:

```bash
nxbench benchmark run --backend networkx
```

3. View results:

```bash
nxbench viz serve  # Launch interactive dashboard
```

## Command Line Interface

The CLI provides comprehensive management of benchmarks, datasets, and visualization:

```bash
# Data Management
nxbench data download karate  # Download specific dataset
nxbench data list --category social  # List available datasets

# Benchmarking
nxbench benchmark run --backend all  # Run all benchmarks
nxbench benchmark export results.csv  # Export results
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
- CuGraph (optional, requires CUDA)
- GraphBLAS (optional)
- nx-parallel (optional)

## Development

```bash
# Install development dependencies
pip install -e .[test,scrape,doc] # testing, scraping of real-world graph data, and documentation

# Run tests
pytest

# Build documentation
cd docs && make html
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

- NetworkX community for the core graph library
- NetworkRepository.com for dataset access
- ASV team for benchmark infrastructure

## Contact

For questions or suggestions:

- Open an issue on GitHub
- Email: <dpysalexander@gmail.com>
