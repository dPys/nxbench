[![Python](https://img.shields.io/pypi/pyversions/nxbench.svg)](https://badge.fury.io/py/nxbench)
[![PyPI](https://badge.fury.io/py/nxbench.svg)](https://badge.fury.io/py/nxbench)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/dPys/nxbench/graph/badge.svg?token=1M8NM7MQLI)](https://codecov.io/gh/dPys/nxbench)

# NxBench

<p align="center">
  <img src="doc/_static/assets/nxbench_logo.png" alt="NxBench Logo" width="125"/>
</p>

**nxbench** is a comprehensive benchmarking suite designed to facilitate comparative profiling of graph analytic algorithms across NetworkX and compatible backends. Built using Prefect and Dask, nxbench places an emphasis on extensible and granular performance analysis, enabling developers and researchers to optimize their graph analysis workflows efficiently and reproducibly.

## Key Features

- **Cross-Backend Benchmarking**: Leverage NetworkX's backend system to profile algorithms across multiple implementations (NetworkX, nx-parallel, GraphBLAS, and CuGraph)
- **Configurable Suite**: YAML-based configuration for algorithms, datasets, and benchmarking parameters
- **Real-World Datasets**: Automated downloading and caching of networks and their metadata from NetworkRepository
- **Synthetic Graph Generation**: Support for generating benchmark graphs using any of NetworkX's built-in generators
- **Validation Framework**: Comprehensive result validation for correctness across implementations
- **Performance Monitoring**: Track execution time and memory usage with detailed metrics
- **Interactive Visualization**: Dynamic dashboard for exploring benchmark results using Plotly Dash
- **Flexible Storage**: SQL result storage with pandas integration for dowstream analysis

## Installation (Non-Docker Setup)

### Prerequisites

- **Python 3.10+**: Ensure you have a compatible Python environment.
- **PostgreSQL**: To run Prefect Orion with a persistent database, we recommend PostgreSQL for better concurrency and stability than an ephemeral in-memory database.
- **NetworkX Backend Installations**: To comparative benchmark graph algorithm performance across different [NetworkX backends](https://networkx.org/documentation/stable/reference/backends.html), you will need to install each backend.

#### Setting up PostgreSQL

In a terminal window:

1. **Install PostgreSQL**:

   - On macOS (with Homebrew):

     ```bash
     brew install postgresql
     brew services start postgresql
     ```

   - On Linux (Debian/Ubuntu):

     ```bash
     sudo apt-get update && sudo apt-get install -y postgresql postgresql-contrib
     sudo service postgresql start
     ```

   - On Windows:
     Download and run the [PostgreSQL installer](https://www.postgresql.org/download/windows/) and follow the prompts.

2. **Create a PostgreSQL User and Database**:

   ```bash
   psql postgres
   ```

   In the `psql` prompt, run:

   ```sql
   CREATE USER prefect_user WITH PASSWORD 'pass';

   CREATE DATABASE prefect_db OWNER prefect_user;
   GRANT ALL PRIVILEGES ON DATABASE prefect_db TO prefect_user;
   ```

   Exit the prompt with \q.

   This sets up a prefect_user with password pass and a database named prefect_db.

#### Supported Backends

- NetworkX (default)
- nx-CuGraph (requires separate CuGraph installation and supported GPU hardware)
- GraphBLAS Algorithms (optional)
- nx-parallel (optional)

### Installing `nxbench`

In a new terminal window:

PyPi:

```bash
pip install nxbench
```

From source (local clone):

```bash
git clone https://github.com/dpys/nxbench.git
cd nxbench
make install
```

This should install nxbench and all required dependencies (including prefect, asyncpg, and related packages).

## Installation (Docker Setup)

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

2. Start an instance of an orion server in a separate terminal window:

```bash
export PREFECT_API_URL="http://127.0.0.1:4200/api"
export PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://prefect_user:pass@localhost:5432/prefect_db"
prefect server start
```

3. In the original terminal window, run benchmarks based on the configuration:

```bash
nxbench --config 'nxbench/configs/example.yaml' benchmark run
```

4. Export results:

```bash
nxbench --config 'nxbench/configs/example.yaml' benchmark export 'results/9e3e8baa4a3443c392dc8fee00373b11_20241220002902.json' --output-format csv --output-file 'results/results.csv'  # convert benchmarked results from a run with hash `9e3e8baa4a3443c392dc8fee00373b11_20241220002902` into csv format.
```

5. View results:

```bash
nxbench viz serve  # launch the interactive results visualization dashboard.
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
nxbench --config 'nxbench/configs/example.yaml' benchmark export 'results/9e3e8baa4a3443c392dc8fee00373b11_20241220002902.json' --output-format sql --output-file 'results/benchmarks.sqlite' # export the results from a run with hash `9e3e8baa4a3443c392dc8fee00373b11_20241220002902` into a sql database
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

## Reproducible benchmarking through containerization

```bash
# Run benchmarks with GPU
NUM_GPU=1 docker-compose -f docker/docker-compose.gpu.yaml up nxbench

# Run benchmarks CPU-only
docker-compose -f docker/docker-compose.cpu.yaml up nxbench

# Start visualization dashboard
docker-compose -f docker/docker-compose.cpu.yaml up dashboard

# Run specific backend
docker-compose -f docker/docker-compose.cpu.yaml run --rm nxbench --config 'nxbench/configs/example.yaml' benchmark run --backend networkx

# Export results from a run with hash `9e3e8baa4a3443c392dc8fee00373b11_20241220002902`
docker-compose -f docker/docker-compose.cpu.yaml run --rm nxbench --config 'nxbench/configs/example.yaml' benchmark export 'nxbench_results/9e3e8baa4a3443c392dc8fee00373b11_20241220002902.json' --output-format csv --output-file 'nxbench_results/results.csv'
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

## Contact

For questions or suggestions:

- Open an issue on GitHub
- Email: <dpysalexander@gmail.com>
