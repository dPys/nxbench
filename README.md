# nxbench

**nxbench** is a comprehensive benchmarking suite designed to facilitate the comparative profiling of graph analytic algorithms across various backends supported by NetworkX. With an emphasis on performance, reproducibility, and scalability, nxbench enables developers and researchers to optimize their graph analysis workflows efficiently.

## Key Features

- **Cross-Library Benchmarking**: Leverage NetworkX's backend system to profile and compare multiple implementations of graph algorithms.
- **Support for Diverse Graph Types and Datasets**: Benchmark algorithms on a wide range of graph structures.
- **Hardware-Specific Benchmarks**: Capture performance metrics for both CPU and GPU configurations (where applicable).
- **Flexible Data Management**: Store and query benchmark results using SQL and in-memory dataframes.
- **Visualization Dashboard**: Interactive visualizations using Plotly Dash for exploring benchmark results.
- **Docker Integration**: Reproducible benchmarking environment with Docker and Docker Compose support.
- **CLI**: Command-line interface for easy management of benchmarks and querying of results.

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/dpys/nxbench.git
cd nxbench
pip install -r requirements.txt
```

To use Docker:

```bash
docker-compose up --build
```

## Quick Start

1. **Run Benchmarks**:

   ```bash
   python -m nxbench.cli --run --algorithms pagerank --backends networkx graph_tool --graph-types directed --nodes 1000 10000
   ```

2. **View Results**:

   ```bash
   python -m nxbench.cli --query --algorithm pagerank --backend networkx
   ```

3. **Launch the Dashboard**:

   ```bash
   python -m nxbench.viz.app
   ```

## Components

### 1. Benchmarking Core

A modular suite to benchmark any registered NX algorithm across various backends, enabling easy extensibility for future algorithms and libraries.

### 2. Database Backend

Efficiently manage benchmark results with a database system:

- Store, query, and analyze results with SQL or Pandas.
- Flexible export options to CSV, JSON, and more.

### 3. Visualization Dashboard

Explore and analyze benchmark results with an interactive dashboard:

- Dynamic plots: heatmaps, box plots, and performance comparisons.
- User-friendly interface to filter and visualize data.

### 4. Docker Integration

Reproducible and consistent benchmarking using containerized environments, with support for both CPU and GPU setups.

### 5. Command-Line Interface (CLI)

Manage benchmarks easily:

- Run benchmarks with custom configurations.
- Query results and export data for further analysis.

## Configuration

Customize benchmarking parameters with a `config.yaml` file:
```
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request to discuss changes or new features.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or suggestions, please reach out at [dpysalexander@gmail.com].
