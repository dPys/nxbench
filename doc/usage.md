# Usage

## Quick Start

1. **Configure Your Benchmarks**: Create a YAML configuration file (e.g., `configs/example.yaml`):

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

2. **Run Benchmarks Based on the Configuration**:

   ```bash
   nxbench --config 'nxbench/configs/example.yaml' benchmark run
   ```

3. **Export Results**:

   ```bash
   nxbench --config 'nxbench/configs/example.yaml' benchmark export 'results/results.csv' --output-format csv  # convert benchmark results into CSV format.
   ```

4. **View Results**:

   ```bash
   nxbench viz serve  # launch the interactive dashboard.
   ```

## Advanced Command-Line Interface

The CLI provides comprehensive management of benchmarks, datasets, and visualization.

### Data Management

- **Download a Specific Dataset**:

  ```bash
  nxbench data download karate
  ```

- **List Available Datasets by Category**:

  ```bash
  nxbench data list --category social
  ```

### Benchmarking

- **Run Benchmarks with Verbose Output**:

  ```bash
  nxbench --config 'nxbench/configs/example.yaml' -vvv benchmark run
  ```

- **Export Results to a SQL Database**:

  ```bash
  nxbench --config 'nxbench/configs/example.yaml' benchmark export 'results/benchmarks.sqlite' --output-format sql
  ```

- **Compare Benchmarks Between Commits**:

  ```bash
  nxbench benchmark compare HEAD HEAD~1
  ```

### Visualization

- **Launch the Dashboard**:

  ```bash
  nxbench viz serve
  ```

- **Generate a Static ASV Report**:

  ```bash
  nxbench viz publish
  ```

## Supported Backends

- **NetworkX** (default)
- **Nx-CuGraph** (requires separate CuGraph installation and supported GPU hardware)
- **GraphBLAS Algorithms** (optional)
- **nx-parallel** (optional)

## Reproducible Benchmarking Through Containerization

### Running Benchmarks with GPU Support

```bash
docker-compose up nxbench
```

### Running Benchmarks on CPU Only

```bash
NUM_GPU=0 docker-compose up nxbench
```

### Starting the Visualization Dashboard

```bash
docker-compose up dashboard
```

### Running Benchmarks with a Specific Backend

```bash
docker-compose run --rm nxbench --config 'nxbench/configs/example.yaml' benchmark run --backend networkx
```

### Viewing Results

```bash
docker-compose run --rm nxbench --config 'nxbench/configs/example.yaml' benchmark export results.csv
```
