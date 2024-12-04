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
   nxbench --config 'configs/example.yaml' benchmark run
   ```

3. **Export Results**:

   ```bash
   nxbench benchmark export 'results/results.csv' --output-format csv  # Convert benchmark results into CSV format.
   ```

4. **View Results**:

   ```bash
   nxbench viz serve  # Launch the interactive dashboard.
   ```

## Advanced Command-Line Interface

The CLI provides comprehensive management of benchmarks, datasets, and visualization.

### Validating ASV Configuration

```bash
asv check
```

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
  nxbench --config 'configs/example.yaml' -vvv benchmark run
  ```

- **Export Results to a SQL Database**:

  ```bash
  nxbench benchmark export 'results/benchmarks.sqlite' --output-format sql
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
- **CuGraph** (requires separate CUDA installation and supported GPU hardware)
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
docker-compose run --rm nxbench benchmark run --backend networkx
```

### Viewing Results

```bash
docker-compose run --rm nxbench benchmark export results.csv
```
