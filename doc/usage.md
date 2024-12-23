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

2. **Start an instance of an orion server in a separate terminal window:**

  ```bash
  export PREFECT_API_URL="http://127.0.0.1:4200/api"
  export PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://prefect_user:pass@localhost:5432/prefect_db"
  prefect server start
  ```

3. **Run Benchmarks Based on the Configuration**:

   ```bash
   nxbench --config 'nxbench/configs/example.yaml' benchmark run
   ```

4. **Export Results**:

   ```bash
   nxbench --config 'nxbench/configs/example.yaml' benchmark export 'results/9e3e8baa4a3443c392dc8fee00373b11_20241220002902.json' --output-format csv --output-file 'results/results.csv'  # convert benchmarked results from a run with hash `9e3e8baa4a3443c392dc8fee00373b11_20241220002902` into csv format.
   ```

5. **View Results**:

   ```bash
   nxbench viz serve  # launch the interactive results visualization dashboard.
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
  nxbench --config 'nxbench/configs/example.yaml' benchmark export 'results/9e3e8baa4a3443c392dc8fee00373b11_20241220002902.json' --output-format sql --output-file 'results/benchmarks.sqlite'
  ```

### Visualization

- **Launch the Dashboard**:

  ```bash
  nxbench viz serve
  ```

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
docker-compose -f docker/docker-compose.cpu.yaml run --rm nxbench --config 'nxbench/configs/example.yaml' benchmark run --backend networkx
```

### Exporting results from a run with hash `9e3e8baa4a3443c392dc8fee00373b11_20241220002902`

```bash
docker-compose -f docker/docker-compose.cpu.yaml run --rm nxbench --config 'nxbench/configs/example.yaml' benchmark export 'nxbench_results/9e3e8baa4a3443c392dc8fee00373b11_20241220002902.json' --output-format csv --output-file 'nxbench_results/results.csv'
```
