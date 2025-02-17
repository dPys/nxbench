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

- **Export Results to a Postgres SQL Database**:

  ```bash
  nxbench --config 'nxbench/configs/example.yaml' benchmark export 'results/9e3e8baa4a3443c392dc8fee00373b11_20241220002902.json' --output-format sql
  ```

### Visualization

- **Launch the Dashboard**:

  ```bash
  nxbench viz serve
  ```

## Reproducible Benchmarking Through Containerization

Use the provided convenience script (docker/nxbench-run.sh) instead of invoking docker-compose directly. This script automatically resolves your configuration file, switches between CPU and GPU modes, detects the desired nxbench subcommand, and mounts the host's results directory when needed.

```bash
# Download a Dataset (e.g. Karate):
docker/nxbench-run.sh --config 'nxbench/configs/example.yaml' data download karate

# List Available Datasets by Category:
docker/nxbench-run.sh --config 'nxbench/configs/example.yaml' data list --category social

# Run benchmarks
docker/nxbench-run.sh --config 'nxbench/configs/example.yaml' benchmark run

# Run benchmarks (with GPU support)
docker/nxbench-run.sh --gpu --config 'nxbench/configs/example.yaml' benchmark run

# Export Benchmark Results to CSV:
docker/nxbench-run.sh --config 'nxbench/configs/example.yaml' benchmark export 'nxbench_results/9e3e8baa4a3443c392dc8fee00373b11_20241220002902.json' --output-format csv --output-file 'nxbench_results/results.csv'

# Launch the Visualization Dashboard:
docker/nxbench-run.sh --config 'nxbench/configs/example.yaml' viz serve
# Note: The dashboard service requires that benchmark results have been generated and exported (i.e. a valid results/results.csv file
# exists).
```

## Adding a New Backend

> **Note:** The following guide assumes you have a recent version of NxBench with the new `BackendManager` and associated tools (e.g., [`core.py`](../nxbench/backends/core.py) and [`registry.py`](../nxbench/backends/registry.py)) already in place. It also assumes that your backend follows the [guidelines for developing custom NetworkX backends](https://networkx.org/documentation/stable/reference/backends.html#docs-for-backend-developers)

### 1. Verify Your Backend is Installable

1. **Install** your backend via `pip` (or conda, etc.).
   For example, if your backend library is `my_cool_backend`, ensure that:

   ```bash
   pip install my_cool_backend
   ```

2. **Check import**: NxBench’s detection system simply looks for `importlib.util.find_spec("my_cool_backend")`. So if your library is not found by Python, NxBench will conclude it is unavailable.

### 2. Write a Conversion Function

In NxBench, a “backend” is simply a library or extension that **converts a `networkx.Graph` into an alternate representation**. You must define one or more **conversion** functions:

```python
def convert_my_cool_backend(nx_graph: networkx.Graph, num_threads: int):
    import my_cool_backend
    # Possibly configure multi-threading if relevant:
    # my_cool_backend.configure_threads(num_threads)

    # Convert the Nx graph to your library’s internal representation:
    return my_cool_backend.from_networkx(nx_graph)
```

## 3. (Optional) Write a Teardown Function

If your backend has special cleanup needs (e.g., free GPU memory, close connections, revert global state, etc.), define a teardown function:

```python
def teardown_my_cool_backend():
    import my_cool_backend
    # e.g. my_cool_backend.shutdown()
    pass
```

If your backend doesn’t need cleanup, skip this or simply define an empty function.

## 4. Register with NxBench

Locate NxBench’s [registry.py](../nxbench/backends/registry.py) (or a similar file where other backends are registered). Add your calls to `backend_manager.register_backend(...)`:

```python
from nxbench.backends.registry import backend_manager
import networkx as nx  # only if needed

def convert_my_cool_backend(nx_graph: nx.Graph, num_threads: int):
    import my_cool_backend
    # Possibly configure my_cool_backend with num_threads
    return my_cool_backend.from_networkx(nx_graph)

def teardown_my_cool_backend():
    # e.g. release resources
    pass

backend_manager.register_backend(
    name="my_cool_backend",         # The name NxBench will use to refer to it
    import_name="my_cool_backend",  # The importable Python module name
    conversion_func=convert_my_cool_backend,
    teardown_func=teardown_my_cool_backend  # optional
)
```

**Important**:

- `name` is the “human-readable” alias in NxBench.
- `import_name` is the actual module import path. They can be the same (most common) or different if your library’s PyPI name differs from its Python import path.

## 5. Confirm It Works

1. **Check NxBench logs**: When NxBench runs, it will detect whether `"my_cool_backend"` is installed by calling `importlib.util.find_spec("my_cool_backend")`.
2. **Run a quick benchmark**:

   ```bash
   nxbench --config my_config.yaml benchmark run
   ```

   If you see logs like “Chosen backends: [‘my_cool_backend’ …]” then NxBench recognized your backend. If it fails with “No valid backends found,” ensure your library is installed and spelled correctly.

## 6. (Optional) Version Pinning

If you want NxBench to only run your backend if it matches a pinned version (e.g. `my_cool_backend==2.1.0`), add something like this to your NxBench config YAML:

```yaml
environ:
  backend:
    my_cool_backend:
      - "my_cool_backend==2.1.0"
```

NxBench will:

- Detect the installed version automatically (via `my_cool_backend.**version**` or PyPI metadata)
- Skip running if it doesn’t match `2.1.0`.

---

### That’s it

You’ve successfully added a new backend to NxBench! Now, NxBench can detect it, convert graphs for it, optionally tear it down, and track its version during benchmarking.
