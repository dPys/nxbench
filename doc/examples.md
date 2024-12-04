# Examples

## Configuration Examples

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

### Example 1: Basic Configuration

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

### Example 2: Advanced Algorithm Configuration

```yaml
algorithms:
  - name: "betweenness_centrality"
    func: "networkx.betweenness_centrality"
    params:
      normalized: true
    requires_directed: false
    groups: ["centrality"]
    validate_result: "nxbench.validation.validate_betweenness_centrality"

datasets:
  - name: "road_network"
    source: "networkrepository"
    params:
      format: "edgelist"
```

## Usage Examples

### Running a Specific Benchmark Configuration

```bash
nxbench --config 'configs/advanced.yaml' benchmark run
```

### Exporting Benchmark Results to CSV

```bash
nxbench benchmark export 'results/advanced_results.csv' --output-format csv
```

### Comparing Benchmark Results Between Different Commits

```bash
nxbench benchmark compare v1.0.0 v0.9.0
```

### Visualizing Results in a Web Browser

```bash
nxbench viz serve --port 8050
```

### Generating a Static Report

```bash
nxbench viz publish --output-dir 'reports/static_report'
```

## Containerization Examples

### Building the Docker Image

```bash
docker build -t nxbench:latest .
```

### Running Benchmarks Inside a Docker Container

```bash
docker run --rm nxbench:latest benchmark run --backend cugraph
```

### Accessing the Visualization Dashboard via Docker Compose

```bash
docker-compose up dashboard
```
