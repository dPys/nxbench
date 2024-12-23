
# **Nxbench Configuration Guide**

## **Introduction**

The included `example.yaml` file serves as the boilerplate configuration for `nxbench`, defining the algorithms to benchmark, the datasets to use, validation settings, benchmarking matrices, and environment configurations.

Properly configuring the configuration yaml file ensures accurate and meaningful benchmarking results tailored to your specific requirements.

## **Configuration Structure**

The YAML configuration file is divided into several key sections:

1. **Algorithms**
2. **Datasets**
3. **Validation**
4. **Environment**

Each section plays a crucial role in setting up the benchmarking environment. Below, we delve into each section, explaining their fields and providing examples for clarity.

---

## **1. Algorithms**

### **Purpose**

Defines the list of algorithms that `nxbench` will benchmark. Each algorithm entry specifies the function to benchmark, its parameters, grouping categories, and optional validation functions.

### **Fields**

- **`name`** *(string, required)*: A unique identifier for the algorithm.
- **`func`** *(string, required)*: The fully qualified Python path to the function implementing the algorithm.
- **`params`** *(dictionary, optional)*: A set of parameters to pass to the algorithm function.
- **`requires_directed`** *(boolean, optional)*: Indicates if the algorithm requires a directed graph. Defaults to `false`.
- **`groups`** *(list of strings, optional)*: Categories or tags to group algorithms for selective benchmarking.
- **`validate_result`** *(string, optional)*: Fully qualified path to a validation function to verify the correctness of the algorithm's output.
- **`min_rounds`** *(integer, optional)*: Minimum number of benchmarking rounds to ensure statistical significance.
- **`warmup`** *(boolean, optional)*: If `true`, runs warm-up iterations to stabilize performance measurements.
- **`warmup_iterations`** *(integer, optional)*: Number of warm-up iterations to execute.

### **Example Entries**

#### **Basic Configuration**

```yaml
algorithms:
  - name: "pagerank"
    func: "networkx.pagerank"
    params:
      alpha: 0.85
    groups: ["centrality"]
```

#### **Advanced Configuration with Validation and Warm-up**

```yaml
algorithms:
  - name: "betweenness_centrality"
    func: "networkx.betweenness_centrality"
    params:
      normalized: true
      endpoints: false
    requires_directed: false
    groups: ["centrality", "path_based"]
    validate_result: "nxbench.validation.validate_betweenness_centrality"
    min_rounds: 5
    warmup: true
    warmup_iterations: 20
```

---

## **2. Datasets**

### **Purpose**

Specifies the datasets on which the algorithms will be benchmarked. Datasets can be sourced from repositories or generated on-the-fly using graph generators.

### **Fields**

- **`name`** *(string, required)*: A unique identifier for the dataset.
- **`source`** *(string, required)*: Specifies the source of the dataset. Common sources include:
  - `"networkrepository"`: Fetches datasets from the Network Repository.
  - `"generator"`: Generates datasets using NetworkX graph generators.
- **`params`** *(dictionary, optional)*: Parameters required to fetch or generate the dataset.
- **`metadata`** *(dictionary, optional)*: Additional information about the dataset, such as whether it's directed or weighted.

### **Example Entries**

#### **Basic Configuration**

```yaml
datasets:
  - name: "karate"
    source: "networkrepository"
    params: {}
```

#### **Dataset Generated on-the-fly**

```yaml
datasets:
  - name: "erdos_renyi_small"
    source: "generator"
    params:
      generator: "networkx.erdos_renyi_graph"
      n: 1000
      p: 0.01
    metadata:
      directed: false
      weighted: false
```

---

## **3. Validation**

### **Purpose**

Configures how benchmark results are validated to ensure the correctness and reliability of the algorithms' outputs.

### **Fields**

- **`skip_slow`** *(boolean, optional)*: If `true`, skips validation for algorithms or datasets that are time-consuming.
- **`validate_all`** *(boolean, optional)*: If `true`, validates all benchmark results.
- **`error_on_fail`** *(boolean, optional)*: If `true`, raises an error when validation fails.
- **`report_memory`** *(boolean, optional)*: If `true`, includes memory usage in the validation reports.

### **Example Entry**

```yaml
validation:
  skip_slow: false
  validate_all: true
  error_on_fail: true
  report_memory: true
```

---

## **4. Environment**

### **Purpose**

Configures environment settings, such as the python and dependency versions.

### **Fields**

- **`python`** *(list of strings, required)*: List of valid python versions (e.g. "3.10", "3.11")
- **`req`** *(list of strings, required)*: Lists the Python dependencies required for benchmarking.

### **Example Entry**

```yaml
environ:
  backend:
    - "networkx"
    - "parallel"
    - "graphblas"
  num_threads:
    - "1"
    - "4"
    - "8"
  req:
    - "networkx==3.4.2"
    - "nx_parallel==0.3"
    - "graphblas_algorithms==2023.10.0"
  pythons:
    - "3.10"
    - "3.11"
```
