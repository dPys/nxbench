# Installation

Clone the repository and install the package:

```bash
git clone https://github.com/dpys/nxbench.git
cd nxbench
pip install -e .
```

For benchmarking using CUDA-based tools like [CuGraph](https://github.com/rapidsai/cugraph):

```bash
pip install -e .[cuda]  # CUDA support is needed for CuGraph benchmarking
```

## Development Setup

Install development dependencies:

```bash
pip install -e .[test,doc]  # For testing and documentation
```

Run tests to ensure everything is set up correctly:

```bash
make test
```
