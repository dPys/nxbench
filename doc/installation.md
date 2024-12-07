# Installation

PyPi:

```bash
pip install nxbench
```

For benchmarking using CUDA-based tools like [CuGraph](https://github.com/rapidsai/cugraph):

```bash
pip install nxbench[cuda]
```

## Development Setup

Install development dependencies (testing and documentation):

```bash
pip install -e .[test,doc]
```

Run tests to ensure everything is set up correctly:

```bash
make test
```
