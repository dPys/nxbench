# Installation

PyPi:

```bash
pip install nxbench
```

From a local clone:

```bash
make install
```

Docker:

```bash
# CPU-only
docker-compose -f docker/docker-compose.cpu.yaml build

# With GPU
docker-compose -f docker/docker-compose.gpu.yaml build
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
