Welcome to NxBench's Documentation
==================================

Overview
========
**nxbench** is a comprehensive benchmarking suite designed to facilitate comparative profiling of graph analytic algorithms across NetworkX and compatible backends. Built with an emphasis on extensibility and detailed performance analysis, nxbench aims to enable developers and researchers to optimize their graph analysis workflows efficiently and reproducibly.

Key Features
============
- **Cross-Backend Benchmarking**: Leverage NetworkX's backend system to profile algorithms across multiple implementations (NetworkX, nx-parallel, GraphBLAS, and CuGraph).
- **Configurable Suite**: YAML-based configuration for algorithms, datasets, and benchmarking parameters.
- **Real-World Datasets**: Automated downloading and caching of networks and their metadata from NetworkRepository.
- **Synthetic Graph Generation**: Support for generating benchmark graphs using any of NetworkX's built-in generators.
- **Validation Framework**: Comprehensive result validation for correctness across implementations.
- **Performance Monitoring**: Track execution time and memory usage with detailed metrics.
- **Interactive Visualization**: Dynamic dashboard for exploring benchmark results using Plotly Dash.
- **Flexible Storage**: SQLite-based result storage with pandas integration for analysis.
- **CI Integration**: Support for automated benchmarking through ASV (Airspeed Velocity).

.. toctree::
   :maxdepth: 2

   installation
   usage
   examples
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
