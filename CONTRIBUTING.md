
# Contributing to nxbench

Thank you for your interest in contributing to nxbench! We welcome contributions from the community to help improve and expand this project.

## Table of Contents

- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Coding Guidelines](#coding-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Setting Up the Development Environment

1. Fork the nxbench repository on GitHub.
2. Clone your fork locally:

```bash
git clone git@github.com:your-username/nxbench.git
cd nxbench
```

3. Create a virtual environment:

```bash
python -m venv nxbench-dev
source nxbench-dev/bin/activate  # On Windows, use `nxbench-dev\Scripts\activate`
```

4. Install the development dependencies:

```bash
pip install -e ".[developer]"
```

5. Install pre-commit hooks:

```bash
pre-commit install
```

## Coding Guidelines

- Follow [PEP 8](https://pep8.org/) style guide for Python code.
- Use absolute imports (no relative imports).
- Write clear, concise, and well-documented code.
- Follow the project's existing code structure and naming conventions.
- Handle exceptions and errors gracefully.
- Ensure there are no redundant lines of code.

## Testing

- Write unit tests for all new functionality.
- Ensure all tests pass before submitting a pull request.
- To run tests:

```bash
pytest
```

- Aim for high test coverage (at least 90% for new code).

## Documentation

- Use [numpydoc](https://numpydoc.readthedocs.io/en/latest/) style for docstrings.
- Update relevant documentation for any changes or new features.
- Include examples in docstrings where appropriate.

## Pull Request Process

1. Create a new branch for your feature or bugfix:

```bash
git checkout -b feature-or-fix-name
```

2. Make your changes and commit them with a clear commit message.
3. Run pre-commit hooks:

```bash
pre-commit run --all-files
```

4. Push your branch to your fork:

```bash
git push origin feature-or-fix-name
```

5. Open a pull request against the `main` branch of the nxbench repository.
6. Ensure the PR description clearly describes the problem and solution.
7. Reference any relevant issues in the PR description.

## Reporting Issues

- Use the GitHub issue tracker to report bugs or suggest enhancements.
- Provide a clear and detailed description of the issue or suggestion.
- Include steps to reproduce for bugs, and example use cases for enhancements.

Thank you for contributing to nxbench! 🎉
