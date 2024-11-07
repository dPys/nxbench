.PHONY: clean_local install lint test doc_api doc_html clean_doc

clean_local:
	@find . -name "*.pyc" -type f -delete
	@find . -name "*.py.cover" -type f -delete
	@find . -name "__pycache__" -type d -delete
	@rm -f .coverage coverage.xml
	@ruff clean
	@rm -rf .mypy_cache .pytest_cache build dist nxbench.egg-info htmlcov .benchmarks env

install: clean_local
	pip uninstall nxbench -y
	pip install -e .

lint: clean_local
	isort .
	ruff format .
	black .

test: clean_local
	pytest -vvv --disable-warnings --cov=nxbench --cov-report=xml --cov-report=html --cov-report=term-missing

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = doc
BUILDDIR      = doc/build

doc_html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

doc_api:
	@sphinx-apidoc -o "$(SOURCEDIR)/api" nxbench **/tests/*

clean_doc:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
