.PHONY: help install install-dev format lint check test smoke-test demo benchmark notebook clean setup-colab

PYTHON ?= python

help:
	@echo "Scientific ML on Zero Budget — Command Menu"
	@echo "--------------------------------------------------------"
	@echo "make install       : Install core runtime dependencies"
	@echo "make install-dev   : Install development and testing tools"
	@echo "make format        : Auto-format code with Ruff"
	@echo "make lint          : Run Ruff linter"
	@echo "make test          : Run CPU unit tests with pytest"
	@echo "make smoke-test    : Run fast end-to-end CPU training smoke test"
	@echo "make demo          : Run presentation walkthrough script"
	@echo "make benchmark     : Generate benchmark tables and plots"
	@echo "make check         : Run linting and unit test suite"
	@echo "make clean         : Purge caches, temporary files and data"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .
	pre-commit install

format:
	ruff format .

lint:
	ruff check .

check: lint test

test:
	pytest tests/ --cov=utils --cov-report=term-missing

smoke-test:
	$(PYTHON) main_demo.py

demo:
	$(PYTHON) demo_script.py

benchmark:
	$(PYTHON) benchmark_results.py

setup-colab:
	$(PYTHON) setup_colab.py

clean:
	rm -rf __pycache__ utils/__pycache__ tests/__pycache__ .pytest_cache .ruff_cache htmlcov .coverage
	rm -rf sciml_data demo_checkpoints sciml_checkpoints build dist *.egg-info
