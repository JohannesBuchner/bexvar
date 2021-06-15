.PHONY: clean clean-test clean-pyc clean-build docs help show
.DEFAULT_GOAL := help
.SECONDARY:

export PRINT_HELP_PYSCRIPT

PYTHON := python3

help:
	@$(PYTHON) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-doc ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '*.so' -exec rm -f {} +
	find . -name '*.c' -exec rm -f {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

clean-doc:
	rm -rf README.html

#lint: ## check style with flake8
#	flake8 snowline tests

test: ## run tests quickly with the default Python
	${PYTHON} tutorial/run.py
	rst2html5.py README.rst > README.html

test-all: ## run tests on every Python version with tox
	tox

show: flatdist.txt.gz_out_gauss/plots/corner.pdf
	xdg-open $^

coverage: ## check code coverage quickly with the default Python
	coverage run --source snowline -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

doc: ## generate Sphinx HTML documentation, including API docs
	# python3 tutorial/run.py
	rst2html5.py README.rst > README.html

release: dist ## package and upload a release
	twine upload -s dist/*.tar.gz

dist: clean ## builds source and wheel package
	$(PYTHON) setup.py sdist
	$(PYTHON) setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	$(PYTHON) setup.py install
