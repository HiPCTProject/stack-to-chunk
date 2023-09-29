# stack-to-chunk

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Licence][licence-badge]](./LICENCE.md)

<!--
[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
-->

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/HiPCTProject/stack-to-chunk/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/HiPCTProject/stack-to-chunk/actions/workflows/tests.yml
[linting-badge]:            https://github.com/HiPCTProject/stack-to-chunk/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/HiPCTProject/stack-to-chunk/actions/workflows/linting.yml
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/stack-to-chunk
[conda-link]:               https://github.com/conda-forge/stack-to-chunk-feedstock
[pypi-link]:                https://pypi.org/project/stack-to-chunk/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/stack-to-chunk
[pypi-version]:             https://img.shields.io/pypi/v/stack-to-chunk
[licence-badge]:            https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
<!-- prettier-ignore-end -->

Convert stacks of images to chunked datasets

## About

### Project Team

HiP-CT Project ([d.stansby@ucl.ac.uk](mailto:d.stansby@ucl.ac.uk))

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`stack-to-chunk` requires Python 3.11.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment created using a environment management tool such as [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) or [Conda](https://conda.io/projects/conda/en/latest/). To install the latest development version of `stack-to-chunk` using `pip` in the currently active environment run

```sh
pip install git+https://github.com/HiPCTProject/stack-to-chunk.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/HiPCTProject/stack-to-chunk.git
```

and then install in editable mode by running

```sh
pip install -e .
```

### Running Locally

How to run the application on your local system.

### Running Tests

<!-- How to run tests on your local system. -->

Tests can be run across all compatible Python versions in isolated environments using
[`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

## Acknowledgements
