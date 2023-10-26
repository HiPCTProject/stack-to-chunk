# stack-to-chunk

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![codecov](https://codecov.io/gh/HiPCTProject/stack-to-chunk/graph/badge.svg?token=GBOWQFNYMP)](https://codecov.io/gh/HiPCTProject/stack-to-chunk)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/HiPCTProject/stack-to-chunk/main.svg)](https://results.pre-commit.ci/latest/github/HiPCTProject/stack-to-chunk/main)
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

Convert stacks of images to a chunked zarr dataset that can be used for visualisation with [neuroglancer](https://github.com/google/neuroglancer).

Specifically this code is written to:

- Take stacks of 2D images (e.g., TIFF, JPEG files) that represent a 3D spatial volume as input.
- Convert them to an [OME Next Generation File Format (NGFF)](https://ngff.openmicroscopy.org/0.4/index.html) zarr dataset suitable for multiscale viewing with [neuroglancer](https://github.com/google/neuroglancer).

## Internals

The code is designed based on the following assumptions:

1. Input data are stored in individual 2D slices. Reading part of a single slice requires reading the whole slice into memory, and this is an expensive operation.
1. Writing a single chunk of output data is an expensive operation.
1. Reading a single chunk of output data is a cheap operation.

If we have input slices of shape `(nx, ny)`, and an output chunk shape of `(nc, nc, nc)` it makes sense to split the conversion into individual 'slabs' that have shape `(nx, ny, nc)`. This means there is a one-to-one mapping from slices to slabs, and slabs to chunks, allowing each slab to be processed in parallel without interfering with the other slabs.

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
