---
layout: default
permalink: get-started/getting-started-python
title: Python
nav_order: 10
parent: Getting Started
---

# Getting Started with Python
{: .no_toc }

  NOTE: Iree's Python API is currently being reworked. Some of these
  instructions may be in a state of flux as they document the end state.

The IREE compiler API is called `iree.compiler`.

There are additional ancillary modules that are not part of the public API.

Note this guide does not cover IREE integrations with other frontends, such as
TensorFlow. For those, see the relevant
[getting started guides](../get-started).

## Prerequisites

You should already have IREE cloned and building on your machine. See the other
[getting started guides](../get-started) for instructions.

> Note
> {: .label .label-blue }
> Support is only complete with CMake.

Minimally, the following CMake flags must be specified:

* `-DIREE_BUILD_PYTHON_BINDINGS=ON`

## Python Setup

Install [Python 3](https://www.python.org/downloads/) `>= 3.6` and
[pip](https://pip.pypa.io/en/stable/installing/), if needed.

> Note
> {: .label .label-blue }
> If using `pyenv` (or an interpreter manager that
  depends on it like `asdf`), you'll need to use
  [`--enable-shared`](https://github.com/pyenv/pyenv/tree/master/plugins/python-build#building-with---enable-shared)
  during interpreter installation.

(Recommended) Setup a virtual environment with `venv` (or your preferred
mechanism):

```shell
# Note that venv is only available in python3 and is therefore a good check
# that you are in fact running a python3 binary.
$ python -m venv .venv
$ source .venv/bin/activate
# When done: run 'deactivate'
```

As we distribute `manylinux2014` binaries, your pip version should be listed as
supported on the compatibility table at https://github.com/pypa/manylinux. As
needed, you can upgrade pip using:

```shell
$ python -m pip install --upgrade pip
```

Install packages:

```shell
$ python -m pip install numpy absl-py
```

## Building

From the *parent* directory of the IREE git repository clone, create and enter a
build directory, such as:

```shell
$ mkdir iree-build
$ cd iree-build
```

Then build like this:

```shell
$ cmake ../iree -G Ninja \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_BUILD_PYTHON_BINDINGS=ON  .
$ cmake --build .
```

## Running Python Tests

We continue to assume that we are in the build directory where we made the build
in the previous section.

To run tests for core Python bindings built with CMake:

```shell
$ ctest -L bindings/python
```
## Using Colab

There are some sample colabs in the `colab` folder. If you have built the
project with CMake/ninja and set your `PYTHONPATH` to the `bindings/python`
directory in the build dir (or installed per below), you should be able to
start a kernel by following the stock instructions at
https://colab.research.google.com/ .

## Installing and Packaging

There is a `setup.py` in the `bindings/python` directory under the build dir.
To install into your (hopefully isolated) virtual env:

```shell
# See the above note about python3, and the above step setting PYTHONPATH.
python bindings/python/setup.py install
```

To create wheels (platform dependent and locked to your Python version
without further config):

```shell
python bindings/python/setup.py bdist_wheel
```

Note that it is often helpful to differentiate between the environment used to
build and the one used to install. While this is just "normal" python
knowledge, here is an incantation to do so:

```shell
# From parent/build environment.
python -m pip freeze > /tmp/requirements.txt
deactivate  # If already in an environment

# Enter new scratch environment.
python -m venv ./.venv-scratch
source ./.venv-scratch/bin/activate
python -m pip install -r /tmp/requirements.txt

# Install IREE into the new environment.
python bindings/python/setup.py install
```