# Getting Started with Python

  NOTE: Iree's Python API is currently being reworked. Some of these
  instructions may be in a state of flux as they document the end state.

IREE has two primary Python APIs:

* Compiler API: `pyiree.compiler2`, `pyiree.compiler2.tf`
* Runtime API: `pyiree.tf`

There are additional ancillary modules that are not part of the public API.

## Prerequisites

You should already have IREE cloned and building on your machine. See the other
[getting started guides](../get-started) for instructions.

> Note:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Support is only complete with CMake.

Minimally, the following CMake flags must be specified:

* `-DIREE_BUILD_PYTHON_BINDINGS=ON`
* `-DIREE_BUILD_TENSORFLOW_COMPILER=ON` : Optional. Also builds the
  TensorFlow compiler integration.

If building any parts of TensorFlow, you must have a working `bazel` command
on your path. See the `.bazelversion` file at the root of the project for the
recommended version.

## Python Setup

Install a recent version of [Python 3](https://www.python.org/downloads/) and
[pip](https://pip.pypa.io/en/stable/installing/), if needed.

(Recommended) Setup a virtual environment (use your preferred mechanism):

```shell
# Note that venv is only available in python3 and is therefore a good check
# that you are in fact running a python3 binary.
python -m venv .venv
source .venv/bin/activate
# When done: run 'deactivate'
```

Install packages:

```shell
$ python -m pip install --upgrade pip
$ python -m pip install numpy absl-py

# If using the TensorFlow integration
$ python -m pip install tf-nightly
```

## Running Python Tests

To run tests for core Python bindings built with CMake:

```shell
$ cd build
$ ctest -L bindings/python
```

To run tests for the TensorFlow integration, which include end-to-end backend
comparison tests:

```shell
cd build
# TODO: Revisit once more patches land.
ctest -L integrations/tensorflow/e2e

# Or run individually as:
export PYTHONPATH=bindings/python # In build dir
python integrations/tensorflow/e2e/simple_arithmetic_test.py \
  --target_backends=iree_vmla --artifacts_dir=/tmp/artifacts
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
