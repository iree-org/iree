# Getting Started with Python

IREE has Python bindings geared towards lower level compiler interop that are
not intended to be a public API, and integration with Python frontends such as
TensorFlow.

We do not yet provide a pip package for easy installation, so to use IREE's
Python bindings you must build from source.

## Prerequisites

You should already have IREE cloned and building on your machine. See the other
[getting started guides](.) for instructions.

> Note:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Support is best with Bazel.
> For CMake (excluding TensorFlow), set the `IREE_BUILD_PYTHON_BINDINGS` option.

## Python Setup

Install a recent version of [Python 3](https://www.python.org/downloads/) and
[pip](https://pip.pypa.io/en/stable/installing/), if needed.

Install packages:

```shell
$ python3 -m pip install --upgrade pip
$ python3 -m pip install numpy

# If using the TensorFlow integration
$ python3 -m pip install tf-nightly
```

## Running Python Tests

To run tests for core Python bindings built with CMake:

```shell
$ cd build
$ ctest -L bindings/python
```

To run tests for core Python bindings built with Bazel:

```shell
$ bazel test bindings/python/...
```

To run tests for the TensorFlow integration, which include end-to-end backend
comparison tests:

```shell
# Exclude tests that are skipped in the Kokoro CI
$ bazel test \
  --build_tag_filters="-nokokoro" \
  --test_tag_filters="-nokokoro" \
  --define=iree_tensorflow=true \
  integrations/tensorflow/...
```

## Using Colab

See
[start_colab_kernel.py](https://github.com/google/iree/blob/main/colab/start_colab_kernel.py)
and [Using Colab](https://github.com/google/iree/blob/main/docs/using_colab.md)
for setup instructions, then take a look through the
[Colab directory](https://github.com/google/iree/tree/main/colab) for some
sample notebooks.
