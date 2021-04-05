---
layout: default
permalink: get-started/getting-started-tensorflow
title: TensorFlow
nav_order: 11
parent: Getting Started
---

# Getting Started with IREE TensorFlow Integrations
{: .no_toc }

  NOTE: Iree's Python API is currently being reworked. Some of these
  instructions may be in a state of flux as they document the end state.

## Prerequisites

You should have already completed the
[Python getting started guide](../get-started/getting-started-python). Install
the TensorFlow pip package:

```shell
$ python -m pip install tf-nightly
```

## Obtaining IREE Integration Binaries

IREE's compiler integrations into TensorFlow are mediated by standalone binaries
that can be built individually or installed from a distribution. These binaries
are: `iree-tf-import`, `iree-import-tflite`, and `iree-import-xla`. They are
configured in the
[iree_tf_compiler BUILD file](https://github.com/google/iree/blob/main/integrations/tensorflow/iree_tf_compiler/BUILD).
You have a few options for how to obtain these binaries

### Option 1. Building with Bazel

TensorFlow only supports the Bazel build system. If building any parts of
TensorFlow yourself, you must have a working `bazel` command on your path. See
the relevant "OS with Bazel" [getting started](../get-started) doc for more
information.

> Warning:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Building TF binaries takes a very long time,
> especially on smallish machines (IREE devs that work on these typically use
> machines with 96 cores)

For example to run TensorFlow-based tests, you can build `iree-import-tf`

TODO(4979): Rename iree-tf-import to iree-import-tf

```shell
python3 configure_bazel.py
cd integrations/tensorflow
bazel build \
  //iree_tf_compiler:iree-tf-import \
  //iree_tf_compiler:iree-import-tflite \
  //iree_tf_compiler:iree-import-xla

```

The directory containing the binary will be printed (i.e.
`bazel-bin/iree_tf_compiler/iree-tf-import`) and the parent directory must be
passed to `-DIREE_TF_TOOLS_ROOT=` in a subsequent CMake invocation.

### Option 2. Install from a release

TODO(#4980): Document how this works

Roughly:

```shell
python -m pip install \
  iree-tools-tf-snapshot \
  iree-tools-tflite-snapshot \
  iree-tools-xla-snapshot \
  -f https://github.com/google/iree/releases
```

TODO: Need a more sophisticated mechanism than `IREE_TF_TOOLS_ROOT`, which
only supports one directory. Also provide a tool to fetch and unpack
release binaries.

## Building

The IREE Python bindings are only buildable with CMake. Continuing from above,
to build with TensorFlow support, add `-DIREE_BUILD_TENSORFLOW_COMPILER=ON` to
your invocation. If you obtained the integration binaries by a method other than
building them with Bazel, you will also need to pass the path to the directory
in which they are located: `-DIREE_TF_TOOLS_ROOT=path/to/dir` (it defaults to
the location where the Bazel build creates them). From the IREE root directory:

The following CMake flags control:

* `-DIREE_BUILD_TENSORFLOW_COMPILER=ON`: build the TensorFlow integration.
* `-DIREE_BUILD_TFLITE_COMPILER=ON`: build the TFLite integration.
* `-DIREE_BUILD_XLA_COMPILER=ON`: build the XLA integration.
* `-DIREE_TF_TOOLS_ROOT`: path to directory containing separately-built tools
  for the enabled integrations.

```shell
$ cmake -B ../iree-build-tf -G Ninja \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DIREE_BUILD_TENSORFLOW_COMPILER=ON .
$ cmake --build ../iree-build-tf
```

## Running Tests

To run tests for the TensorFlow integration, which include end-to-end backend
comparison tests:

```shell
$ cd ../iree-build-tf
$ ctest -R 'tensorflow_e2e|bindings/python|integrations/tensorflow/' \
  --output-on-failure

# Or run individually as:
$ export PYTHONPATH=$(pwd)/bindings/python
# This is a Python 3 program. On some systems, such as Debian derivatives,
# use 'python3' instead of 'python'.
$ python ../iree/integrations/tensorflow/e2e/simple_arithmetic_test.py \
    --target_backends=iree_vmla --artifacts_dir=/tmp/artifacts
```