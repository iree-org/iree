# IREE TensorFlow Importers

This project contains IREE frontends for importing various forms of TensorFlow
formats.

## Quick Development Setup

Pip install editable (recommend to do this in a virtual environment):

```
# All at once:
pip install -e python_projects/*

# Or one at a time:
pip install -e python_projects/iree_tflite
pip install -e python_projects/iree_tf
```

Test installed:

```
iree-import-tflite -h
iree-import-tf -h
```

## Run test suite

You need to make sure that the iree compiler and runtime are on your PYTHONPATH.
The easiest way to do this is to install wheels with pip. For development,
the following should do it:

```
source ~/path/to/iree-build/.env && export PYTHONPATH
```

Run the test suite with:

```
pip install lit

# Just run the default tests:
lit -v test/

# Can also run vulkan tests with:
lit -v -D FEATURES=vulkan test/

# Can disable the default LLVM CPU tests:
lit -v -D DISABLE_FEATURES=llvmcpu -D FEATURES=vulkan test/

# Individual test directories, files or globs can be run individually.
lit -v $(find test -name '*softplus*')
```

## Updating Tensorflow Importers in CI

CI uses Tensorflow importers to run integration tests and benchmarks. They might
need an update in CI if you want new features or bugfixes from the frontends.

Tensorflow importers are wrappers which call Tensorflow Python APIs to perform
conversions. To bump the Tensorflow version, you need to update the
pinned version of Tensorflow that CI jobs install in
[integrations/tensorflow/test/requirements.txt](/integrations/tensorflow/test/requirements.txt).
