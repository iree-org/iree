# IREE TensorFlow Importers

This project contains IREE frontends for importing various forms of TensorFlow
formats.

## Quick Development Setup

This assumes that you have an appropriate `bazel` installed.
Build the importer binaries:

```
# All of them (takes a long time).
bazel build iree_tf_compiler:importer-binaries

# Or individuals:
bazel build iree_tf_compiler:iree-import-tflite
bazel build iree_tf_compiler:iree-import-tf
```

Symlink binaries into python packages (only needs to be done once):

```
./symlink_binaries.sh
```

Pip install editable (recommend to do this in a virtual environment):

```
# All at once:
pip install -e python_projects/*

# Or one at a time:
pip install -e python_projects/iree_tflite
pip install -e python_projects/iree_xla
pip install -e python_projects/iree_tf
```

Test installed:

```
iree-import-tflite -help
iree-import-tf -help
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
