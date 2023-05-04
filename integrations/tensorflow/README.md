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

## Updating Tenserflow Importers in CI

CI uses Tenserflow importers to run integration tests and benchmarks. They might
need an update in CI if you want new features/bugfixes from the frontends.

Tenserflow importers are wrappers which call Tensorflow Python API to do
conversion. CI installs a pinned version of Tensorflow in the docker images. To
bump the Tenserflow version, you need to:

1.  Update the Tensorflow pinned version in
    [integrations/tensorflow/test/requirements.txt](integrations/tensorflow/test/requirements.txt).
2.  Follow [build_tools/docker/README.md](build_tools/docker/README.md) to
    rebuild the `frontends` docker image and its descendants.

Here is the command to rebuild and update the docker images:

```sh
python3 build_tools/docker/manage_images.py --image frontends
```

To modify the import tools themselves, you can directly change their code in
[integrations/tensorflow/python_projects](integrations/tensorflow/python_projects)
without updating the dockers.
