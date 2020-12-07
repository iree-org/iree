# IREE Python API

Top-level packages:

* `pyiree.compiler2` : Main compiler API (soon to be renamed to 'compiler').
* `pyiree.rt` : Runtime components for executing binaries.
* `pyiree.tools.core` : Core tools for executing the compiler.
* `pyiree.tools.tf` : TensorFlow compiler tools (if enabled).

Deprecated packages:

* `pyiree.compiler`
* `pyiree.common`
* `pyiree.tf.compiler`

## Installing

First perform a normal CMake build with the following options:

* `-DIREE_BUILD_PYTHON_BINDINGS=ON` : Enables Python Bindings
* `-DIREE_BUILD_TENSORFLOW_COMPILER=ON` (optional) : Enables building the
  TensorFlow compilers (note: requires additional dependencies. see overall
  build docs).

Then from the build directory, run:

```shell
# Install into a local installation or virtualenv.
python bindings/python/setup.py install

# Build wheels.
python bindings/python/setup.py bdist_wheel
```

## Development mode

For development, just set your `PYTHONPATH` environment variable to the
`bindings/python` directory in your CMake build dir.

## Run tests

Tests under `bindings/python/tests` can be run directly once installed.
Additional tests under `integrations/tensorflow/e2e` will be runnable soon.
