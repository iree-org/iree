# IREE Python API

Top-level packages:

* `pyiree.compiler` : Main compiler API.
* `pyiree.rt` : Runtime components for executing binaries.
* `pyiree.tools.core` : Core tools for executing the compiler.
* `pyiree.tools.tf` : TensorFlow compiler tools (if enabled).

Deprecated packages:

* `pyiree.compiler`
* `pyiree.common`
* `pyiree.tf.compiler`

## Installing

First perform a normal CMake build/install with the following options:

* `-DCMAKE_INSTALL_PREFIX=...path to install to...` : Sets up installation
  prefix.
* `-DIREE_BUILD_PYTHON_BINDINGS=ON` : Enables Python Bindings

Then from the install directory, run:

```shell
# Multiple packages will exist under python_packages. Choose the one you want.
cd python_packages/iree_compiler
# Install into a local installation or virtualenv.
python setup.py install
python -m pip wheel .
```

## Development mode

For development, just set your `PYTHONPATH` environment variable to the
`bindings/python` directory in your CMake build dir.

## Run tests

Tests under `bindings/python/tests` can be run directly once installed.
Additional tests under `integrations/tensorflow/e2e` will be runnable soon.
