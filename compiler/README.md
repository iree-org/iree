# IREE Compiler

This directory contains the IREE compiler sources.

## Standalone Python Builds

The included `setup.py` file can be used to build Python binaries or directly
install the IREE compiler API. Do note that the compiler is quite heavy and
unless you are developing it and on a significant machine, you will want to
use released binaries.

There are two ways to build/install Python packages:

* Directly from the source tree (this is how official releases are done).
* From the build directory while developing.

It is recommended to use your favorite method for managing
[virtual environemnts](https://docs.python.org/3/library/venv.html) instead
of modifying the system installation.

Only relatively recent versions of `pip` are supported. Always use the latest
via `pip install --upgrade pip`.

You can build either from the source or build tree (assumes that CMake has
been configured and the project built). The latter is typically used by
project developers who are already setup for development and want to
incrementally generate Python packages without rebuilding.

To build a wheel that can be installed on the same Python version and OS:

```
python -m pip wheel compiler/
```

To directly install:

```
python -m pip install compiler/
```

In order to sanity check once the package is installed:

```
python compiler/src/iree/compiler/bindings/python/test/transforms/ireec/compile_sample_module.py
```
