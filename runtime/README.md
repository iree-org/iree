# IREE runtime

Note that this directory is in a transitional state. The C code still lives
in directories under `iree/` and will be relocated here in the future.

## Language Bindings

### Python

The included `setup.py` file can be used to build Python binaries or directly
install the IREE runtime API. Do note that the runtime is quite heavy and
unless you are developing it and on a significant machine, you will want to
use released binaries.

There are two ways to build/install Python packages:

* Directly from the source tree (this is how official releases are done).
* From the build directory while developing.

It is recommended to use your favorite method for managing
[virtual environments](https://docs.python.org/3/library/venv.html) instead
of modifying the system installation.

Only relatively recent versions of `pip` are supported. Always use the latest
via `pip install --upgrade pip`.

You can build either from the source or build tree (assumes that CMake has
been configured and the project built). The latter is typically used by
project developers who are already setup for development and want to
incrementally generate Python packages without rebuilding.

To build a wheel that can be installed on the same Python version and OS:

```
python -m pip wheel runtime/
```

To directly install:

```
python -m pip install runtime/
```

