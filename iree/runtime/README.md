# IREE Higher-Level Runtime API

This directory implements a higher-level runtime API on top of the low level
APIs split across `iree/base/api.h`, `iree/hal/api.h`, and `iree/vm/api.h`.

Using this higher level API may pull in additional dependencies and perform
additional allocations compared to what you can get by directly going to the
lower levels. For the most part, the higher level and lower levels APIs may be
mixed.

See [the demo directory](./demo/) for sample usage.

## Standalone Python Builds

The included `setup.py` file can be used to build Python binaries or directly
install the IREE runtime API. Do note that the runtime is quite heavy and
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
python -m pip wheel iree/runtime
```

To directly install:

```
python -m pip install iree/runtime
```
