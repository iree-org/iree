# Getting Started on Linux with Bazel

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.

    * This document parallels getting_started_windows_bazel.md and
      getting_started_macos_bazel.md
      Please keep them in sync.
-->

This guide walks through building the core compiler and runtime parts of IREE
from source. Auxiliary components like the Python bindings and Vulkan driver are
documented separately, as they require further setup.

## Prerequisites

### Install Bazel

Install Bazel version > 2.0.0 (see
[`.bazelversion`](https://github.com/google/iree/blob/main/.bazelversion) for
the specific version IREE uses) by following the
[official docs](https://docs.bazel.build/versions/master/install.html).

### Install a Compiler

We recommend Clang. GCC is not fully supported.

```shell
$ sudo apt install clang
```

Set environment variables for Bazel:

```shell
export CC=clang
export CXX=clang++
```

### Install python3 numpy

```shell
$ python3 -m pip install numpy
```

## Clone and Build

### Clone

Clone the repository, initialize its submodules and configure:

```shell
$ git clone https://github.com/google/iree.git
$ cd iree
$ git submodule update --init
$ python3 configure_bazel.py
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Editors and other programs can also clone the
> repository, just make sure that they initialize the submodules.

### Build

Run all core tests:

```shell
$ bazel test -k iree/...
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;You can add flags like
> `--test_env=IREE_VULKAN_DISABLE=1` to your test command to change how/which
> tests run.

In general, build artifacts will be under the `bazel-bin` directory at the top
level.

## Recommended user.bazelrc

You can put a user.bazelrc at the root of the repository and it will be ignored
by git. The recommended contents for Linux are:

```shell
build --disk_cache=/tmp/bazel-cache

# Use --config=debug to compile iree and llvm without optimizations
# and with assertions enabled.
build:debug --config=asserts --compilation_mode=opt '--per_file_copt=iree|llvm@-O0' --strip=never

# Use --config=asserts to enable assertions in iree and llvm.
build:asserts --compilation_mode=opt '--per_file_copt=iree|llvm@-UNDEBUG'
```

## What's next?

### Take a Look Around

Build all of IREE's 'tools' directory:

```shell
$ bazel build iree/tools/...
```

Check out what was built:

```shell
$ ls bazel-bin/iree/tools/
$ ./bazel-bin/iree/tools/iree-translate --help
```

Translate a
[MLIR file](https://github.com/google/iree/blob/main/iree/tools/test/simple.mlir)
and execute a function in the compiled module:

```shell
$ ./bazel-bin/iree/tools/iree-run-mlir ./iree/tools/test/simple.mlir \
    -input-value="i32=-2" -iree-hal-target-backends=vmla -print-mlir
```

### Further Reading

*   For an introduction to IREE's project structure and developer tools, see
    [Developer Overview](../developing_iree/developer_overview.md)
*   To target GPUs using Vulkan, see
    [Getting Started on Linux with Vulkan](getting_started_linux_vulkan.md)
*   To use IREE's Python bindings, see
    [Getting Started with Python](getting_started_python.md)
