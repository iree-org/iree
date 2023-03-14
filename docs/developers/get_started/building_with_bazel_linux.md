# Getting Started on Linux with Bazel

**NOTE** Bazel build support is primarily for internal project infrastructure.
We strongly recommend users build with CMake instead.

This guide walks through building the core compiler and runtime parts of IREE
from source. Auxiliary components like the Python bindings and Vulkan driver are
documented separately, as they require further setup.

## Prerequisites

### Install Bazel

Install Bazel, matching IREE's
[`.bazelversion`](https://github.com/openxla/iree/blob/main/.bazelversion) by
following the
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
$ git clone https://github.com/openxla/iree.git
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

# Use --config=debug to compile IREE and LLVM without optimizations
# and with assertions enabled.
build:debug --config=asserts --compilation_mode=opt '--per_file_copt=iree|llvm@-O0' --strip=never

# Use --config=asserts to enable assertions. This has to be done globally:
# Code compiled with and without assertions can't be linked together (ODR violation).
build:asserts --compilation_mode=opt '--copt=-UNDEBUG'
```

## What's next?

### Take a Look Around

Build all of IREE's 'tools' directory:

```shell
$ bazel build tools/...
```

Check out what was built:

```shell
$ ls bazel-bin/tools/
$ ./bazel-bin/tools/iree-compile --help
```

Translate a
[MLIR file](https://github.com/openxla/iree/blob/main/samples/models/simple_abs.mlir)
and execute a function in the compiled module:

```shell
# iree-run-mlir <compiler flags> [input.mlir] <runtime flags>
$ ./bazel-bin/tools/iree-run-mlir \
  --iree-hal-target-backends=vmvx --print-mlir \
  ./samples/models/simple_abs.mlir \
  --input=f32=-2
```
