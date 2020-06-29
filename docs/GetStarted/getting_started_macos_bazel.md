# Getting Started on macOS with Bazel

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.

    * This document parallels getting_started_linux_bazel.md and
      getting_started_windows_bazel.md
      Please keep them in sync.
-->

This guide walks through building the core compiler and runtime parts of IREE
from source. Auxiliary components like the Python bindings and Vulkan driver are
not documented for macOS at this time.

IREE is not officially supported on macOS at this time. It may work, but it is
not a part of our open source CI, and may be intermittently broken.
Contributions related to macOS support and documentation are welcome however.

## Prerequisites

### Install Homebrew

This guide uses [Homebrew](https://brew.sh/) to install IREE's dependencies.

```shell
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

### Install Bazel

Install Bazel via Homebrew:

```shell
$ brew install bazel
```

Note: when you first run `bazel` to build IREE, it will prompt you to copy and
run a shell command to select the right version.

### Install python3 numpy

```shell
$ python3 -m pip install numpy --user
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

Run all core tests that pass on our OSS CI:

```shell
$ bazel test -k //iree/... \
    --test_env=IREE_VULKAN_DISABLE=1 \
    --build_tag_filters="-nokokoro" \
    --test_tag_filters="--nokokoro,-driver=vulkan"
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Not all tests are passing on macOS, but the build does
> complete successfully at the time of writing.

In general, build artifacts will be under the `bazel-bin` directory at the top
level.

## Recommended user.bazelrc

You can put a user.bazelrc at the root of the repository and it will be ignored
by git. The recommended contents for Linux/macOS are:

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
    [Developer Overview](../developer_overview.md) <!-- TODO: Link to macOS
    versions of these guides once they are developed.
*   To target GPUs using Vulkan, see
    [Getting Started on Linux with Vulkan](getting_started_linux_vulkan.md)
*   To use IREE's Python bindings, see
    [Getting Started with Python](getting_started_python.md) -->
