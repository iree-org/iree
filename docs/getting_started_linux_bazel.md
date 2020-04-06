# Getting Started on Linux with Bazel

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.

    * This document parallels getting_started_windows_bazel.md.
      Please keep them in sync.
-->

This guide walks through building the core compiler and runtime parts of IREE
from source. Auxilary components like the Python bindings and Vulkan driver are
documented separately, as they require further setup.

## Prerequisites

### Install Bazel

Install Bazel version > 2.0.0 (see
[.bazelversion](https://github.com/google/iree/blob/master/.bazelversion) for
the specific version IREE uses) by following the
[official docs](https://docs.bazel.build/versions/master/install.html).

### Install a Compiler

We recommend Clang. GCC is not fully supported.

```shell
$ sudo apt install clang
```

Set environment variables:

```shell
export CC=clang
export CXX=clang++
```

## Clone and Build

### Clone

Clone the repository and initialize its submodules:

```shell
$ git clone https://github.com/google/iree.git
$ cd iree
$ git submodule update --init
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

```
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
[MLIR file](https://github.com/google/iree/blob/master/iree/tools/test/simple.mlir)
and execute a function in the compiled module:

```shell
$ ./bazel-bin/iree/tools/iree-run-mlir ./iree/tools/test/simple.mlir -input-value="i32=-2" -iree-hal-target-backends=vmla -print-mlir
```

### Further Reading

More documentation coming soon...

<!-- TODO(scotttodd): Vulkan on Linux -->
<!-- TODO(scotttodd): Running samples -->
<!-- TODO(scotttodd): "getting_started.md" equivalent for iree-translate etc. -->
