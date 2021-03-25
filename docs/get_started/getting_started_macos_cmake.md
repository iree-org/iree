---
layout: default
permalink: get-started/getting-started-macos-cmake
title: macOS with CMake
nav_order: 7
parent: Getting Started
---

# Getting Started on macOS with CMake
{: .no_toc }

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.

    * This document parallels getting_started_linux_cmake.md and
      getting_started_windows_cmake.md
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

### Install CMake

IREE uses [CMake](https://cmake.org/) version `>= 3.13.4`. Brew ships the latest
release.

```shell
$ brew install cmake
$ cmake --version  # >= 3.13.4
```

### Install Ninja

[Ninja](https://ninja-build.org/) is a fast build system that you can use as a
CMake generator.

```shell
$ brew install ninja
```

## Clone and Build

### Clone

Clone the repository and initialize its submodules:

```shell
$ git clone https://github.com/google/iree.git
$ cd iree
$ git submodule update --init
```

> Tip
> {: .label .label-green }
> Editors and other programs can also clone the
> repository, just make sure that they initialize the submodules.

### Build

Configure:

```shell
$ cmake -G Ninja -B ../iree-build/ .
```

Note: this should use `Clang` by default on macOS. `GCC` is not fully supported
by IREE.

> Tip
> {: .label .label-green }
> The root
> [CMakeLists.txt](https://github.com/google/iree/blob/main/CMakeLists.txt)
> file has options for configuring which parts of the project to enable.<br>
> &nbsp;&nbsp;&nbsp;&nbsp;These are further documented in [CMake Options and Variables](cmake_options_and_variables.md).

Build all targets:

```shell
$ cmake --build ../iree-build/
```

## What's next?

### Take a Look Around

Check out the contents of the 'tools' build directory:

```shell
$ ls ../iree-build/iree/tools
$ ../iree-build/iree/tools/iree-translate --help
```

Translate a
[MLIR file](https://github.com/google/iree/blob/main/iree/tools/test/iree-run-mlir.mlir)
and execute a function in the compiled module:

```shell
$ ../iree-build/iree/tools/iree-run-mlir $PWD/iree/tools/test/iree-run-mlir.mlir \
  -function-input="i32=-2" -iree-hal-target-backends=vmla -print-mlir
```

### Further Reading

*   For an introduction to IREE's project structure and developer tools, see
    [Developer Overview](../developing_iree/developer_overview.md).
*   To understand how IREE implements a HAL driver using Metal, see
    [Metal HAL Driver](../design_docs/metal_hal_driver.md). <!-- TODO: Link to
    macOS versions of these guides once they are developed.
*   To use IREE's Python bindings, see
    [Getting Started with Python](getting_started_python.md) -->