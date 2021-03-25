---
layout: default
permalink: get-started/getting-started-windows-cmake
title: Windows with CMake
nav_order: 4
parent: Getting Started
---

# Getting Started on Windows with CMake
{: .no_toc }

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.

    * This document parallels getting_started_linux_cmake.md and
      getting_started_macos_cmake.md
      Please keep them in sync.
-->

This guide walks through building the core compiler and runtime parts of IREE
from source. Auxiliary components like the Python bindings and Vulkan driver are
documented separately, as they require further setup.

## Prerequisites

### Install CMake

Install CMake version >= 3.13.4 from the
[downloads page](https://cmake.org/download/).

> Tip
> {: .label .label-green }
> Your editor of choice likely has plugins for CMake,
> such as the Visual Studio Code
> [CMake Tools](https://github.com/microsoft/vscode-cmake-tools) extension.

### Install Ninja

[Ninja](https://ninja-build.org/) is a fast build system that you can use as a
CMake generator. Download it from the
[releases page](https://github.com/ninja-build/ninja/releases), extract
somewhere, and add it to your PATH.

### Install a Compiler

We recommend MSVC from either the full Visual Studio or from "Build Tools For
Visual Studio":

*   Choose either option from the
    [downloads page](https://visualstudio.microsoft.com/downloads/) and during
    installation make sure you include "C++ Build Tools"
*   Initialize MSVC by running `vcvarsall.bat`:

    ```powershell
    > & "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ```

## Clone and Build

### Clone

Using your shell of choice (such as PowerShell or [cmder](https://cmder.net/)),
clone the repository and initialize its submodules:

```powershell
> git clone https://github.com/google/iree.git
> cd iree
> git submodule update --init
```

> Tip
> {: .label .label-green }
> Clone to a short path like `C:\projects\` to avoid
> issues with Windows maximum path lengths (260 characters).

> Tip
> {: .label .label-green }
> Editors and other programs can also clone the
> repository, just make sure that they initialize the submodules.

### Build

Configure:

```powershell
> cmake -G Ninja -B ..\iree-build\ .
```

> Tip
> {: .label .label-green }
> The root
> [CMakeLists.txt](https://github.com/google/iree/blob/main/CMakeLists.txt)
> file has options for configuring which parts of the project to enable.<br>
> &nbsp;&nbsp;&nbsp;&nbsp;These are further documented in [CMake Options and Variables](cmake_options_and_variables.md).

Build all targets:

```powershell
> cmake --build ..\iree-build\
```

## Target Configuration

### LLVM AOT Backend

`-iree-hal-target-backends=dylib-llvm-aot` can be used to generate modules with
ahead-of-time compiled kernels stored in DLLs. Run the iree-opt/iree-translate
tools from a command prompt with `lld-link.exe` or `link.exe` tools on the
`PATH` and the MSVC/Windows SDK environment variables; the easiest way to get
this configured is to use the `vsvarsall.bat` or `vcvars64.bat` files to set
your environment. See
[the Microsoft documentation](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=vs-2019)
for details on configuring the toolchain.

If you want to manually specify the linker used, set the
`IREE_LLVMAOT_LINKER_PATH` environment variable to the path of the linker:

```powershell
> set IREE_LLVMAOT_LINKER_PATH="C:\Tools\LLVM\bin\lld-link.exe"
```

Translate a source MLIR file into an IREE module:

```powershell
> ..\iree-build\iree\tools\iree-translate.exe \
  -iree-mlir-to-vm-bytecode-module \
  -iree-hal-target-backends=dylib-llvm-aot \
  iree/tools/test/iree-run-module.mlir \
  -o %TMP%/simple-llvm_aot.vmfb
```

Note that this will use the host machine as the target by default, and the
exact target triple and architecture can be specified with flags when
cross-compiling:

```powershell
> ..\iree-build\iree\tools\iree-translate.exe \
  -iree-mlir-to-vm-bytecode-module \
  -iree-hal-target-backends=dylib-llvm-aot \
  -iree-llvm-target-triple=x86_64-pc-windows-msvc \
  -iree-llvm-target-cpu=host \
  -iree-llvm-target-cpu-features=host \
  iree/tools/test/iree-run-module.mlir \
  -o %TMP%/simple-llvm_aot.vmfb
```

## What's next?

### Take a Look Around

Check out the contents of the 'tools' build directory:

```powershell
> dir ..\iree-build\iree\tools
> ..\iree-build\iree\tools\iree-translate.exe --help
```

Translate a
[MLIR file](https://github.com/google/iree/blob/main/iree/tools/test/iree-run-mlir.mlir)
and execute a function in the compiled module:

```powershell
> ..\iree-build\iree\tools\iree-run-mlir.exe .\iree\tools\test\iree-run-mlir.mlir -function-input="i32=-2" -iree-hal-target-backends=vmla -print-mlir
```

### Further Reading

*   For an introduction to IREE's project structure and developer tools, see
    [Developer Overview](../developing_iree/developer_overview.md)
*   To target GPUs using Vulkan, see
    [Getting Started on Windows with Vulkan](getting_started_windows_vulkan.md)
*   To use IREE's Python bindings, see
    [Getting Started with Python](getting_started_python.md)