# Getting Started on Windows with CMake

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

Install CMake version >= 3.13 from the
[downloads page](https://cmake.org/download/).

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Your editor of choice likely has plugins for CMake,
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
    > "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
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

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Editors and other programs can also clone the
> repository, just make sure that they initialize the submodules.

### Build

Configure:

```powershell
> cmake -G Ninja -B build\ .
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;The root
> [CMakeLists.txt](https://github.com/google/iree/blob/main/CMakeLists.txt)
> file has options for configuring which parts of the project to enable.<br>
> &nbsp;&nbsp;&nbsp;&nbsp;These are further documented in [CMake Options and Variables](cmake_options_and_variables.md).

Build all targets:

```powershell
> cmake --build build\
```

## What's next?

### Take a Look Around

Check out the contents of the 'tools' build directory:

```powershell
> dir build\iree\tools
> .\build\iree\tools\iree-translate.exe --help
```

Translate a
[MLIR file](https://github.com/google/iree/blob/main/iree/tools/test/simple.mlir)
and execute a function in the compiled module:

```powershell
> .\build\iree\tools\iree-run-mlir.exe .\iree\tools\test\simple.mlir -input-value="i32=-2" -iree-hal-target-backends=vmla -print-mlir
```

### Further Reading

*   For an introduction to IREE's project structure and developer tools, see
    [Developer Overview](../developing_iree/developer_overview.md)
*   To target GPUs using Vulkan, see
    [Getting Started on Windows with Vulkan](getting_started_windows_vulkan.md)
*   To use IREE's Python bindings, see
    [Getting Started with Python](getting_started_python.md)
