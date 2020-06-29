# Getting Started on Windows with Bazel

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.

    * This document parallels getting_started_linux_bazel.md and
      getting_started_macos_bazel.md
      Please keep them in sync.
-->

This guide walks through building the core compiler and runtime parts of IREE
from source. Auxiliary components like the Python bindings and Vulkan driver are
documented separately, as they require further setup.

## Prerequisites

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;You can simplify installation by using a package
> manager like [Scoop](https://scoop.sh/) or
> [Chocolatey](https://chocolatey.org/).

### Install Bazel

Install Bazel version > 2.0.0 (see
[`.bazelversion`](https://github.com/google/iree/blob/main/.bazelversion) for
the specific version IREE uses) by following the
[official docs](https://docs.bazel.build/versions/master/install-windows.html).

Also install [MSYS2](https://www.msys2.org/) by following Bazel's documentation.

### Install Python3

Instructions for installation can be found
[here](https://www.python.org/downloads/windows/).

### Install Build Tools For Visual Studio

Install the full Visual Studio or "Build Tools For Visual Studio" from the
[downloads page](https://visualstudio.microsoft.com/downloads/).

Set a few environment variables. You are welcome to configure these however you
choose. For example, you could set them as system or user level environment
variables through your "System Properties" or you could use a shell such as
PowerShell or [cmder](https://cmder.net/)). Setting them through PowerShell
would look like this:

```powershell
> $env:BAZEL_VS = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
```

## Clone and Build

### Clone

Using your shell of choice (such as PowerShell or [cmder](https://cmder.net/)),
clone the repository, initialize its submodules, and configure:

```powershell
> git clone https://github.com/google/iree.git
> cd iree
> git submodule update --init
> python configure_bazel.py
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Editors and other programs can also clone the
> repository, just make sure that they initialize the submodules.

### Build

Run all core tests:

```powershell
> bazel test -k iree/...
```

In general, build artifacts will be under the `bazel-bin` directory at the top
level.

## Recommended user.bazelrc

You can put a user.bazelrc at the root of the repository and it will be ignored
by git. The recommended contents for Windows are:

```
build --disk_cache=c:/bazelcache
build:debug --compilation_mode=dbg --copt=/O2 --per_file_copt=iree@/Od --strip=never
```

## What's next?

### Take a Look Around

Build all of IREE's 'tools' directory:

```powershell
> bazel build iree/tools/...
```

Check out what was built:

```powershell
> dir bazel-bin\iree\tools\
> .\bazel-bin\iree\tools\iree-translate.exe --help
```

Translate a
[MLIR file](https://github.com/google/iree/blob/main/iree/tools/test/simple.mlir)
and execute a function in the compiled module:

```powershell
> .\bazel-bin\iree\tools\iree-run-mlir.exe .\iree\tools\test\simple.mlir -input-value="i32=-2" -iree-hal-target-backends=vmla -print-mlir
```

### Further Reading

*   For an introduction to IREE's project structure and developer tools, see
    [Developer Overview](../developer_overview.md)
*   To target GPUs using Vulkan, see
    [Getting Started on Windows with Vulkan](getting_started_windows_vulkan.md)
*   To use IREE's Python bindings, see
    [Getting Started with Python](getting_started_python.md)
