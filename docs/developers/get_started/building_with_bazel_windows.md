# Getting Started on Windows with Bazel

**NOTE** Bazel build support is primarily for internal project infrastructure.
Bazel on Windows in particular is particularly unstable and unsupported.
We strongly recommend users build with CMake instead.

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
[`.bazelversion`](https://github.com/openxla/iree/blob/main/.bazelversion) for
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
> git clone https://github.com/openxla/iree.git
> cd iree
> git submodule update --init
> python configure_bazel.py
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Clone to a short path like `C:\projects\` to avoid
> issues with Windows maximum path lengths (260 characters).

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Editors and other programs can also clone the
> repository, just make sure that they initialize the submodules.

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;configure_bazel.py only detects that you have Windows
> and will output the default `--config=windows` to `configured.bazelrc`, which
> assumes the latest version of MSVC. To avoid some warnings, you may want to
> replace it with `--config=msvc2017`.

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
> bazel build tools/...
```

Check out what was built:

```powershell
> dir bazel-bin\iree\tools\
> .\bazel-bin\tools\iree-compile.exe --help
```

Translate a
[MLIR file](https://github.com/openxla/iree/blob/main/samples/models/simple_abs.mlir)
and execute a function in the compiled module:

```powershell
> REM iree-run-mlir <compiler flags> [input.mlir] <runtime flags>
> .\bazel-bin\tools\iree-run-mlir.exe --iree-hal-target-backends=vmvx --print-mlir .\iree\samples\models\simple_abs.mlir --input=f32=-2
```
