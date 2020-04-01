# Getting Started on Linux with CMake

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.

    * This document parallels getting_started_windows_cmake.md.
      Please keep them in sync.
-->

## Prerequisites

### Install CMake

Install CMake version >= 3.13:

```shell
$ sudo apt install cmake
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;Your editor of choice likely has plugins for CMake,
> such as the Visual Studio Code
> [CMake Tools](https://github.com/microsoft/vscode-cmake-tools) extension.

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

Configure:

```shell
$ cmake -B build/ .
```

> Tip:<br>
> &nbsp;&nbsp;&nbsp;&nbsp;The root
> [CMakeLists.txt](https://github.com/google/iree/blob/master/CMakeLists.txt) file
> has options for configuring which parts of the project to enable.

Build all targets:

```shell
$ cmake --build build/ -j 8
```

## What's next?

### Take a Look Around

Check out the contents of the 'tools' build directory:

```shell
$ ls build/iree/tools
$ ./build/iree/tools/iree-translate --help
```

Translate a
[MLIR file](https://github.com/google/iree/blob/master/iree/tools/test/simple.mlir)
and execute a function in the compiled module:

```shell
$ ./build/iree/tools/iree-run-mlir $PWD/iree/tools/test/simple.mlir -input-value="i32=-2" -iree-hal-target-backends=vmla -print-mlir
```

### Further Reading

More documentation coming soon...

<!-- TODO(scotttodd): Vulkan / other driver configuration -->
<!-- TODO(scotttodd): Running tests -->
<!-- TODO(scotttodd): Running samples -->
<!-- TODO(scotttodd): "getting_started.md" equivalent for iree-translate etc. -->
