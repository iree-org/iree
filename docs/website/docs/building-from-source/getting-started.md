# Getting started

## Prerequisites

You will need to install [CMake](https://cmake.org/), the
[Ninja](https://ninja-build.org/) CMake generator, and the clang or MSVC C/C++
compilers. The tests also requires [Python3](https://www.python.org/) and the
python package [requests](https://requests.readthedocs.io/en/latest/) to run.

???+ Note
    You are welcome to try different CMake generators and compilers, but IREE
    devs and CIs exclusively use these and other configurations are "best
    effort". Additionally, compilation on macOS is "best effort" as well, though
    we generally expect it to work due to its similarity with Linux. Patches to
    improve support for these are always welcome.

=== "Linux"

    1. Install a compiler/linker (typically "clang" and "lld" package)

    2. Install [CMake](https://cmake.org/download/) (typically "cmake" package)

    3. Install [Ninja](https://ninja-build.org/) (typically "ninja-build"
       package)

    On a relatively recent Debian/Ubuntu:

    ``` shell
    sudo apt install cmake ninja-build clang lld
    ```

=== "macOS"

    1. Install [CMake](https://cmake.org/download/) (typically "cmake" package)

    2. Install [Ninja](https://ninja-build.org/) (typically "ninja-build"
       package)

    If using Homebrew:

    ``` shell
    brew install cmake ninja
    ```

=== "Windows"

    1. Install MSVC from Visual Studio or "Tools for Visual Studio" on the
       [official downloads page](https://visualstudio.microsoft.com/downloads/)

    2. Install CMake from the
       [official downloads page](https://cmake.org/download/)

    3. Install Ninja either from the
       [official site](https://ninja-build.org/)

    !!! note
        You will need to initialize MSVC by running `vcvarsall.bat` to use it
        from the command line. See the
        [official documentation](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line)
        for details.

## Clone and build

Use [Git](https://git-scm.com/) to clone the IREE repository and initialize its
submodules:

``` shell
git clone https://github.com/openxla/iree.git
cd iree
git submodule update --init
```

Configure then build all targets using CMake:

Configure CMake:

=== "Linux"

    ``` shell
    # Recommended for simple development using clang and lld:
    cmake -GNinja -B ../iree-build/ -S . \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DIREE_ENABLE_LLD=ON

    # Alternately, with system compiler and your choice of CMake generator:
    # cmake -B ../iree-build/ -S .
    ```

=== "macOS"

    ``` shell
    # Recommended for simple development using clang and lld:
    cmake -GNinja -B ../iree-build/ -S . \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_ENABLE_ASSERTIONS=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DIREE_ENABLE_LLD=ON

    # Alternately, with system compiler and your choice of CMake generator:
    # cmake -B ../iree-build/ -S .
    ```

=== "Windows"

    ``` shell
    cmake -GNinja -B ../iree-build/ -S . \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_ENABLE_ASSERTIONS=ON
    ```

Build:

``` shell
cmake --build ../iree-build/
```

???+ Tip "Tip - Build types"
    We recommend using the `RelWithDebInfo` build type by default for a good
    balance of debugging information and performance. The `Debug`, `Release`,
    and `MinSizeRel` build types are useful in more specific scenarios.
    In particular, note that several useful LLVM debugging features are only
    available in `Debug` builds. See the
    [official CMake documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)
    for general details.

???+ Tip "Tip - Faster recompilation with ccache"
    We recommend using [`ccache`](https://ccache.dev/) together with CMake. To
    use it, configure CMake with:

    ``` shell
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    ```

    See also our [developer documentation for ccache](https://github.com/openxla/iree/blob/main/docs/developers/developing_iree/ccache.md).

## What's next?

<!-- TODO(scotttodd): "at this point you can..." -->

### Running tests

Build test dependencies and run tests:

``` shell
cmake --build ../iree-build --target iree-run-tests
```

Internally, this builds dependencies via the `iree-test-deps` target and
invokes [CTest](https://gitlab.kitware.com/cmake/community/-/wikis/doc/ctest/Testing-With-CTest).

The parallel testing level can be set via the environment variable
`CTEST_PARALLEL_LEVEL` when invoking ctest in this fashion. Instructions
are printed to test with a custom command line.

### Take a look around

Check out the contents of the 'tools' build directory:

``` shell
ls ../iree-build/tools/
../iree-build/tools/iree-compile --help
```

<!-- TODO(scotttodd): troubleshooting section? link to github issues? -->
