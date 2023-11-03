---
icon: octicons/bookmark-16
---

# Getting started

## :octicons-download-16: Prerequisites

IREE can be built from source using [CMake](https://cmake.org/). We also
recommend the [Ninja](https://ninja-build.org/) CMake generator and the
[clang](https://clang.llvm.org/) or MSVC C/C++ compilers.

??? Note "Note - Other CMake generators and compilers"
    IREE developers and CIs primarily use Ninja, clang, and MSVC. Other
    configurations (including the Makefile generator and gcc) are "best effort".
    Patches to improve support are always welcome.

=== ":fontawesome-brands-linux: Linux"

    1. Install a compiler/linker (typically "clang" and "lld" package)

    2. Install [CMake](https://cmake.org/download/) (typically "cmake" package)

    3. Install [Ninja](https://ninja-build.org/) (typically "ninja-build"
       package)

    On Debian/Ubuntu:

    ``` shell
    sudo apt install cmake ninja-build clang lld
    ```

=== ":fontawesome-brands-apple: macOS"

    1. Install [CMake](https://cmake.org/download/)

    2. Install [Ninja](https://ninja-build.org/)

    If using Homebrew:

    ``` shell
    brew install cmake ninja
    ```

=== ":fontawesome-brands-windows: Windows"

    1. Install MSVC from Visual Studio or "Tools for Visual Studio" on the
       [official downloads page](https://visualstudio.microsoft.com/downloads/)

    2. Install CMake from the
       [official downloads page](https://cmake.org/download/)

    3. Install Ninja from the [official site](https://ninja-build.org/)

    !!! note
        Initialize MSVC by running `vcvarsall.bat` to build on the command line.
        See the
        [official documentation](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line)
        for details.

<!-- TODO(#12921): add notes about Docker and/or dev containers  -->

## :octicons-rocket-16: Quickstart: clone and build

Use [Git](https://git-scm.com/) to clone the IREE repository and initialize its
submodules:

``` shell
git clone https://github.com/openxla/iree.git
cd iree
git submodule update --init
```

The most basic CMake workflow is:

``` shell
# Configure
cmake -G Ninja -B ../iree-build/ .

# Build
cmake --build ../iree-build/
```

!!! Caution "Caution - slow builds"
    The compiler build is complex. You will want a powerful machine and to tune
    the settings following the next section. In 2023, we've seen builds take
    around 5-10 minutes on 64-core Linux machines.

    Use case permitting, disabling the compiler build with
    `-DIREE_BUILD_COMPILER=OFF` will drastically simplify the build.

## :octicons-sliders-16: Configuration settings

The configure step should be customized for your build environment. These
settings can improve compile and link times substantially.

<!-- TODO(#5804): add notes about CMake presets?  -->

=== ":fontawesome-brands-linux: Linux"

    ``` shell
    # Recommended development options using clang and lld:
    cmake -G Ninja -B ../iree-build/ -S . \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_ENABLE_ASSERTIONS=ON \
        -DIREE_ENABLE_SPLIT_DWARF=ON \
        -DIREE_ENABLE_THIN_ARCHIVES=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DIREE_ENABLE_LLD=ON
    ```

=== ":fontawesome-brands-apple: macOS"

    ``` shell
    # Recommended development options using clang and lld:
    cmake -G Ninja -B ../iree-build/ -S . \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_ENABLE_ASSERTIONS=ON \
        -DIREE_ENABLE_SPLIT_DWARF=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DIREE_ENABLE_LLD=ON
    ```

    It is also possible to add `-DIREE_ENABLE_THIN_ARCHIVES=ON` if the
    `CMAKE_AR` variable is defined and points to the path of either the GNU
    binutils or LLVM `ar` program, overriding the default Apple `ar`.

=== ":fontawesome-brands-windows: Windows"

    ``` shell
    # Recommended development options:
    cmake -G Ninja -B ../iree-build/ -S . \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DIREE_ENABLE_ASSERTIONS=ON
    ```

???+ Tip "Tip - CMAKE_BUILD_TYPE values"
    We recommend using the `RelWithDebInfo` build type by default for a good
    balance of debug info and performance. The `Debug`, `Release`, and
    `MinSizeRel` build types are useful in more specific cases. Note that
    several useful LLVM debugging features are only available in `Debug` builds.
    See the
    [official CMake documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)
    for general details.

???+ Tip "Tip - Faster recompilation with ccache"
    We recommend using [`ccache`](https://ccache.dev/) with CMake, especially
    when rebuilding the compiler. To use it, configure CMake with:

    ``` shell
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    ```

    See also our [developer documentation for ccache](../developers/building/cmake-with-ccache.md).

### :octicons-gear-16: Optional components

By default, the CMake build includes:

* All compiler targets (`llvm-cpu`, `cuda`, `vulkan-spirv`, etc.)
* All runtime HAL drivers (`local-task`, `cuda`, `vulkan`, etc.)
* All compiler input formats (StableHLO, TOSA, etc.)
* All compiler output formats (VM bytecode, C)

The default build does _not_ include:

* Compiler or runtime bindings (Python, TFLite, etc.)
* Advanced features like AddressSanitizer or tracing instrumentation
* Experimental components

These can be changed via the `IREE_` CMake options listed in the root
[`CMakeLists.txt`](https://github.com/openxla/iree/blob/main/CMakeLists.txt).

### Extensions and integrations

When using IREE within other projects, you can register compiler plugins and
runtime HAL drivers. You can also bring your own copy of LLVM and some other
tools. See the root
[`CMakeLists.txt`](https://github.com/openxla/iree/blob/main/CMakeLists.txt)
for details.

## :octicons-code-16: Tests and samples

### Running tests

Tests are run via [ctest](https://cmake.org/cmake/help/latest/manual/ctest.1.html).
To build and run the core project tests:

``` shell
# Build default targets
cmake --build ../iree-build/

# Run tests
ctest --test-dir ../iree-build/
```

!!! Caution
    This has two limitations:

    1. Large tests are excluded from the build by default
    2. Some tests require hardware like a GPU and will fail on unsupported systems

To build and then run all tests:

``` shell
# 1. Build default targets
cmake --build ../iree-build/

# 2. Build test dependencies
cmake --build ../iree-build/ --target iree-test-deps

# 3. Run tests
ctest --test-dir ../iree-build/


# Or combine all steps using a utility target
cmake --build ../iree-build --target iree-run-tests
```

To run only certain tests, we have a
[helper script](https://github.com/openxla/iree/blob/main/build_tools/cmake/ctest_all.sh)
that converts environment variables into ctest filters:

``` shell
# Run default tests
./build_tools/cmake/ctest_all.sh ../iree-build

# Run tests, turning CUDA on and Vulkan off
export IREE_CUDA_DISABLE=0
export IREE_VULKAN_DISABLE=1
./build_tools/cmake/ctest_all.sh ../iree-build
```

### Running samples

``` shell
# Build
cmake --build ../iree-build/

# Run a standalone sample application
../iree-build/runtime/src/iree/runtime/demo/hello_world_embedded
# 4xf32=1 1.1 1.2 1.3
#  *
# 4xf32=10 100 1000 10000
#  =
# 4xf32=10 110 1200 13000

# Try out the developer tools
ls ../iree-build/tools/
../iree-build/tools/iree-compile --help
../iree-build/tools/iree-run-module --help
```

## :simple-python: Python bindings

Python packages can either be built from source or installed from our releases.
See the [Python bindings](../reference/bindings/python.md) page for details
about the bindings themselves.

### Dependencies

You will need a recent Python installation >=3.9 (we aim to support
[non-eol Python versions](https://endoflife.date/python)).

???+ Tip "Tip - Managing Python versions"
    Make sure your 'python' is what you expect:

    === ":fontawesome-brands-linux: Linux"

        Note that on multi-python systems, this may have a version suffix, and on
        many Linuxes where python2 and python3 can co-exist, you may also want to
        use `python3`.

        ``` shell
        which python
        python --version
        ```

    === ":fontawesome-brands-apple: macOS"

        Note that on multi-python systems, this may have a version suffix, and on
        macOS where python2 and python3 can co-exist, you may also want to use `python3`.

        ``` shell
        which python
        python --version
        ```

    === ":fontawesome-brands-windows: Windows"
        The
        [Python launcher for Windows](https://docs.python.org/3/using/windows.html#python-launcher-for-windows) (`py`) can help manage versions.

        ``` powershell
        which python
        python --version
        py --list-paths
        ```

???+ Tip "Tip - Virtual environments"
    We recommend using virtual environments to manage python packages, such as
    through `venv`
    ([about](https://docs.python.org/3/library/venv.html),
    [tutorial](https://docs.python.org/3/tutorial/venv.html)):

    === ":fontawesome-brands-linux: Linux"

        ``` shell
        python -m venv .venv
        source .venv/bin/activate
        ```

    === ":fontawesome-brands-apple: macOS"

        ``` shell
        python -m venv .venv
        source .venv/bin/activate
        ```

    === ":fontawesome-brands-windows: Windows"

        ``` powershell
        python -m venv .venv
        .venv\Scripts\activate.bat
        ```

    When done, run `deactivate`.

``` shell
# Upgrade PIP before installing other requirements
python -m pip install --upgrade pip

# Install IREE build requirements
python -m pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt
```

### Building with CMake

To build the Python bindings, configure CMake with the
`IREE_BUILD_PYTHON_BINDINGS` option. We also recommend explicitly setting which
Python executable to use with `Python3_EXECUTABLE`:

``` shell
# Configure (including other options as discussed above)
cmake -G Ninja -B ../iree-build/ \
  -DIREE_BUILD_PYTHON_BINDINGS=ON  \
  -DPython3_EXECUTABLE="$(which python)" \
  .

# Build
cmake --build ../iree-build/
```

### Using the Python bindings

Extend your `PYTHONPATH` with IREE's `bindings/python` paths and try importing:

=== ":fontawesome-brands-linux: Linux"

    ``` shell
    source ../iree-build/.env && export PYTHONPATH
    python -c "import iree.compiler"
    python -c "import iree.runtime"
    ```

=== ":fontawesome-brands-apple: macOS"

    ``` shell
    source ../iree-build/.env && export PYTHONPATH
    python -c "import iree.compiler"
    python -c "import iree.runtime"
    ```

=== ":fontawesome-brands-windows: Windows"

    ``` powershell
    ../iree-build/.env.bat
    python -c "import iree.compiler"
    python -c "import iree.runtime"
    ```

Using IREE's ML framework importers requires a few extra steps:

``` shell
# Install test requirements
python -m pip install -r integrations/tensorflow/test/requirements.txt

# Install pure Python packages (no build required)
python -m pip install integrations/tensorflow/python_projects/iree_tf
python -m pip install integrations/tensorflow/python_projects/iree_tflite

# Then test the tools:
iree-import-tf --help
iree-import-tflite --help
```
