# Getting started

<!-- TODO(scotttodd): Introduction, when to build from source -->

## Prerequisites

You will need to install [CMake](https://cmake.org/), along with a C/C++
compiler:

=== "Linux"

    <!-- TODO(scotttodd): annotation about gcc vs clang -->

    ``` shell
    sudo apt install cmake clang
    export CC=clang
    export CXX=clang++
    ```

=== "Windows"

    1. Install CMake from the
       [official downloads page](https://cmake.org/download/)

    2. Install MSVC from Visual Studio or "Tools for Visual Studio" on the
       [official downloads page](https://visualstudio.microsoft.com/downloads/)

    !!! note
        You will need to initialize MSVC by running `vcvarsall.bat` to use it
        from the command line. See the
        [official documentation](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line)
        for details.

## Clone and build

Use [Git](https://git-scm.com/) to clone the IREE repository and initialize its
submodules:

``` shell
git clone https://github.com/google/iree.git
cd iree
git submodule update --init
```

Configure then build all targets using CMake:

``` shell
cmake -B ../iree-build/ .
cmake --build ../iree-build/
```

???+ Tip
    Most IREE Core devs use [Ninja](https://ninja-build.org/) as the CMake
    generator. The benefit is that it works the same across all platforms and
    automatically takes advantage of parallelism. to use it, add a `-GNinja`
    argument to your initial cmake command (and make sure to install
    `ninja-build` from either your favorite OS package manager, or generically
    via `python -m pip install ninja`).


## What's next?

<!-- TODO(scotttodd): "at this point you can..." -->

### Running tests

Run all built tests through
[CTest](https://gitlab.kitware.com/cmake/community/-/wikis/doc/ctest/Testing-With-CTest):

``` shell
cd ../iree-build/
ctest --output-on-failure
```

### Take a look around

Check out the contents of the 'tools' build directory:

``` shell
ls ../iree-build/iree/tools/
../iree-build/iree/tools/iree-translate --help
```

<!-- TODO(scotttodd): troubleshooting section? link to github issues? -->
