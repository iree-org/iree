---
icon: octicons/sliders-16
---

# Building with Bazel

This page walks through building IREE from source using the
[Bazel build system](https://bazel.build/).

!!! warning

    Bazel build support is primarily for internal project infrastructure. We
    strongly recommend [using CMake](../../building-from-source/index.md)
    instead.

    Our Bazel configuration is also _only_ tested on Linux. Windows and macOS
    may be unstable.

## :octicons-download-16: Prerequisites

=== ":fontawesome-brands-linux: Linux"

    1. Install Bazel, matching IREE's
        [`.bazelversion`](https://github.com/openxla/iree/blob/main/.bazelversion)
        by following the
        [official docs](https://bazel.build/install).

    2. Install a compiler such as Clang (GCC is not fully supported).

        ```shell
        sudo apt install clang
        ```

        Set environment variables for Bazel:

        ```shell
        export CC=clang
        export CXX=clang++
        ```

    3. Install Python build requirements:

        ```shell
        python -m pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt
        ```

=== ":fontawesome-brands-apple: macOS"

    1. Install [Homebrew](https://brew.sh/):

        ```shell
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
        ```

    2. Install Bazel, matching IREE's
        [`.bazelversion`](https://github.com/openxla/iree/blob/main/.bazelversion)
        by following the [official docs](https://bazel.build/install/os-x) or
        via Homebrew:

        ```shell
        brew install bazel
        ```

    3. Install Python build requirements:

        ```shell
        python -m pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt
        ```

=== ":fontawesome-brands-windows: Windows"

    !!! tip

        You can simplify installation by using a package manager like
        [Scoop](https://scoop.sh/) or [Chocolatey](https://chocolatey.org/).

    1. Install Bazel, matching IREE's
        [`.bazelversion`](https://github.com/openxla/iree/blob/main/.bazelversion)
        by following the [official docs](https://bazel.build/install/windows).

        Also install [MSYS2](https://www.msys2.org/) by following Bazel's documentation.

    2. Install Python3 ([docs here](https://www.python.org/downloads/windows/))
        and Python build requirements:

        ```shell
        python -m pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt
        ```

    3. Install the full Visual Studio or "Build Tools For Visual Studio" from the
        [downloads page](https://visualstudio.microsoft.com/downloads/) then
        set the `BAZEL_VS` environment variable:

        ```powershell
        > $env:BAZEL_VS = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
        ```

## :octicons-rocket-16: Quickstart: clone and build

### Clone

Use [Git](https://git-scm.com/) to clone the IREE repository and initialize its
submodules:

```shell
git clone https://github.com/openxla/iree.git
cd iree
git submodule update --init
```

Configure Bazel:

```shell
# This generates a `configured.bazelrc` file by analyzing your environment.
# Skipping this step will make it difficult to select your platform/compiler.
python3 configure_bazel.py
```

=== ":fontawesome-brands-linux: Linux"

    (No Linux-specific tips for configuring)

=== ":fontawesome-brands-apple: macOS"

    (No macOS-specific tips for configuring)

=== ":fontawesome-brands-windows: Windows"

    !!! tip

        Clone to a short path like `C:\projects\` to avoid issues with Windows
        maximum path lengths (260 characters).

    !!! tip

        `configure_bazel.py` only detects that you have Windows and will output
        the default `--config=windows` to `configured.bazelrc`, which assumes
        the latest version of MSVC. To avoid some warnings, you may want to
        replace it with (for example) `--config=msvc2022`.

### Build

Run all tests:

```shell
bazel test -k //...
```

Run all tests _except_ those that require CUDA:

```shell
bazel test -k //... \
    --iree_drivers=local-sync,local-task,vulkan \
    --test_tag_filters="-driver=cuda,-target=cuda" \
    --build_tag_filters="-driver=cuda,-target=cuda"
```

Run all tests _except_ those that require a GPU (any API):

```shell
bazel test -k //... \
    --iree_drivers=local-sync,local-task,vulkan \
    --test_tag_filters="-driver=vulkan,-driver=metal,-driver=cuda,-target=cuda" \
    --build_tag_filters="-driver=cuda,-target=cuda"
```

!!! tip

    See the
    [`build_tools/bazel/build_core.sh`](https://github.com/openxla/iree/blob/main/build_tools/bazel/build_core.sh)
    script for examples of other flags and environment variables that can be
    used to configure what Bazel runs.

In general, build artifacts will be under the `bazel-bin` directory at the top
level.

## :octicons-gear-16: Recommended `user.bazelrc`

You can put a user.bazelrc at the root of the repository and it will be ignored
by git.

=== ":fontawesome-brands-linux: Linux"

    ```shell
    build --disk_cache=/tmp/bazel-cache

    # Use --config=debug to compile IREE and LLVM without optimizations
    # and with assertions enabled.
    build:debug --config=asserts --compilation_mode=opt '--per_file_copt=iree|llvm@-O0' --strip=never

    # Use --config=asserts to enable assertions. This has to be done globally:
    # Code compiled with and without assertions can't be linked together (ODR violation).
    build:asserts --compilation_mode=opt '--copt=-UNDEBUG'
    ```

=== ":fontawesome-brands-apple: macOS"

    ```shell
    build --disk_cache=/tmp/bazel-cache

    # Use --config=debug to compile IREE and LLVM without optimizations
    # and with assertions enabled.
    build:debug --config=asserts --compilation_mode=opt '--per_file_copt=iree|llvm@-O0' --strip=never

    # Use --config=asserts to enable assertions. This has to be done globally:
    # Code compiled with and without assertions can't be linked together (ODR violation).
    build:asserts --compilation_mode=opt '--copt=-UNDEBUG'
    ```

=== ":fontawesome-brands-windows: Windows"

    ```shell
    build --disk_cache=c:/bazelcache
    build:debug --compilation_mode=dbg --copt=/O2 --per_file_copt=iree@/Od --strip=never
    ```

## What's next?

### Take a Look Around

Build all of IREE's 'tools' directory:

```shell
bazel build tools/...
```

Check out what was built:

```shell
ls bazel-bin/tools/
./bazel-bin/tools/iree-compile --help
```

Translate a
[MLIR file](https://github.com/openxla/iree/blob/main/samples/models/simple_abs.mlir)
and execute a function in the compiled module:

```shell
# iree-run-mlir <compiler flags> [input.mlir] <runtime flags>
$ ./bazel-bin/tools/iree-run-mlir \
  --iree-hal-target-backends=vmvx --print-mlir \
  ./samples/models/simple_abs.mlir \
  --input=f32=-2
```
