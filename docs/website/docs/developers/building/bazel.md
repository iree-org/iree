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

    1. **Recommended:** Install [bazelisk](https://github.com/bazelbuild/bazelisk)
        to automatically use the correct Bazel version from
        [`.bazelversion`](https://github.com/iree-org/iree/blob/main/.bazelversion):

        ```shell
        # Via Go (if installed)
        go install github.com/bazelbuild/bazelisk@latest

        # Or download binary from GitHub releases
        ```

        **Alternative:** Install Bazel 7.3.1 manually by following the
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

    2. **Recommended:** Install [bazelisk](https://github.com/bazelbuild/bazelisk)
        to automatically use the correct Bazel version from
        [`.bazelversion`](https://github.com/iree-org/iree/blob/main/.bazelversion):

        ```shell
        brew install bazelisk
        ```

        **Alternative:** Install Bazel 7.3.1 manually by following the
        [official docs](https://bazel.build/install/os-x) or via Homebrew:

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

    1. **Recommended:** Install [bazelisk](https://github.com/bazelbuild/bazelisk)
        to automatically use the correct Bazel version from
        [`.bazelversion`](https://github.com/iree-org/iree/blob/main/.bazelversion):

        ```powershell
        # Via Scoop
        scoop install bazelisk

        # Via Chocolatey
        choco install bazelisk

        # Or download binary from GitHub releases
        ```

        **Alternative:** Install Bazel 7.3.1 manually by following the
        [official docs](https://bazel.build/install/windows).

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

**TL;DR** - Copy and paste this to get started:

```shell
# Clone and setup
git clone https://github.com/iree-org/iree.git
cd iree
git submodule update --init
python -m pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt

# Add tools to PATH (from repo root)
export PATH="$PWD/build_tools/bin:$PATH"

# Configure once
iree-bazel-configure

# Build and test
iree-bazel-build //tools:iree-compile
iree-bazel-test //...
```

!!! tip "Building with CUDA?"

    Enable GPU drivers before configure:

    ```shell
    IREE_HAL_DRIVER_CUDA=ON iree-bazel-configure
    ```

    Requires CUDA toolkit installed separately.

!!! note "Driver configuration is sticky"

    Re-run `iree-bazel-configure` whenever you change `IREE_HAL_DRIVER_*` settings.

!!! tip "Working with Claude Code?"

    All `iree-bazel-*` tools provide machine-readable documentation via `--agent_md`:

    ```shell
    # Generate documentation for Claude
    iree-bazel-configure --agent_md >> CLAUDE.local.md
    ```

    This appends concise tool documentation to your local Claude instructions,
    teaching Claude how to use the `iree-bazel-*` tools in your project.

### Detailed steps

1. **Clone the repository:**

    ```shell
    git clone https://github.com/iree-org/iree.git
    cd iree
    git submodule update --init
    ```

    !!! tip "Windows: Short paths"

        Clone to a short path like `C:\projects\` to avoid issues with Windows
        maximum path lengths (260 characters).

2. **Install Python dependencies:**

    ```shell
    python -m pip install -r runtime/bindings/python/iree/runtime/build_requirements.txt
    ```

3. **Add wrapper tools to PATH:**

    ```shell
    export PATH="$PWD/build_tools/bin:$PATH"
    ```

    The `iree-bazel-*` wrapper tools simplify common workflows and provide sensible
    defaults for local development. Run from the IREE repository root.

4. **Configure once per worktree:**

    ```shell
    iree-bazel-configure
    ```

    This detects your platform/compiler and creates configuration files
    (`configured.bazelrc` and `user.bazelrc`). Re-run this command to change
    driver settings or after major environment changes.

5. **Build and test:**

    ```shell
    # Build specific targets
    iree-bazel-build //tools:iree-compile

    # Run all tests
    iree-bazel-test //...

    # Run tests in a specific directory
    iree-bazel-test //runtime/...
    ```

Build artifacts are under the `bazel-bin/` directory, test logs under `bazel-testlogs/`.

## :octicons-gear-16: Local development configuration

For faster local development iteration, use the `--config=localdev` flag:

```shell
bazel build --config=localdev //tools/...
bazel test --config=localdev //runtime/...
```

!!! note

    The `iree-bazel-*` wrapper tools use `--config=localdev` by default, so you
    don't need to specify it manually.

This enables several optimizations:

- **Skymeld** (merged analysis/execution phases for faster multi-target builds)
- **Ramdisk sandbox** on Linux (reduces I/O bottleneck with many CPU cores)

The `localdev` config is defined in
[`build_tools/bazel/iree.bazelrc`](https://github.com/iree-org/iree/blob/main/build_tools/bazel/iree.bazelrc)
and works on all platforms.

## :octicons-pencil-16: Optional `user.bazelrc`

!!! note

    `iree-bazel-configure` creates `user.bazelrc` automatically with platform-specific
    defaults (including disk cache location). You can customize it further as shown below.

You can put a `user.bazelrc` at the root of the repository for personal
customizations (ignored by git). For example:

=== ":fontawesome-brands-linux: Linux"

    ```shell
    # Always use localdev optimizations
    build --config=localdev

    # Use --config=debug to compile IREE and LLVM without optimizations
    # and with assertions enabled.
    build:debug --config=asserts --compilation_mode=opt '--per_file_copt=iree|llvm@-O0' --strip=never

    # Use --config=asserts to enable assertions. This has to be done globally:
    # Code compiled with and without assertions can't be linked together (ODR violation).
    build:asserts --compilation_mode=opt '--copt=-UNDEBUG'
    ```

=== ":fontawesome-brands-apple: macOS"

    ```shell
    # Always use localdev optimizations
    build --config=localdev

    # Use --config=debug to compile IREE and LLVM without optimizations
    # and with assertions enabled.
    build:debug --config=asserts --compilation_mode=opt '--per_file_copt=iree|llvm@-O0' --strip=never

    # Use --config=asserts to enable assertions. This has to be done globally:
    # Code compiled with and without assertions can't be linked together (ODR violation).
    build:asserts --compilation_mode=opt '--copt=-UNDEBUG'
    ```

=== ":fontawesome-brands-windows: Windows"

    ```shell
    # Always use localdev optimizations
    build --config=localdev

    # Debug config for Windows
    build:debug --compilation_mode=dbg --copt=/O2 --per_file_copt=iree@/Od --strip=never
    ```

## :octicons-tools-16: Bazel Wrapper Tools

IREE provides wrapper scripts in `build_tools/bin/` that simplify common Bazel workflows:

**Setup:**

```shell
# Add to your PATH (in ~/.bashrc or similar)
export PATH="$PATH:/path/to/iree/build_tools/bin"
```

**Available tools:**

- `iree-bazel-configure` - Initial Bazel setup (run once per worktree)
- `iree-bazel-build [target]` - Build with optimized defaults
    (supports `-w` watch mode)
- `iree-bazel-test [target]` - Run tests with configured drivers
    (supports `-w` watch mode)
- `iree-bazel-run <target> [-- args]` - Build and execute from current directory
    (supports `-w` watch mode)
- `iree-bazel-query <expr>` - Query the build graph
- `iree-bazel-try <file>` - Compile/run C++ snippets without BUILD files
- `iree-bazel-fuzz <target>` - Run fuzzer with corpus management

**Examples:**

```shell
# Configure with CUDA support
IREE_HAL_DRIVER_CUDA=ON iree-bazel-configure

# Build compiler tools
iree-bazel-build //tools:iree-compile

# Run all tests (using configured drivers)
iree-bazel-test //...

# Reconfigure to enable multiple drivers
IREE_HAL_DRIVER_CUDA=ON IREE_HAL_DRIVER_VULKAN=ON iree-bazel-configure

# Run tool from current directory
iree-bazel-run //tools:iree-compile -- input.mlir -o output.vmfb

# Quick C++ experiment
iree-bazel-try -e '#include "iree/base/api.h"
int main() { return iree_status_is_ok(iree_ok_status()) ? 0 : 1; }'
```

**Watch mode:**

Add `-w` or `--watch` to automatically rebuild/retest on file changes:

```shell
# Rebuild on changes
iree-bazel-build -w //tools:iree-compile

# Rerun tests on changes (great for TDD)
iree-bazel-test -w //runtime/src/iree/base:status_test

# Rebuild and restart binary on changes
iree-bazel-run -w //tools:iree-opt -- input.mlir
```

Watch mode uses [ibazel](https://github.com/bazelbuild/bazel-watcher) which must
be installed separately:

```shell
go install github.com/bazelbuild/bazel-watcher/cmd/ibazel@latest
```

!!! note "Bazel server lock behavior"

    Watch mode holds the Bazel server lock while the binary runs (necessary for
    ibazel to control process lifecycle). Normal mode releases the lock after
    building, allowing parallel builds in other shells.

**Environment variables:**

- `IREE_BAZEL_CACHE_DIR`: Bazel disk cache location
    (default: `~/.cache/bazel-iree`)
- `IREE_HAL_DRIVER_*`: Enable/disable HAL drivers
    (configure-time, matches CMake)
    - `IREE_HAL_DRIVER_CUDA=ON`: Enable CUDA driver
    - `IREE_HAL_DRIVER_VULKAN=ON`: Enable Vulkan driver
    - Values: ON/YES/TRUE/Y/1 or OFF/NO/FALSE/N/0
- `IREE_FUZZ_CACHE`: Fuzzer corpus cache
    (default: `~/.cache/iree-fuzz-cache`)
- `IREE_FUZZ_CORPUS`: Fuzzer dictionaries
    (default: `~/.cache/iree-fuzz-corpus`)

**Tips:**

- Add `-v` flag to see the exact Bazel command being executed
- Wrappers use `--config=localdev` by default for faster local iteration
- Run `<tool> --help` for detailed usage information

## :octicons-book-16: Advanced: Under the Hood

### Wrapper → Raw Bazel mapping

The wrapper tools are thin shells around standard Bazel commands. Here's what
they do:

| Wrapper | Equivalent Raw Command |
|---------|------------------------|
| `iree-bazel-build //foo` | `bazel build --config=localdev //foo` |
| `iree-bazel-build -w //foo` | `ibazel build --config=localdev //foo` |
| `iree-bazel-test //...` | `bazel test --config=localdev //...` |
| `iree-bazel-test -w //...` | `ibazel test --config=localdev //...` |
| `iree-bazel-run //foo` | Build + exec (releases lock) |
| `iree-bazel-run -w //foo` | `ibazel run --run_under="cd $PWD && " //foo` |
| `iree-bazel-configure` | `python3 configure_bazel.py` + create `user.bazelrc` |

**Debugging wrappers:** Use `-v` or `--verbose` flag to print the exact Bazel
command being executed.

**For CI/scripts:** Raw `bazel` commands work fine. Wrappers are optional
helpers for local development.

### Configuration files

- `configured.bazelrc` - Written by `configure_bazel.py` (platform detection,
  driver settings)
- `user.bazelrc` - Created by `iree-bazel-configure` (disk cache location,
  custom configs)
- `build_tools/bazel/iree.bazelrc` - Main IREE configuration (localdev, GPU,
  etc.)

All three files are loaded automatically by Bazel in this order.

## What's next?

### Take a Look Around

Build all of IREE's 'tools' directory:

```shell
bazel build //tools/...
```

Check out what was built:

```shell
ls bazel-bin/tools/
./bazel-bin/tools/iree-compile --help
```

Translate a
[MLIR file](https://github.com/iree-org/iree/blob/main/samples/models/simple_abs.mlir)
and execute a function in the compiled module:

```shell
# iree-run-mlir <compiler flags> [input.mlir] <runtime flags>
$ ./bazel-bin/tools/iree-run-mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  --print-mlir \
  ./samples/models/simple_abs.mlir \
  --input=f32=-2
```
