# IREE Developer Tools

This directory contains command-line tools that developers add to their PATH for
convenient access across all IREE worktrees. Tools here provide streamlined
workflows for building, testing, and experimenting with IREE.

**Naming convention**: All tools must be prefixed with `iree-` to avoid PATH
conflicts. Do not use names that conflict with existing IREE tools (e.g., don't
name something `iree-run-module` or `iree-compile`).

## Setup

Add this directory to your PATH:

```bash
export PATH="$PATH:$PWD/build_tools/bin"
```

Or add to your shell configuration for persistence.

---

## Bazel Tools

Wrapper tools for building IREE with Bazel. These handle configuration, provide
better defaults, and work from any subdirectory within a worktree.

For complete Bazel build documentation, see:
[Building with Bazel](../../docs/website/docs/developers/building/bazel.md)

### Quick Start

```bash
# Configure once per worktree
iree-bazel-configure

# Build
iree-bazel-build //tools:iree-compile

# Test
iree-bazel-test //runtime/src/iree/base:status_test

# Run
iree-bazel-run //tools:iree-compile -- --help
```

> **Tip: Working with Claude Code?**
>
> All `iree-bazel-*` tools provide machine-readable documentation via `--agent_md`:
>
> ```shell
> # Generate documentation for Claude
> iree-bazel-configure --agent_md >> CLAUDE.local.md
> ```
>
> This appends concise tool documentation to your local Claude instructions,
> teaching Claude how to use the `iree-bazel-*` tools in your project.

### Tools

| Tool | Description |
|------|-------------|
| `iree-bazel-configure` | Configure IREE for Bazel builds (run once per worktree) |
| `iree-bazel-build` | Build targets |
| `iree-bazel-test` | Run tests |
| `iree-bazel-run` | Build and run executables from current directory |
| `iree-bazel-query` | Query the build graph |
| `iree-bazel-cquery` | Configuration-aware query (resolved select(), actual targets) |
| `iree-bazel-try` | Compile and run C/C++ snippets without BUILD files |
| `iree-bazel-fuzz` | Run libFuzzer targets with persistent corpus |
| `iree-bazel-lib` | Shared library (sourced by other tools) |

### Common Options

All tools support:
- `-h, --help` - Show help
- `-n, --dry_run` - Show command without executing
- `-v, --verbose` - Verbose output

Short flags can be combined: `-nv` is equivalent to `-n -v`.

### Configuration

Driver configuration happens during `iree-bazel-configure`:

```bash
# Enable CUDA driver
IREE_HAL_DRIVER_CUDA=ON iree-bazel-configure

# Enable multiple drivers
IREE_HAL_DRIVER_CUDA=ON IREE_HAL_DRIVER_VULKAN=ON iree-bazel-configure
```

### Output Directories

Built artifacts are placed in standard Bazel output directories at the repo root:
- `bazel-bin/` - Built executables and libraries
- `bazel-testlogs/` - Test outputs and logs
- `bazel-out/` - Full build tree

### Watch Mode

Several tools support watch mode (`-w`) which rebuilds/reruns on file changes:

```bash
iree-bazel-build -w //tools:iree-opt     # Rebuild on changes
iree-bazel-test -w //runtime/...:test    # Rerun tests on changes
iree-bazel-run -w //tools:iree-opt -- input.mlir  # Restart on changes
```

Watch mode requires [ibazel](https://github.com/bazelbuild/bazel-watcher).

### Quick Experiments with iree-bazel-try

Compile and run C/C++ snippets against IREE without writing BUILD files:

```bash
# Quick API exploration
iree-bazel-try -e '
#include "iree/base/api.h"
#include <stdio.h>
int main() {
  printf("ok: %d\n", iree_status_is_ok(iree_ok_status()));
  return 0;
}'

# Run tests
iree-bazel-try -e '
#include "iree/testing/gtest_harness.h"
TEST(Status, Ok) { IREE_EXPECT_OK(iree_ok_status()); }
'

# MLIR transforms
echo 'func.func @f() { return }' | iree-bazel-try -e '
#include "iree/compiler/Tools/MlirTransformHarness.h"
void xform(ModuleOp m) { m.walk([](Operation *op) { llvm::outs() << op->getName() << "\n"; }); }
MLIR_TRANSFORM_MAIN_NO_PRINT(xform)
'
```

Run `iree-bazel-try --help` for full documentation.
