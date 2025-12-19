# Using IREE with a Custom LLVM via Bzlmod

This document explains how projects that depend on IREE can provide their own LLVM
instead of using IREE's bundled version.

## Terminology

### Bzlmod
Bazel's module system (introduced in Bazel 6.0, default in Bazel 7.0+). It replaces
the legacy WORKSPACE file with `MODULE.bazel` for managing external dependencies.

### Root Module
The top-level project being built. In bzlmod, only the root module's `MODULE.bazel`
is fully evaluated - dependency modules have limited control over the build graph.

### Module Extension
A mechanism for creating repositories dynamically in bzlmod. Extensions are defined
in `.bzl` files and invoked via `use_extension()` in MODULE.bazel.

### `use_extension()`
Runs a module extension's implementation function, which typically creates repositories.
Returns an extension proxy that can be passed to `use_repo()`.

### `use_repo()`
Imports repositories created by a module extension into the current module's visibility
scope. Without `use_repo()`, repos created by an extension exist but aren't accessible
to your BUILD files.

```python
# Extension creates repos internally
ext = use_extension("@some_module//:extensions.bzl", "some_extension")

# use_repo makes specific repos visible as @repo_a, @repo_b, etc.
use_repo(ext, "repo_a", "repo_b")
```

### `use_repo_rule()`
Imports a repository rule from another module so it can be called directly in
MODULE.bazel to create a repository.

### `llvm-raw`
A repository containing the raw LLVM source code. This is the input to the LLVM
build configuration.

### `llvm-project`
The configured LLVM repository created by `llvm_configure`. It overlays Bazel BUILD
files onto the `llvm-raw` source and extracts CMake configuration variables.

### `llvm-project-overlay`
The bzlmod module name for LLVM's Bazel integration (located at
`llvm-project/utils/bazel/`). It provides the `llvm_repos_extension` and
`llvm_configure` rule.

## How It Works

IREE's module extension (`iree_extension`) creates the `llvm-raw` repository
**only when IREE is the root module**:

```python
# In build_tools/bazel/extensions.bzl
def _iree_extension_impl(module_ctx):
    if any([m.is_root and m.name == "iree_core" for m in module_ctx.modules]):
        new_local_repository(
            name = "llvm-raw",
            build_file_content = "# empty",
            path = "third_party/llvm-project",
        )
    # ... other repos
```

When your project depends on IREE, IREE is **not** the root module - your project is.
Therefore, IREE's extension will not create `llvm-raw`, and you must provide it yourself.

## MODULE.bazel Ordering

The order of statements in MODULE.bazel matters:

1. `module()` - must be first
2. `bazel_dep()` - declare module dependencies
3. `local_path_override()` - must come after the `bazel_dep()` it overrides
4. `use_extension()` - must come after the `bazel_dep()` that provides the extension
5. `use_repo()` - must come after its corresponding `use_extension()`
6. `use_repo_rule()` + invocation - can reference repos created by earlier extensions

## Example: Using Your Own LLVM

```python
# my_project/MODULE.bazel

module(
    name = "my_project",
    version = "1.0.0",
)

# Standard bazel dependencies (must match or be compatible with IREE's versions)
bazel_dep(name = "bazel_skylib", version = "1.8.2")
bazel_dep(name = "platforms", version = "1.0.0")
bazel_dep(name = "rules_cc", version = "0.2.11")
# ... other deps as needed

# Depend on IREE
bazel_dep(name = "iree_core", version = "0.0.1")

# Override IREE to use your local checkout (optional, for development)
local_path_override(
    module_name = "iree_core",
    path = "third_party/iree",
)

# Depend on LLVM overlay module
bazel_dep(name = "llvm-project-overlay", version = "main")
local_path_override(
    module_name = "llvm-project-overlay",
    path = "my/custom/llvm-project/utils/bazel",
)

# Create your own llvm-raw repository pointing to your LLVM
new_local_repository = use_repo_rule(
    "@bazel_tools//tools/build_defs/repo:local.bzl",
    "new_local_repository",
)
new_local_repository(
    name = "llvm-raw",
    path = "my/custom/llvm-project",
    build_file_content = "# empty",
)

# Use LLVM's extension for third-party deps (gmp, mpfr, etc.)
llvm_repos_ext = use_extension(
    "@llvm-project-overlay//:extensions.bzl",
    "llvm_repos_extension",
)
use_repo(
    llvm_repos_ext,
    "gmp",
    "mpc",
    "mpfr",
    "nanobind",
    "pfm",
    "pybind11",
    "vulkan_sdk",
)

# Use IREE's extension (won't create llvm-raw since you're the root module)
iree_ext = use_extension(
    "@iree_core//build_tools/bazel:extensions.bzl",
    "iree_extension",
)
use_repo(
    iree_ext,
    "com_github_dvidelabs_flatcc",
    "com_google_benchmark",
    "com_google_googletest",
    "stablehlo",
    # ... other IREE repos you need
)

# Configure LLVM (creates llvm-project from your llvm-raw)
llvm_configure = use_repo_rule(
    "@llvm-raw//utils/bazel:configure.bzl",
    "llvm_configure",
)
llvm_configure(name = "llvm-project")
```

## Using LLVM from an HTTP Archive

If you want to fetch LLVM from a release tarball instead of a local path:

```python
# my_project/MODULE.bazel

http_archive = use_repo_rule(
    "@bazel_tools//tools/build_defs/repo:http.bzl",
    "http_archive",
)

LLVM_COMMIT = "abc123..."  # Your desired commit
LLVM_SHA256 = "..."        # SHA256 of the tarball

http_archive(
    name = "llvm-raw",
    build_file_content = "# empty",
    sha256 = LLVM_SHA256,
    strip_prefix = "llvm-project-" + LLVM_COMMIT,
    urls = ["https://github.com/llvm/llvm-project/archive/{}.tar.gz".format(LLVM_COMMIT)],
)
```

## Version Compatibility

When providing your own LLVM, ensure compatibility with IREE:

1. **LLVM Version**: IREE targets a specific LLVM commit. Check IREE's
   `third_party/llvm-project` submodule for the expected version.

2. **Bazel Dependencies**: Your LLVM's `utils/bazel/MODULE.bazel` declares
   dependency versions. These should be compatible with IREE's dependencies.

3. **API Compatibility**: LLVM APIs change between versions. Your LLVM must
   be API-compatible with what IREE expects.

## Troubleshooting

### "repository 'llvm-raw' is not defined"
You haven't created the `llvm-raw` repository. As the root module, you must
define it yourself (see examples above).

### Build errors in LLVM code
Your LLVM version may be incompatible with IREE. Check that your LLVM commit
is close to IREE's expected version.

### Duplicate repository errors
Multiple modules may be trying to create the same repository. Ensure only
one source defines each repository name.
