# Using IREE with Custom MLIR-Adjacent Dependencies via Bzlmod

This document explains how projects that depend on IREE can provide their own
LLVM, StableHLO, torch-mlir, and compiler plugin registry instead of using
IREE's bundled defaults.

## Terminology

### Bzlmod
Bazel's module system (introduced in Bazel 6.0, default in Bazel 7.0+). It replaces
the legacy WORKSPACE file with `MODULE.bazel` for managing external dependencies.

### Root Module
The top-level project being built. Its `MODULE.bazel` controls dependency overrides
and can inject repositories into extensions used by dependency modules.

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

### Raw source repositories
Repositories such as `llvm-raw` and `torch-mlir-raw` contain unconfigured
upstream source trees. They are inputs to repository rules that overlay Bazel
BUILD files and produce configured repositories such as `llvm-project` and
`torch-mlir`.

### `llvm-project`
The configured LLVM repository created by `llvm_configure`. It overlays Bazel BUILD
files onto the `llvm-raw` source and extracts CMake configuration variables.

### `llvm-project-overlay`
The bzlmod module name for LLVM's Bazel integration (located at
`llvm-project/utils/bazel/`). It provides the `llvm_repos_extension` and
`llvm_configure` rule.

## How It Works

IREE's module extension (`iree_extension`) creates MLIR-adjacent source
repositories **only when IREE is the root module**:

```python
# In build_tools/bazel/extensions.bzl
def _iree_extension_impl(module_ctx):
    iree_root = str(module_ctx.path(Label("//:MODULE.bazel")).dirname)
    if any([m.is_root and m.name == "iree_core" for m in module_ctx.modules]):
        new_local_repository(
            name = "llvm-raw",
            build_file_content = "# empty",
            path = iree_root + "/third_party/llvm-project",
        )
        local_repository(
            name = "stablehlo",
            path = iree_root + "/third_party/stablehlo",
        )
        new_local_repository(
            name = "torch-mlir-raw",
            build_file_content = "# empty - BUILD files overlaid by torch_mlir_configure",
            path = iree_root + "/third_party/torch-mlir",
        )
    # ... other repos
```

When your project depends on IREE, IREE is **not** the root module - your project is.
Therefore, IREE's extension will not create these MLIR-adjacent repositories,
and you must provide the ones needed by the compiler plugins you enable.

Creating a repository in your root module does not automatically make it visible
to IREE. Use `inject_repo()` to supply the raw source repositories to IREE's
extension and `llvm-raw` to LLVM's extension, as shown below.

## Label Resolution in Macros

In legacy macros, target references resolve according to how they are written:

- `Label("...")` resolves using the package and repository mapping of the `.bzl`
  file containing the `Label()` call. The reference retains that meaning when
  the macro is called from another repository.
- A raw label string passed to a rule's label attribute resolves using the
  package and repository mapping of the `BUILD` file calling the macro.

For raw strings, declare or import the repository under the expected name in
the caller's `MODULE.bazel`. This does not require `override_repo()`.

To replace a repository provided by a module extension, use `override_repo()`
in the root module. Given an extension proxy `ext` from `use_extension()` and
an existing root-visible repository `my_dependency`:

```python
override_repo(ext, dependency = "my_dependency")
```

This replaces the extension's `dependency` repository, including references
resolved to it through definition-site `Label()` calls.

See Bazel's [label resolution in macros](https://bazel.build/extending/legacy-macros#label-resolution-in-macros)
and [repository overrides](https://bazel.build/external/extension#overriding-and-injecting-module-extension-repos)
for details.

## MODULE.bazel Ordering

The order of statements in MODULE.bazel matters:

1. `module()` - must be first
2. `bazel_dep()` - declare module dependencies
3. `local_path_override()` - must come after the `bazel_dep()` it overrides
4. `use_extension()` - must come after the `bazel_dep()` that provides the extension
5. `use_repo()` - must come after its corresponding `use_extension()`
6. `use_repo_rule()` + invocation - can reference repos created by earlier extensions

## Example: Using Your Own LLVM, StableHLO, and torch-mlir

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
bazel_dep(name = "rules_python", version = "1.9.0")
bazel_dep(name = "rules_shell", version = "0.6.1")
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

# Create your own raw repositories pointing to the upstream projects you want.
new_local_repository = use_repo_rule(
    "@bazel_tools//tools/build_defs/repo:local.bzl",
    "new_local_repository",
)
new_local_repository(
    name = "llvm-raw",
    path = "my/custom/llvm-project",
    build_file_content = "# empty",
)
new_local_repository(
    name = "torch-mlir-raw",
    path = "my/custom/torch-mlir",
    build_file_content = "# empty",
)
local_repository = use_repo_rule(
    "@bazel_tools//tools/build_defs/repo:local.bzl",
    "local_repository",
)
local_repository(
    name = "stablehlo",
    path = "my/custom/stablehlo",
)

# Use LLVM's extension for its generated repositories. Other dependencies,
# such as gmp and mpfr, are declared by the LLVM overlay's MODULE.bazel.
llvm_repos_ext = use_extension(
    "@llvm-project-overlay//:extensions.bzl",
    "llvm_repos_extension",
)
use_repo(
    llvm_repos_ext,
    "pyyaml",
    "vulkan_sdk",
)
inject_repo(llvm_repos_ext, "llvm-raw")

# Use IREE's extension (won't create llvm-raw since you're the root module)
iree_ext = use_extension(
    "@iree_core//build_tools/bazel:extensions.bzl",
    "iree_extension",
)
inject_repo(iree_ext, "llvm-raw", "stablehlo", "torch-mlir-raw")
use_repo(
    iree_ext,
    "com_github_dvidelabs_flatcc",
    "com_google_benchmark",
    "com_google_googletest",
    # ... other IREE repos you need
)

# Configure LLVM for calls from your project's BUILD files. IREE configures
# its own llvm-project repository from the same injected llvm-raw sources.
llvm_configure = use_repo_rule(
    "@llvm-raw//utils/bazel:configure.bzl",
    "llvm_configure",
)
llvm_configure(name = "llvm-project")

# Configure torch-mlir from your raw source repository if you enable the Torch
# input plugin.
torch_mlir_configure = use_repo_rule(
    "@torch-mlir-raw//utils/bazel:configure.bzl",
    "torch_mlir_configure",
)
torch_mlir_configure(
    name = "torch-mlir",
    src_workspace = "@torch-mlir-raw//:CMakeLists.txt",
)
```

## Custom Compiler Plugin Registry

IREE's `//compiler/plugins` package loads the plugin registry from the root
workspace:

```python
load("@//build_tools/bazel:default_compiler_plugins.bzl", ...)
```

When IREE is the root module, this resolves to IREE's default registry. A
downstream root workspace can provide a file at the same path to register
additional compiler plugins, replace registration targets, or change the
default enabled plugin IDs. In-tree IREE plugin labels should be qualified with
`@iree_core//...` from such a downstream file.

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
define it yourself if you configure LLVM (see examples above).

### "repository 'stablehlo' is not defined"
You enabled the StableHLO input plugin without providing a `stablehlo`
repository from your root module.

### "repository 'torch-mlir' is not defined"
You enabled the Torch input plugin without configuring a `torch-mlir` repository
from your root module.

### Build errors in LLVM code
Your LLVM version may be incompatible with IREE. Check that your LLVM commit
is close to IREE's expected version.

### Duplicate repository errors
Check for duplicate names within your module's repository mapping. For example,
after defining `stablehlo` locally and injecting it into IREE's extension, do not
also import it with `use_repo(iree_ext, "stablehlo")`. Different modules may each
use the same apparent repository name.
