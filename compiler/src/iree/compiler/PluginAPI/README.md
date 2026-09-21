# IREE Plugin API

This is a work in progress to enable IREE compiler plugin support per
[RFC - Proposal to Build IREE Compiler Plugin Mechanism](https://github.com/iree-org/iree/issues/12520).
This document will be replaced with a more comprehensive single-source once
the work is complete.

## Interim Developer Docs

The `PluginManager` mirrors the execution hierarchy of the C API bindings
(`compiler/bindings/c/iree/compiler/embedding_api`):

* Global Initialization
* Global CLI setup
* Session (`iree_compiler_session_t`)
* Invocation (`iree_compiler_invocation_t`)

Compiler plugins are activated at the session level (`iree_compiler_session_t`)
and can be independently selected and activated based on session level flags
(`ireeCompilerSessionSetFlags` / `ireeCompilerSessionGetFlags`). Optionally,
when running in an LLVM-like tool, session level options can be bootstrapped
from the Global CLI.

This necessitates a two-phase hierarchy where we maintain a registry of
*available* plugins, using them to bootstrap options setup. Based on flags and
configuration, some subset of *available* plugins will be activated and bound
to a session (which has a 1:1 relationship with an `MLIRContext`).

Most of these mechanics are opaque to the user, if desired, by the use of the
`PluginSession` CRTP base class, which can be used to handle the boiler-plate
and provide an `OptionsBinder` based class for options. Typically, such a
plugin will ignore everything up to its `onActivate()` hook, which is called
once an `MLIRContext` has been set and is ready for use. At this point, its
specified `OptionsTy` class will be available in the `PluginSession` as
`options`, with all configuration complete.

### Static linking

In CMake, select statically linked compiler plugins with
`-DIREE_COMPILER_PLUGINS=<id1;id2>`. In Bazel, use
`--iree_compiler_plugins=<id1,id2>`. Selection does two things:

* Causes the generated `PluginAPI/Config/StaticLinkedPlugins.inc` to have
  a `HANDLE_PLUGIN_ID(plugin_id)` line.
* Adds the corresponding cc_library dep to the
  `iree::compiler::PluginAPI::Config::StaticLinkedPlugins` target.

During `PluginManager` initialization, the `StaticLinkedPlugins.inc` file is
processed to generate a call to
`iree_register_compiler_plugin_##plugin_id(PluginRegistrar*)`, which is provided
by the plugin and completes registration.

### Dynamic linking

Dynamic compiler plugins are experimental and supported on Linux and macOS.
In CMake, enable compiler symbol exports with
`-DIREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS=ON`. Bazel's compiler shared
library exports these symbols without an additional option.
Each library is opened with `dlopen()` and queried through one exported
symbol for its id, API version and header hash; a mismatch in version or hash is
refused. Plugins are named on the command line or in the environment:

```sh
iree-compile --iree-load-plugin=/path/to/libmy_plugin.so --iree-plugin=my_id ...
IREE_LOAD_PLUGINS=/path/to/libmy_plugin.so \
  iree-compile --iree-plugin=my_id ...
```

Repeat `--iree-load-plugin=<path>` to load multiple libraries, or set
`IREE_LOAD_PLUGINS` to a comma-separated list of paths. Loading makes a plugin
available; `--iree-plugin=<id>` activates an explicitly selected plugin for a
session. Use `--iree-print-plugin-info` during compilation to list available
and activated IDs.

Embedded users must set `IREE_LOAD_PLUGINS` before the first
`ireeCompilerGlobalInitialize()` call. Loading happens once per process;
subsequent initialization calls do not reread the environment or load new
plugins. `ireeCompilerSessionSetFlags` can select an already registered plugin
with `--iree-plugin=<id>`, but cannot load a library with `--iree-load-plugin`.

Load and registration errors detected by the loader are reported. The tools
then exit; a host of the compiler
library carries on with successful registrations, skipping dynamic plugins
whose registration callback fails or whose IDs collide. Registrations from a
failed callback are discarded; callbacks must not leave other global side
effects on failure. `IREE_DEFINE_COMPILER_PLUGIN` serves
static and dynamic registration from one source.

CMake provides `iree_compiler_register_dynamic_plugin`; Bazel provides
`iree_compiler_register_experimental_dynamic_plugin`. Both build the library
and apply the symbol rename described below. An install tree
provides it through `find_package(IREECompiler)`; see
`samples/compiler_plugins/out_of_tree_example/README.md`.

#### Build requirements

The compiler renames every `llvm::` and `mlir::` symbol so it can share a
process with another LLVM. A plugin is renamed the same way and resolves against
the compiler's shared library, so:

* The tools link the compiler as a shared library, the default in both build
  systems (`IREE_LINK_COMPILER_SHARED_LIBRARY` in CMake,
  `//compiler/src/iree/compiler/API:link_shared` in Bazel). Statically linked
  tools do not provide the exported compiler ABI required by these plugins.
* `-DIREE_ENABLE_THIN_ARCHIVES=OFF`, the default. `llvm-objcopy` cannot rewrite
  a thin archive's members.
* RTTI and exception settings match the compiler's. In-tree builds inherit
  them; out-of-tree builds set them by hand.
* The same plugin API headers. `IREE_COMPILER_PLUGIN_ABI_HASH` covers
  `Client.h`, `PluginEntryPoint.h`, `Pipelines/Options.h` and
  `Utils/OptionUtils.h`. The llvm/mlir headers behind them are not hashed; the
  rename does not check their compatibility. Use the host compiler's exact
  IREE and LLVM/MLIR revisions and ABI-affecting build settings (including
  assertions and the C++ standard library). A matching hash and successful
  symbol resolution do not guarantee ABI compatibility; an incompatible
  plugin may still load and crash or corrupt memory.

## Extension points

Plugins function by responding to a number of extension points, which
provide the means for further customization. This will be extended over time:

* `static registerPasses()` : Called early in plugin loading to perform static
  registration of passes and pipelines so that they can be used from the
  command line environment and mnemonic tools. This is not much different
  from `globalInitialize()` below, but it is intended for regular use and
  called out separately to avoid triggering warnings related to use of
  global initialization.
* `onActivate()` : Called when a plugin is activated for a session, having
  both `options` and `context` available. This is the recommended point to
  provide a `DialectRegistry` and configure appropriate context hooks for
  configuring MLIR prior to any parsing or operation creation.

HAL targets:

* `populateHALTargetBackends()`

Input dialects:

* `extendCustomInputConversionPassPipeline()`: Called to extend a pass pipeline
  with conversion passes for a given conversion type.
* `populateCustomInputConversionTypes()`: Called to get a list of all
  conversion types this plugin _can_ support.
* `populateDetectedCustomInputConversionTypes()`: Called to get a list of all
  conversion types this plugin _found_ within a given module

Less frequently used extension points:

* `static globalInitialize()` : Perform once-only process level initialization,
  regardless of whether a plugin will be activated. This happens before command
  line processing and should only be used to massage process-wide static
  registration like things, as third party libraries may require.
* `static registerDialects(DialectRegistry&)` : Extends the process wide
  initial dialect registry. This should not be used unless if absolutely
  necessary or if interfacing to legacy codebases that require it.

## Current Status

* Statically linked plugins are selected with `IREE_COMPILER_PLUGINS` in CMake
  and `--iree_compiler_plugins` in Bazel.
* Dynamic plugins are experimental in both build systems. CMake requires
  `IREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS=ON`; Bazel has no equivalent
  feature gate.
* `samples/compiler_plugins/example` is registered both ways from one source.
  `samples/compiler_plugins/out_of_tree_example` has its own dialect and pass,
  as a plugin in another repository would.
* See `iree_compiler_plugin.cmake` and
  `build_tools/cmake/iree_plugin_register.cmake` for the CMake integration,
  and `build_tools/bazel/renamed_link.bzl` for Bazel.
