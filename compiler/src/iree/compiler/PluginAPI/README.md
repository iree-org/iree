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

Plugins can be statically linked into the compiler by way of the
`-DIREE_COMPILER_PLUGINS=` option. This does two things:

* Causes the generated `PluginAPI/Config/StaticLinkedPlugins.inc` to have
  a `HANDLE_PLUGIN_ID(plugin_id)` line.
* Adds the corresponding cc_library dep to the
  `iree::compiler::PluginAPI::Config::StaticLinkedPlugins` target.

During `PluginManager` initialization, the `StaticLinkedPlugins.inc` file is
processed to generate a call to
`iree_register_compiler_plugin_##plugin_id(PluginRegistrar*)`, which is provided
by the plugin and completes registration.

### Dynamic linking

Plugins can also be loaded at run time, from a build configured with
`-DIREE_COMPILER_DYNAMIC_PLUGINS=ON`. Each library is `dlopen()`'d and asked
for its registration through one exported symbol, which reports the plugin's
id and the API version it was built against; a version mismatch is refused
rather than run. Plugins are named either on the command line or, where there
is no command line, in the environment:

```sh
iree-compile --iree-load-plugin=/path/to/libmy_plugin.so --iree-plugin=my_id ...
IREE_LOAD_PLUGINS=/path/to/libmy_plugin.so   # comma-separated, same effect
```

A plugin that fails to load is reported and skipped, so one bad path does not
take the compiler down with it. Registration is otherwise identical to the
static case, and the same source can serve both.

Both build systems provide `iree_compiler_register_dynamic_plugin`, which
builds the library and applies the ABI rename described below.

#### What a dynamic plugin has to agree with

The compiler renames every `llvm::` and `mlir::` symbol to an IREE-private
spelling, so that it can share a process with a foreign LLVM. A plugin is
renamed the same way and resolves against the compiler's shared library, which
constrains its build:

* The tools must link the compiler as a shared library, which is the default
  in both build systems (`IREE_LINK_COMPILER_SHARED_LIBRARY` in CMake,
  `//compiler/src/iree/compiler/API:link_shared` in Bazel). A statically
  linked tool exports no renamed symbols for a plugin to resolve against.
* `-DIREE_ENABLE_THIN_ARCHIVES=OFF`, which is the default. The rename runs
  `llvm-objcopy` over every archive, and a thin archive holds no member
  objects for it to rewrite.
* The plugin's RTTI and exception settings must match the compiler's.
  Building the plugin in the IREE tree gets this right on its own; an
  out-of-tree build has to match its host by hand.
* The same IREE revision. The API version catches a changed entry point, not
  a changed `Client.h`.

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

* Statically linked, named plugins are supported in both build systems, with
  optional inclusion through `IREE_COMPILER_PLUGINS`.
* Dynamically loaded plugins are supported in both build systems, from a build
  configured with `IREE_COMPILER_DYNAMIC_PLUGINS=ON`.
* `samples/compiler_plugins/example` is registered both ways from one source.
  `samples/compiler_plugins/out_of_tree_example` is shaped the way a plugin
  living in another repository would be, with its own dialect and pass.
* See `iree_compiler_plugin.cmake` and
  `build_tools/cmake/iree_plugin_register.cmake` for the CMake integration,
  and `build_tools/bazel/renamed_link.bzl` for Bazel.
