# IREE Plugin API

This is a work in progress to enable IREE compiler plugin support per
[RFC - Proposal to Build IREE Compiler Plugin Mechanism](https://github.com/openxla/iree/issues/12520).
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

(Not yet implemented)

Dynamic linking proceeds similarly, driven by a combination of environment
variables, API calls to load plugin libs or pre-parsed CLI flags. For each
plugin library located in such a way, it will be `dlopen()`'d and the
corresponding entry point found and used, similar to the static linking case.

Note that only compilers built with `-DIREE_COMPILER_BUILD_SHARED_LIBS=ON` is
supported for this case. That carries a number of restrictions and other issues
that are outside of the immediate scope of plugins.

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

Less frequently used extension points:

* `static globalInitialize()` : Perform once-only process level initialization,
  regardless of whether a plugin will be activated. This happens before command
  line processing and should only be used to massage process-wide static
  registration like things, as third party libraries may require.
* `static registerDialects(DialectRegistry&)` : Extends the process wide
  initial dialect registry. This should not be used unless if absolutely
  necessary or if interfacing to legacy codebases that require it.

## Current Status

* Statically linked, named plugins are supported in CMake. A mechanism has
  not been created for Bazel (which will just always appear to have no
  plugins).
* An example in-tree plugin is under `compiler/plugins/example`.
* See `iree_compiler_plugin.cmake` for the CMake integration. Specifically,
  the `-DIREE_COMPILER_PLUGINS=example` flag can be used to statically link
  the example plugin.
