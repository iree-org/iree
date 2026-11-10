# IREE Plugin API

This is a work in progress to enable IREE compiler plugin support per
[RFC - Proposal to Build IREE Compiler Plugin Mechanism](https://github.com/iree-org/iree/issues/12520).

Dynamic loading is experimental: the API and C++ ABI may change without
backward compatibility. Rebuild plugins with the compiler they will run in.

## Interim Developer Docs

The `PluginManager` mirrors the execution hierarchy of the C API bindings
(`compiler/bindings/c/iree/compiler/embedding_api.h`):

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
symbol for its ID and compatibility metadata. A header hash mismatch is
rejected before registration. Plugins are named on the command line or in the environment:

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

The loader reports load and registration errors. The command-line tools exit
on these errors. An embedded compiler keeps successful registrations and skips
failed dynamic plugins. Registrations from a failed callback are discarded;
callbacks must clean up any other side effects before returning failure.
Registering the same ID twice within a callback is a programming error and
aborts the process. `IREE_DEFINE_COMPILER_PLUGIN` defines both entry points so
one source can support static and dynamic linking.

CMake provides `iree_compiler_register_dynamic_plugin`; Bazel provides
`iree_compiler_register_experimental_dynamic_plugin`. Both build the library
and apply the symbol rename described below. An install tree
provides it through `find_package(IREECompiler)`; see
`samples/compiler_plugins/out_of_tree_example/README.md`.

#### Build requirements

Build the plugin against the host compiler's exact IREE and LLVM/MLIR revisions,
including generated headers from that build. Match these compilation settings:

* Target architecture, pointer width, and platform C++ ABI. Using the host's
  C++ compiler and toolchain is the simplest way to keep these aligned.
* C++ standard library and its ABI options, such as `_GLIBCXX_USE_CXX11_ABI`,
  `_GLIBCXX_DEBUG`, or libc++ ABI configuration macros.
* RTTI and exception handling (`LLVM_ENABLE_RTTI`, `LLVM_ENABLE_EH`, and the
  effective `-frtti`/`-fno-rtti` and `-fexceptions`/`-fno-exceptions` flags).
* Assertions (`NDEBUG`) and LLVM options that change header definitions or
  layouts, such as `LLVM_ENABLE_ABI_BREAKING_CHECKS`. Check the actual compiler
  flags: `IREE_ENABLE_ASSERTIONS` can enable assertions in a release build.
* Sanitizer instrumentation and its runtime requirements, when enabled.

In-tree plugins inherit the build settings. External plugins must match them
explicitly. Matching only `CMAKE_BUILD_TYPE` is insufficient. Optimization levels
and debug information need not match unless they change one of the settings
above.

The tools must link the shared compiler library, the default in both build
systems (`IREE_LINK_COMPILER_SHARED_LIBRARY` in CMake,
`//compiler/src/iree/compiler/API:link_shared` in Bazel). CMake also requires
`IREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS=ON` and
`IREE_ENABLE_THIN_ARCHIVES=OFF`. Plugin archives must contain position-independent
code and must not be thin archives: `llvm-objcopy` rewrites their members.

The build rules rename the plugin's `llvm::` and `mlir::` symbols to match the
compiler and resolve references against its shared library. Do not link another
copy of LLVM, MLIR, or the IREE compiler into the plugin.

#### Compatibility checks and versions

Use `IREE_DEFINE_COMPILER_PLUGIN(id, register_fn, "version")`. It fills in the
compatibility fields from the host headers; plugin authors do not maintain
compatibility version numbers. The supplied version string identifies the
plugin's own release; it is not checked for compatibility.

`IREE_COMPILER_PLUGIN_ABI_HASH` is the single compatibility ID. The build
calculates it from `Client.h`, `PluginEntryPoint.h`, `Pipelines/Options.h`, and
`Utils/OptionUtils.h`, covering both the entry-point layout and the selected
C++ API headers. There is no manual API or ABI version to bump. IREE maintainers
must keep the registration contract in `PluginEntryPoint.h` current so contract
changes also change the ID.

`IREE_COMPILER_ABI_PREFIX` is a symbol namespace, not a compatibility version.
Keep its default value; the installed CMake package supplies the host's value.
It does not need a bump when the plugin API changes.

The loader checks the fixed-width header hash before reading the remaining
entry-point fields or calling registration. The hash does not cover LLVM/MLIR
headers or build settings. A matching hash and successful symbol resolution do not guarantee ABI compatibility: a mismatched plugin can
still load and crash or corrupt memory.

## Extension points

Plugins function by responding to a number of extension points, which
provide the means for further customization. This will be extended over time:

* `static registerPasses()` : Called early in plugin loading to perform static
  registration of passes and pipelines so that they can be used from the
  command line environment and mnemonic tools. This is not much different
  from `globalInitialize()` below, but it is intended for regular use and
  called out separately to avoid triggering warnings related to use of
  global initialization.
* `onRegisterDialects(DialectRegistry&)`: Registers the session's dialects before
  context initialization and activation.
* `onActivate()` : Called when a plugin is activated for a session, having
  both `options` and `context` available. This is the recommended point to
  configure appropriate context hooks for configuring MLIR prior to any
  parsing or operation creation.

HAL targets:

* `populateHALTargetDevices()`
* `populateHALTargetBackends()`

Input dialects:

* `extendInputConversionPreprocessingPassPipeline()`: Adds preprocessing passes
  for a built-in input type.
* `extendCustomInputConversionPassPipeline()`: Called to extend a pass pipeline
  with conversion passes for a given conversion type.
* `populateCustomInputConversionTypes()`: Called to get a list of all
  conversion types this plugin _can_ support.
* `populateDetectedCustomInputConversionTypes()`: Called to get a list of all
  conversion types this plugin _found_ within a given module

Preprocessing:

* `extendPreprocessingPassPipeline()`: Adds passes at the end of preprocessing.

Less frequently used extension points:

* `static globalInitialize()` : Perform once-only process level initialization,
  regardless of whether a plugin will be activated. This happens before command
  line processing and should only be used to massage process-wide static
  registration like things, as third party libraries may require.
* `static registerGlobalDialects(DialectRegistry&)` : Extends the process wide
  initial dialect registry. Prefer `onRegisterDialects()` unless interfacing
  to legacy codebases that require global registration.

## Current Status

* Statically linked, named plugins are supported in CMake (with optional
  inclusion).
* Statically linked, named plugins are supported in Bazel (with optional
  inclusion).
* Dynamic compiler plugins are experimental and supported on Linux and macOS.
* An example in-tree plugin is under `samples/compiler_plugins/example` and
  supports static and dynamic linking from the same source.
* See `iree_compiler_plugin.cmake` for the CMake integration. Specifically,
  the `-DIREE_COMPILER_PLUGINS=example` flag can be used to statically link
  the example plugin.
* [out_of_tree_example](../../../../../samples/compiler_plugins/out_of_tree_example/README.md)
  adds a dialect and pass, with instructions for building against an install tree.
