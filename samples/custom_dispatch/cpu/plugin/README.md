# Custom CPU Dispatch Functions for Dynamically-linked Plugins

See the [custom_dispatch README](/samples/custom_dispatch/README.md) for an
overview of this approach. This sample is derived from the
[custom_dispatch/cpu/embedded/](/samples/custom_dispatch/cpu/embedded/) sample
and information about the calling conventions can be found there.

This sample demonstrates how to define external device functions that can be
dispatched from within IREE programs via simple function calls. Here the
functions are declared in the MLIR executables, called as normal calls, and
then defined in a .c file that is either compiled for the system platform
(ELF, DLL, DyLib, etc) or cross-compiled into a platform-independent embedded
ELF. The compiler merely emits the imports and leaves it to the user to specify
which plugins to load at runtime such that the imports can be resolved.

Note that dynamically-linked plugins are discouraged unless absolutely required.
Prefer instead to use compiler embedded imports that allow for hermetic
deployable artifacts that don't require re-deploying runtimes and produce the
minimal amount of code as only the imports required by the compiled program are
pulled in and they can be optimized with LTO to propagate ranges/constants.
Dynamically-linked plugins should generally only be used for extreme mechanisms
like JITs, though even those are better done as ahead-of-time code generation in
the compiler. IREE supports dynamically-linked imports for completeness and they
should be used with careful consideration.

## Workflow for System Dynamic Libraries

```
+----------+    +---------------+      +--------------+
| plugin.c | -> | plugin.so/dll |-+    | example.mlir |
+----------+    +---------------+ |    +--------------+
                                  |           v
                                  |      iree-compile
                                  |           v
                                  |    +--------------+
                                  |    | example.vmfb | (non-hermetic)
                                  |    +--------------+
                                  |           |
                                  +-----+-----+
                                        v
                               +-----------------+
                               | iree-run-module |
                               +-----------------+
```

When plugins need to rely on platform-specific functionality (syscalls, TLS,
etc) they can be built as normal system libraries of the type that can be loaded
with dlopen/LoadLibrary/etc. Users will need to handle deployment themselves and
the IREE runtime will load the plugin library using the platform APIs. There are
still restrictions with this approach as the imports provided by the plugin will
be called from arbitrary threads where syscalls, TLS, and other features are
quite complicated to get right. An advantage of system libraries are that most
tooling (perf, debuggers, etc) will work with no additional configuration. As
such it's recommended that if portable ELF libraries are used for deployment
users still preserve a path where they can be compiled as system libraries.

1. The user authors their functions in whatever language they want with whatever
   system dependencies they want (with caveats/YMMV) and exposes them via the
   IREE C [executable_plugin.h](/runtime/src/iree/hal/local/executable_plugin.h)
   API. These functions can cover entire workgroups (and a dispatch can
   be a single workgroup so effectively just function calls) or be utilities
   used by the function for localized work (microkernels, data type conversion,
   etc). It's important to remember that parallelism scheduling is done
   _outside_ of the function via the workgroup count and multiple threads may be
   executing the function at any time.

   In addition to the import function (see
   [custom_dispatch/cpu/embedded/](/samples/custom_dispatch/cpu/embedded/)
   for the structure of the imports) the plugin must provide a query function
   that is used to provide the plugin information to the runtime:

```c
IREE_HAL_EXECUTABLE_PLUGIN_EXPORT const iree_hal_executable_plugin_header_t**
iree_hal_executable_plugin_query(
    iree_hal_executable_plugin_version_t max_version, void* reserved) {
  // Return a plugin header populated with metadata and function pointers.
}
```

2. Source files are compiled to platform dynamic libraries via normal build
   system goo. Each platform and architecture the user is targeting will need
   its own libraries. Note that only the header file is required to be included
   and no IREE runtime libraries need to be linked into the plugin.

```cmake
add_library(my_plugin SHARED my_plugin.c)
target_include_directories(...)
```

3. The user (or compiler transforms) adds calls to their functions by declaring
   them.

```mlir
func.func private @simple_mul_workgroup(
    %binding0: memref<?xf32>, %binding1: memref<?xf32>, %binding2: memref<?xf32>,
    %dim: index, %tid: index)
...
func.call @simple_mul_workgroup(%memref0, %memref1, %memref2, %dim, %tid) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index, index) -> ()
```

4. The user either programmatically registers the plugins via the plugin manager
   or when using IREE tools passes them using the `--executable_plugin=` flag.
   Note that imports are resolved in reverse registration order such that
   fallbacks can be supported; a reference plugin can be registered first
   followed by more specialized plugins that may only handle a subset of
   imports.

```
iree-run-module \
    --device=local-sync \
    --executable_plugin=my_plugins.so \
    --executable_plugin=other_plugins.so \
    --function=mixed_invocation \
    --input=8xf32=2 \
    --input=8xf32=4
```

## Workflow for Embedded ELF Libraries

```
+----------+      +-------------------+       +--------------+
| plugin.c | -+-> | plugin_aarch64.so | -+    | example.mlir |
+----------+  |   +-------------------+  |    +--------------+
              |   +-------------------+  |           v
              +-> | plugin_x86_64.so  | -+      iree-compile
                  +-------------------+  |           v
                       +------------+    |    +--------------+
                       | plugin.sos | <--+    | example.vmfb | (non-hermetic)
                       +------------+         +--------------+
                             |                       |
                             +----------+------------+
                                        v
                               +-----------------+
                               | iree-run-module |
                               +-----------------+
```

The workflow is similar to the system library version except that the plugin
code needs to be written in a bare-metal flavor (no TLS, no threads, no malloc,
etc). Most kernel libraries not performing JITing can be authored like this and
take advantage of the multi-targeting and cross-platform support provided by the
plugin loader. A plugin can be compiled for multiple architectures (aarch64,
x86_64, etc) and then load and run on all platforms (Windows, MacOS, Linux,
and bare-metal).

See the sample `CMakeLists.txt` for how the standalone plugins can be compiled
using the appropriate clang flags. Other compilers can be used if care is taken
to ensure compatible platform-agnostic ELF files. After building each
architecture-specific ELF they can be combined into a FatELF using the
`iree-fatelf` tool; this single `.sos` file can contain multiple architectures
and the required one will be loaded at runtime.

## Instructions

This presumes that `iree-compile` and `iree-run-module` have been installed or
built. [See here](https://openxla.github.io/iree/building-from-source/getting-started/)
for instructions for CMake setup and building from source.

1. Build the `iree-sample-deps` CMake target to compile
   [standalone_plugin.c](./standalone_plugin.c) and
   [system_plugin.c](./system_plugin.c) sources to object files for aarch64 and
   x86_64 or the current target system:

    ```
    cmake --build ../iree-build/ --target iree-sample-deps
    ```

    In a user application this would be replaced with whatever build
    infrastructure the user has for compiling code to object files. No IREE
    compiler or runtime changes are required and the normal compiler install can
    be used. Note that specific flags are required when producing the object
    files.

2. Compile the [example module](./example.mlir) to a .vmfb file and pass
   the path to the build directory so the .spv files can be found:

    ```
    iree-compile \
        samples/custom_dispatch/cpu/plugin/example.mlir \
        -o=/tmp/example.vmfb
    ```

3. Run the example program using the plugins for either platform-independent
   embedded ELF files or the system libraries:

    ```
    iree-run-module \
        --device=llvm-sync \
        --executable_plugin=samples/custom_dispatch/cpu/plugin/standalone_plugin.sos \
        --function=mixed_invocation \
        --input=8xf32=2 \
        --input=8xf32=4 \
        /tmp/example.vmfb
    ```

    ```
    iree-run-module \
        --device=llvm-sync \
        --executable_plugin=samples/custom_dispatch/cpu/plugin/system_plugin.so \
        --function=mixed_invocation \
        --input=8xf32=2 \
        --input=8xf32=4 \
        /tmp/example.vmfb
    ```
