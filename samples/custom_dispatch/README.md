# Custom Dispatches

"Dispatches" in IREE are parallelized device function calls where "device" may
be a CPU task system, a GPU, or a dedicated accelerator. Parallelism is a
first-class concept in this model and plugging custom dispatches into IREE
requires reasoning about these function calls as if they were parallelized and
dispatched across a 3D grid (as on GPUs or CPU task systems). Note that a
degenerate case of grid dispatch is a grid size of 1x1x1 which turns the
dispatches into simple (if inefficient) device-side function calls.

In normal workflows the IREE compiler forms the dispatch functions by way of
fusion and then uses code generation ("codegen") to translate the dispatch
functions into backend-specific forms like PTX, SPIR-V, or LLVM IR. It's
possible to augment this translation by either bypassing these steps entirely
and providing the already translated representation of the dispatch function or
extending the code generation portion by calling out to external functions from
within the generated dispatch (sometimes "microkernels" or "device libraries").

There's ongoing work across the core IREE compiler and specific backends to
enable more extension points and ways to connect to them from frontends or
compiler transforms. This current set of samples demonstrates a very early
version of this extensibility that can be used to tunnel through dispatch
workloads by bypassing code generation (in the case of PTX/SPIR-V) or coarsely
interoperating (CPU function calls). In its current form it is intended for
things that would traditionally be custom ops in ML frameworks and produces much
smaller, hermetic, retargetable, and optimizable programs than traditional
custom ops can as there's nearly zero performance delta between what the
compiler can dispatch and what the user decides to dispatch.

## Approaches

In the fullness of time all backends will support all approaches but currently
there are limitations and these samples only cover the supported cases:

|                    | CPU                | CUDA               | Metal | SPIR-V             | WGSL            |
|--------------------|:------------------:|:------------------:|:-----:|:------------------:|:---------------:|
| Static Functions   | :white_check_mark: | TBD                | TBD   | TBD                | TBD             |
| Dynamic Functions  | :white_check_mark: | TBD                | TBD   | :grey_question:    | :grey_question: |
| Static Dispatches  | TBD                | :white_check_mark: | TBD   | :white_check_mark: | TBD             |
| Dynamic Dispatches | TBD                | TBD                | TBD   | TBD                | TBD             |
| Commands           | TBD                | TBD                | TBD   | TBD                | TBD             |

### Statically-linked Function Calls

**Overview**: user defines functions in .c files, compiles them with specific
settings to .o files, emits calls to the functions in IR interleaved with other
IR, and lets the IREE compiler link the objects into the final binary.

**Workflow**:

```
+-------------+               +---------------------+       +--------------+
| functions.c | -> clang -+-> | functions_aarch64.o | -+    | example.mlir |
+-------------+           |   +---------------------+  |    +--------------+
                          |   +---------------------+  |           v
                          +-> | functions_x86_64.o  | -+----> iree-compile
                              +---------------------+              v
                                                            +--------------+
                                                   hermetic | example.vmfb |
                                                            +--------------+
```

**Samples**:

* CPU: [custom_dispatch/cpu/embedded/](./cpu/embedded/) (.c -> .o)

This approach is usable both from frontends as well as something that can be
synthesized by compiler transforms ("replace op X with call to extern fX and
link in object f.o") and the object files referenced can be generated on the
fly and embedded into the IR to allow for compile-time specialization.

This is the preferred method of extension as it preserves IREE's ability to
specialize executables, optimize aggressively across call boundaries,
hermetically deploy the custom code without runtime changes, and portably target
multiple platforms and architectures.

### Dynamically-linked Function Calls

**Overview**: user defines functions in any source language with a compatible C
ABI, wires them up and links them in their runtime binary, declares the
externally available functions in IR, and emits calls to the functions in IR
interleaved with other IR.

**Workflows**:

Statically-linked into the hosting runtime:
```
                     +--------------+   +--------------+
                     | example.mlir |   | runtime srcs | -+
                     +--------------+   +--------------+  |
+--------------+            v           +--------------+  |
| declarations | ----> iree-compile     | imports.c    | -+-> custom runtime
+--------------+            v           +--------------+            ^
                     +--------------+                               |
                     | example.vmfb | - - - - - - - - - - - - - - - +
                     +--------------+
```

Dynamically-linked into the hosting runtime via system libraries:
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

Dynamically-linked into the hosting runtime via portable embedded ELFs:

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

**Samples**:

* CPU (plugins): [custom_dispatch/cpu/plugin/](./cpu/plugin/) (.c -> .so/.sos)

Unlike the other approaches this requires runtime device support for dynamic
linking and introduces complexity to the user as they must be careful to version
their input programs and their runtime libraries themselves. IREE's CPU backend
does provide basic support for optional imports such that users can emit their
calls and add fallbacks but otherwise such behavior falls on the user to
implement.

### Statically-linked Dispatch Functions

**Overview**: user produces the final dispatch executable binary themselves,
declares it in IR, and dispatches it.

**Workflow**:

```
+------------+              +-------------------+       +--------------+
| kernels.cu | -> nvcc -+-> | kernels_sm_52.ptx | -+    | example.mlir |
+------------+          |   +-------------------+  |    +--------------+
                        |   +-------------------+  |           v
                        +-> | kernels_sm_80.ptx | -+----> iree-compile
                            +-------------------+              v
                                                        +--------------+
                                               hermetic | example.vmfb |
                                                        +--------------+
```

**Samples**:

* CUDA/PTX: [custom_dispatch/cuda/kernels/](./cuda/kernels/) (.cu -> .ptx)
* Vulkan/SPIR-V: [custom_dispatch/vulkan/shaders/](./vulkan/shaders/) (.glsl -> .spv)

Here IREE is used for scheduling the work and ensuring that buffers and
parameters are passed to the dispatch function but otherwise treats them as
opaque. This disables some IREE optimizations but the functions are still able
to be scheduled concurrently and with parallelization. Many custom kernels can
often be implemented like this instead of needing much heavier-weight runtime
custom calls that prevent the asynchronous scheduling that IREE uses.

The dispatch functions are embedded into the IREE compiler outputs such that no
runtime changes are required and compiled programs are hermetic. Multi-targeting
is enabled by allowing the user to provide binaries for the target devices and
architectures they are compiling for.

### Dynamically-linked Dispatch Functions

**Overview**: user defines functions in a target-specific source language,
compiles them into target-specific libraries, wires them up and links them in
their runtime binary, declares the externally available dispatch executable in
IR, and emits calls to the functions in IR interleaved with other IR.

**Workflow**:

```
                     +--------------+   +--------------+
                     | example.mlir |   | runtime srcs | -+
                     +--------------+   +--------------+  |
+--------------+            v           +--------------+  |
| declarations | ----> iree-compile     | dispatches.c | -+-> custom runtime
+--------------+            v           +--------------+            ^
                     +--------------+                               |
                     | example.vmfb | - - - - - - - - - - - - - - - +
                     +--------------+
```

**Samples**: plumbing required.

Similar to statically-linked dispatch functions IREE is doing the scheduling and
managing resources but deferring the entire dispatch logic to externally-sourced
executables. In contrast to the static linking approach this has the compiler
emit references to runtime-provided target-specific executables that must be
built into the runtime. This means that deployment gets more complicated as
compiler-produced outputs are no longer hermetic and users must handle
versioning and platform constraints themselves.

### Custom Commands

**Overview**: user writes custom command buffer operations in VM modules,
declares them in IR, dispatches them, and then links their custom modules into
the runtime.

**Workflow**:

```
                     +--------------+   +--------------+
                     | example.mlir |   | runtime srcs | -+
                     +--------------+   +--------------+  |
+--------------+            v           +--------------+  |
| declarations | ----> iree-compile     | module.c     | -+-> custom runtime
+--------------+            v           +--------------+            ^
                     +--------------+                               |
                     | example.vmfb | - - - - - - - - - - - - - - - +
                     +--------------+
```

**Samples**: plumbing required.

Though more work than the other approaches this allows for a host-side call that
can produce new transfer or execution commands. The compiler effectively treats
these calls as if they were custom versions of `hal.command_buffer.dispatch`/
`iree_hal_command_buffer_dispatch` and the runtime custom module receives the
device, command buffer, and push constants/bindings. Portable modules can
use the HAL APIs to schedule more commands (multiple dispatches, transfers,
collectives, etc) while backend-specific ones can crack open the HAL objects and
get the internals with all the normal API stability caveats.

An example use case would be calling a CUDA library that takes a `CUstream` as
an argument. The user would define their custom module with a call that uses the
`iree/hal/drivers/cuda/api.h` to cast the command buffer to a stream (using
graph capture when the command buffer is constructing a graph), make the call,
and then return. This only works if the library is designed for asynchronous and
deferred execution - if the library makes large allocations, schedules blocking
operations, or has side-effecting behavior then full custom modules must be
used (see the [custom_module/async/](/samples/custom_module/async/) sample).

When at all possible the dispatch function call or dispatch function
substitution approaches should be used instead and in many cases that is
sufficient for most workloads not involving other libraries.

### Compile Time Inlining Custom Function Calls

**Overview**: user defines functions with MLIR dialects IREE is able to ingest
paired with a matcher and replacement pattern. The matcher runs as preprocessing
and calls into the replacement pattern for all successful matches. The
replacement pattern imports a function from the externally 
ABI, wires them up and links them in their runtime binary, declares the
externally available functions in IR, and emits calls to the functions in IR
interleaved with other IR.

**Workflows**:

Statically matched and imported external functions
```
                                            +--------------+
                                            | example.mlir |
+--------------+       +--------------+     +--------------+
| (one of the  |       | functions +  |            v
| above static | ----> | matchers +   | ----> iree-compile
| workflows)   |       | replace.mlir |            v
+--------------+       +--------------+     +--------------+
                                            | example.vmfb |
                                            +--------------+
```

**Samples**:

* CPU: [custom_dispatch/cpu/embedded/](./cpu/embedded/) (.c -> .o)
  * [custom_dispatch/cpu/embedded/](./cpu/embedded/example_transform_spec.mlir) (.mlir)
* Vulkan/SPIR-V: [custom_dispatch/vulkan/shaders/](./vulkan/shaders/) (.glsl -> .spv)
  * [custom_dispatch/vulkan/shaders/](./vulkan/shaders/example_transform_spec.mlir) (.mlir)

The above two samples build on top of a couple of the static workflows shown
above, but should work with any of the other approaches. The idea is to separate
the custom kernel from the target module to be compiled, allowing integration of
custom dispatches with default IREE codegen without the need to build a custom
set of compiler tools around IREE to generate the necessary IR.

There are a number of possible points at which the match and replace can happen;
the above shows it after import + input conversion. Other plugin points are
possible (e.g. before input conversion or after global optimization), but
currently are missing some ergonomics on the available matchers.

### Others

Most other situations are covered by [custom modules](/samples/custom_module/).
These can still be asynchronous and scheduled in device order but incur
additional overheads and deployment complexity as custom runtimes are required.
