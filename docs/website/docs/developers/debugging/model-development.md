---
hide:
  - tags
tags:
icon: octicons/bug-16
---

# Model development debugging

Bringing up new models or diagnosing regressions in existing models written
using one of IREE's supported [ML frameworks](../../guides/ml-frameworks/index.md)
or downstream projects like
[SHARK-Platform](https://github.com/nod-ai/SHARK-Platform) can involve
debugging up and down the tech stack. Here are some tips to make that process
easier.

## Helpful build settings

### Use a debug build

Build with `-DCMAKE_BUILD_TYPE=Debug` or `-DCMAKE_BUILD_TYPE=RelWithDebInfo` to
include debug information in binaries you build.

### Enable assertions

Build with `-DIREE_ENABLE_ASSERTIONS=ON` to ensure that asserts in compiler
and runtime code are included in your program binaries. If an assert is missed
and the program compiles anyways, the output should not be trusted. The compiler
_must_ not crash on valid input programs, so assert failures should be fixed and
not worked around.

!!! note - "Note: release builds and some CI jobs may not have asserts enabled!"

### Run using sanitizers (ASan/TSan/UBSan)

Building and running using [sanitizers](../debugging/sanitizers.md) can catch
memory usage issues (ASan), thread synchronization issues (TSan), and undefined
behavior (UBSan).

## Helpful compiler and runtime flags

### VM execution tracing

The `--trace_execution` flag to runtime tools like `iree-run-module` will print
each VM instruction as it is executed. This can help with associating other logs
and system behavior with the compiled VM program.

### Tensor tracing

* The `--iree-flow-trace-dispatch-tensors` flag to `iree-compile` inserts
  trace markers for all dispatch operation tensor inputs and outputs. This lets
  you see tensor contents change as the program runs.
* The `--iree-flow-break-dispatch` flag to `iree-compile` inserts breaks after
  a specified dispatch, allowing early termination of the program and shorter
  logs when focusing debugging around a specific dispatch

### Executable substitution

Executable sources can be dumped, edited, and then loaded back into a program
using `--iree-hal-dump-executable-sources-to` and
`--iree-hal-substitute-executable-source`. This can be used for performace
tuning or for debugging (e.g. by replacing a complicated dispatch with a
simpler one).

See <https://github.com/iree-org/iree/pull/12240> for examples.

## Alternate perspectives

### Try using other data types

Nearly all targets support the `i32` and `f32` data types well, while higher
and lower bit depth types and more esoteric types like `bf16` and `complex` may
be supported partially or not at all on some targets.

If a program fails to compile or produces incorrect outputs, consider checking
if the program works after converting to other data types.

!!! tip

    These compiler options automatically convert between several types on
    import:

    * `--iree-input-demote-i64-to-i32`
    * `--iree-input-demote-f32-to-f16`
    * `--iree-input-demote-f64-to-f32`
    * `--iree-input-promote-f16-to-f32`
    * `--iree-input-promote-bf16-to-f32`

If using `iree-run-module --input=@path/to/input_values.npy`, consider also
using `.bin` binary files instead of `.npy` numpy files, since IREE supports
different types than numpy and signedness information is lost at that level.

### Try using other targets / devices

Large parts of IREE's compilation pipelines and runtime libraries are shared
between compiler target backends and runtime HAL devices/drivers. If a program
works in one configuration but fails in another, that indicates an issue or
missing functionality in the failing configuration.

Some configurations also offer unique debugging functionality:

Compiler target | Runtime device | Notable properties for debugging
-- | -- | --
`vmvx` | `local-sync` | Easy to step into generated code, limited type support
`llvm-cpu` | `local-sync` | Single-threaded, broad type support
`llvm-cpu` | `local-task` | Multi-threaded, broad type support
`vulkan-spirv` | `vulkan` | Compatible with Renderdoc ([docs here](../performance/profiling-gpu-vulkan.md#renderdoc))
`cuda` | `cuda` | Compatible with [NVIDIA Nsight Graphics](https://developer.nvidia.com/nsight-graphics)
`rocm` | `hip` | Compatible with [Omniperf](https://github.com/ROCm/omniperf)
`metal-spirv` | `metal` | Compatible with the [Metal Debugger](https://developer.apple.com/documentation/xcode/metal-debugger/)

!!! tip

    See the
    [deployment configurations](../../guides/deployment-configurations/index.md)
    pages for more information about each backend and device.

### Run natively and via Python bindings

Some problems manifest only when running through the Python (or some other
language/framework) bindings. The Python bindings have some non-trivial interop
and memory management across the C/C++/Python boundary.

Try extracting standalone `.mlir` files, compiling through `iree-compile`, then
running through `iree-run-module`. Extracting these artifacts can also help
other developers follow your reproduction steps.

## Reducing complexity

### Top-down reduction

Starting from a full program, try to reduce the program size and complexity
while keeping the issue you are debugging present. This can be either a manual
process or the `iree-reduce` tool can automate it. For manual reduction, here
are some general strategies:

* Reduce tensor sizes (e.g. image dimensions, context lengths) in your ML
  framework
* Cut out duplicate layers (e.g. attention blocks in LLMs)
* If your program has multiple functions, test each in isolation

### Bottom-up reduction

Consider writing unit tests for individual ops or combinations of ops to see
if crashes, bugs, numerical issues, etc. can be reproduced at that scale.

Some existing test suites can be found at these locations:

* <https://github.com/iree-org/iree/tree/main/tests/e2e>
* <https://github.com/nod-ai/SHARK-TestSuite/tree/main/iree_tests/onnx/node/generated>
* <https://github.com/nod-ai/SHARK-TestSuite/tree/main/e2eshark/onnx/operators>
* <https://github.com/nod-ai/SHARK-TestSuite/tree/main/e2eshark/pytorch/operators>
* <https://github.com/openxla/stablehlo/tree/main/stablehlo/tests/interpret>
