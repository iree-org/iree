# IREE Best Practices

This page contains a list of best practices for getting the most out of IREE,
spanning model authoring, ahead-of-time compilation, and runtime use. Treat
these as a collection of ideas to consider or areas to start benchmarking when
working on your own applications.

## Introduction

Common themes include:

* Give the compiler as much information as possible
* Give the compiler opportunities to batch work together or defer computation
* Keep compute devices saturated with work through pipelining
* Use dense math where possible, particularly for inner loop bodies
* Limit synchronization points between devices like CPUs and GPUs
* Profile early and often, using the right tools for each level of granularity

## Practices for model authoring

### Track state within your model when possible

If your model is stateful prefer to store that state directly within your
program rather than externalizing it through arguments and return values. By
keeping state inside your program the compiler is better able to reason about
it and function calls will have lower overhead.

If you do externalize state, try to pack that state into a limited number of
arguments.

See the
[variables and state](https://github.com/google/iree/tree/main/iree/samples/variables_and_state)
sample for further guidance on tracking and using state.

### Limit uses of dynamic shapes

While IREE aims to support general dynamic shapes use, it is better able to
optimize parts of programs where shapes are static. Slow varying dimensions
like batch index or timestamp are safer uses of dynamic shapes than faster
varying dimensions like the x/y/channel dimensions of images.

See the
[dynamic shapes](https://github.com/google/iree/tree/main/iree/samples/dynamic_shapes)
sample for further guidance on using dynamic shapes.

## Practices for compilation settings

TODO: which compiler targets to use (try both CUDA and Vulkan?)

TODO: use the most specific LLVM target triple you can?

### Tuning compilation heuristics

IREE runs its own suite of benchmarks continuously using the definitions at
https://github.com/google/iree/tree/main/benchmarks. The flags set for these
benchmarks represent the latest manually tuned values for workloads we track
closely and referencing them may help with your own search for peak performance.

Here is a non-exhaustive list of flags which can be tuned when compiling
through the `iree-translate` tool, while full documentation can be found in the
project source code for each flag:

* `--iree-flow-inline-constants-max-byte-length=[integer]`: Maximum byte-length of constants
that can be inlined into dispatch regions. We find values around 16 work best
for when running on a GPU and values around 2048 work best on CPUs.
* `--iree-llvm-loop-unrolling=true`: This flag and other flags like it in
[LLVMTargetOptions.cpp](https://github.com/google/iree/blob/main/iree/compiler/Dialect/HAL/Target/LLVM/LLVMTargetOptions.cpp)
turn on specific optimizations within LLVM when targeting CPUs.
* `--iree-flow-dispatch-formation-enable-operand-fusion=true`: Enable fusing operand
producers during dispatch region formation.
* `--iree-enable-fusion-with-reduction-ops=true`: Allow fusing generic ops with
reductions.

## Practices for runtime use

TODO: sample code, profile numbers

### Tuning runtime settings

When running on the CPU, the task system flags specified in
[iree/task/api.c](https://github.com/google/iree/blob/main/iree/task/api.c)
give control over how worker threads will be created. For example, the
`--task_topology_group_count=3` flag can be set to explicitly run on three
workers rather than rely on heuristic selection that defaults to one worker
per detected physical core.

If running on a single thread or system with no threading support, the
`dylib-sync` HAL driver can be used instead of the more generic `dylib` HAL
driver. The synchronous driver performs execution inline rather than through
IREE's task scheduling system.

### Do the minimum amount of work: cache queries and reuse buffers

When using IREE's runtime libraries, try to front-load queries, particularly
queries using strings that look up into maps like
`iree_runtime_session_call_by_name`, so that hot sections of code are doing the
minimum amount of work: routing inputs through buffers, scheduling runtime
calls, and routing outputs through other buffers.
