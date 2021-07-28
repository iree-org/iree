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

TODO: mention parameters to tune

TODO: which compiler targets to use (try both CUDA and Vulkan?)

TODO: use the most specific LLVM target triple you can?

## Practices for runtime use

### Do the minimum amount of work: cache queries and reuse buffers

Try to front-load queries, particularly queries using strings that look up into
maps like `iree_runtime_session_call_by_name`, so that hot sections of code are
doing the minimum amount of work: routing inputs through buffers, scheduling
runtime calls, and routing outputs through other buffers.

TODO: sample code, profile numbers
