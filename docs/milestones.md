# IREE Milestones

## Design

Though many of the core dialects are now in place enough for correctness testing
a large majority of the features we are most excited to demonstrate are still
TODO and will be coming over the next few quarters. You can find a highlighted
set of coming features in the [design roadmap](design_roadmap.md).

## Spring/Summer 2020 Focus Areas

IREE is able to run many foundational models and more are expected to come
online this spring. Much of the work has been on infrastructure and getting the
code in a place to allow for rapid parallel development and now work is ramping
up on op coverage and completeness. There's still some core work to be done on
the primary IREE dialects (`flow` and `hal`) prior to beginning the low-hanging
fruit optimization burn-down, but we're getting close!

### Frontend: Enhanced SavedModel/TF2.0 Support

We are now able to import SavedModels written in the TF2.0 style with resource
variables and some simple usages of TensorList (`tf.TensorArray`, etc).

### Coverage: XLA HLO Ops

A select few ops - such as ReduceWindow - are not yet implemented and need to be
both plumbed through the HLO dialect and the IREE lowering process as well as
implemented in the backends. Work is ongoing to complete the remaining ops such
that we can focus on higher-level model usage semantics.

### Scheduler: Dynamic Shapes

Progress is underway on dynamic shape support throughout the stack. The tf2xla
effort is adding shape propagation/inference upstream and we have a decent
amount of glue mostly ready to accept it.

### HAL: Marl CPU Scheduling

We want to plug in [marl](https://github.com/google/marl) to provide
[CPU-side work scheduling](design_roadmap.md#gpu-like-cpu-scheduling) that
matches GPU semantics. This will enable improved CPU utilization and allow us to
verify the approach with benchmarks.

### Codegen: Full Linalg-based Conversion

A large part of the codegen story for both CPU (via LLVM IR) and GPU (via
SPIR-V) relies on the upstream
[Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/) and associated
lowerings. We are contributing here and have partial end-to-end demonstrations
of conversion. By the end of summer we should be fully switched over to this
path and can remove the index-propagation-based SPIR-V lowering approach in
favor of the more generalized solution.

## Beyond

### HAL: Dawn Implementation

To better engage with the WebGPU and WebML efforts we will be implementing a
[Dawn](https://dawn.googlesource.com/dawn/) backend that uses the same generated
SPIR-V kernels as the Vulkan backend which enables us to target Metal, Direct3D
12, and WebGPU. The goal is to get something working in place (even if
suboptimal) such that we can provide feedback to the various efforts.
