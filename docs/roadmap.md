<!--
  Copyright 2019 Google LLC

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

       https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# IREE Roadmap

## Winter 2019

Our goal for the end of the year is to have depth in a few complex examples
(such as streaming speech recognition) and breadth in platforms. This should
hopefully allow for contributions both from Googlers and externally to enable
broader platform support and optimizations as well as prove out some of the core
IREE concepts.

### Frontend: SavedModel/TF2.0

MLIR work to get SavedModels importing and lowering through the new MLIR-based
tf2xla bridge. This will give us a clean interface for writing stateful sample
models for both training and inference. The primary work on the IREE-side is
adding support for global variables to the sequencer IR and sequencer runtime
state tracking.

### Coverage: XLA HLO Ops

A majority of XLA HLO ops (what IREE works with) are already lowering to both
the IREE interpreter and the SPIR-V backend. A select few ops - such as
ReduceWindow and Convolution - are not yet implemented and need to be both
plumbed through the HLO dialect and the IREE lowering process as well as
implemented in the backends.

### Sequencer: IR Refactoring

The current sequencer IR is a placeholder designed to test the HAL backends and
needs to be reworked to its final (initial) form. This means rewriting the IR
description files, implementing lowerings, and rewriting the runtime dispatching
code. This will enable future work on codegen, binary size evaluation,
performance evaluation, and compiler optimizations around memory aliasing and
batching.

### Sequencer: Dynamic Shapes

Dynamic shapes requires a decent amount of work on the MLIR-side to flesh out
the tf2xla bridge such that we can get input IR that has dynamic shapes at all.
The shape inference dialect also needs to be designed and implemented so that we
have shape math in a form we can lower. As both of these are in progress we plan
to mostly design and experiment with how the runtime portions of dynamic shaping
will function in IREE.

### HAL: Dawn Implementation

To better engage with the WebGPU and WebML efforts we will be implementing a
[Dawn](https://dawn.googlesource.com/dawn/) backend that uses the same generated
SPIR-V kernels as the Vulkan backend but enables us to target Metal, Direct3D
12, and WebGPU. The goal is to get something working in place (even if
suboptimal) such that we can provide feedback to the various efforts.

### HAL: SIMD Dialect and Marl Implementation

Reusing most of the SPIR-V lowering we can implement a simple SIMD dialect for
both codegen and JITing. We're likely to start with the
[WebAssembly SIMD spec](https://github.com/WebAssembly/simd/blob/master/proposals/simd/SIMD.md)
for the dialect (with the goal of being trivially compatible with WASM and to
avoid bikeshedding). Once we have at least one lowering to executable code
(either via codegen to JITing) we can use [Marl](https://github.com/google/marl)
to provide the work scheduling. This should be roughly equivalent to performance
to Swiftshader however with far less overhead. The ultimate goal is to be able
to delete the current IREE interpreter.

## Spring 2020

With the foundation laid in winter 2019 we'll be looking to expand support,
continue optimizations and tuning, and implement the cellular batching
techniques at the core of the IREE design.
