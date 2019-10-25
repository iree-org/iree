# IREE: An Experimental MLIR Execution Environment

**DISCLAIMER**: This is not an officially supported Google product. It's an
experimental playground for low-level/tightly integrated machine learning
libraries that make use of modern hardware acceleration APIs and techniques (see
[non goals](#non-goals)).

## Table of Contents

-   [Quickstart](#quickstart)
-   [Project Goals](#project-goals)
-   [Milestones](#milestones)
-   [Status](#status)
-   [Dependencies](#dependencies)
-   [License](#license)

<a name="quickstart"></a>

## Quickstart

More Coming soon! Performing full model translation may require a few steps
(such as ensuring you have a working TensorFlow build), however we'll have
pre-translated example models that allow independent testing of the runtime
portions.

*   [Getting Started on Windows](docs/getting_started_on_windows.md)
*   [Getting Started on Linux](docs/getting_started_on_linux.md)
*   [Getting Started](docs/getting_started.md)

See also:

*   [Using Colab](docs/using_colab.md)
*   [Vulkan and SPIR-V](docs/vulkan_and_spirv.md)

<a name="project-goals"></a>

## Project Goals

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment,
pronounced as "eerie") is an experimental compiler backend for
[MLIR](https://github.com/tensorflow/mlir) that lowers ML models to an IR that
is optimized for real-time mobile/edge inference against heterogeneous hardware
accelerators.

The IR produced contains the sequencing information required to communicate
pipelined data dependencies and parallelism to low-level hardware APIs like
Vulkan and embed hardware/API-specific binaries such as SPIR-V or compiled ARM
code. As the IR is specified against an abstract execution environment there are
many potential ways to run a compiled model, and one such way is included as an
example and testbed for runtime optimization experiments.

The included layered runtime scales from generated code for a particular API
(such as emitting C code calling external DSP kernels), to a HAL (**H**ardware
**A**bstraction **L**ayer) that allows the same generated code to target
multiple APIs (like Vulkan and Direct3D 12), to a full VM allowing runtime model
loading for flexible deployment options and heterogeneous execution. Consider
both the compiler and the included runtime a toolbox for making it easier - via
the versatility of MLIR - to take ML models from their source to some varying
degree of integration with your application.

### Demonstrate MLIR

IREE has been developed alongside MLIR and is used as an example of how
non-traditional ML compiler backends and runtimes can be built: it focuses more
on the math being performed and how that math is scheduled rather than graphs of
"ops" and in some cases allows doing away with a runtime entirely. It seeks to
show how more holistic approaches that exploit the MLIR framework and its
various dialects can be both easy to understand and powerful in the
optimizations to code size, runtime complexity, and performance they enable.

### Demonstrate Advanced Models

By using models with much greater complexity than the usual references (such as
MobileNet) we want to show how weird things can get when model authors are
allowed to get creative: dynamic shapes, dynamic flow control, dynamic
multi-model dispatch (including models that conditionally dispatch other
models), streaming models, tree-based search algorithms, etc. We are trying to
build IREE from the ground-up to enable these models and run them efficiently on
modern hardware. Many of our example models are sequence-to-sequence language
models from the [Lingvo](https://github.com/tensorflow/lingvo) project
representing cutting edge speech recognition and translation work.

### Demonstrate ML-as-a-Game-Engine

An observation that has driven the development of IREE is one of ML workloads
not being much different than traditional game rendering workloads: math is
performed on buffers with varying levels of concurrency and ordering in a
pipelined fashion against accelerators designed to make such operations fast. In
fact, most ML is performed on the same hardware that was designed for games! Our
approach is to use the compiler to transform ML workloads to ones that look
eerily _(pun intended)_ similar to what a game performs in per-frame render
workloads, optimize for low-latency and predictable execution, and integrate
well into existing systems both for batched and interactive usage. The IREE
runtime is designed to feel more like game engine middleware than a standalone
ML inference system, though we still have much work towards that goal. This
should make it easy to use existing tools for high-performance/low-power
optimization of GPU workloads, identify driver or system issues introducing
latency, and help to improve the ecosystem overall.

### Demonstrate Standards-based ML via Vulkan and SPIR-V

With the above observation that ML can look like games from the systems
perspective it follows that APIs and technologies good for games should probably
also be good for ML. In many cases we've identified only a few key differences
that exist and just as extensions have been added and API improvements have been
made to graphics/compute standards for decades we hope to demonstrate and
evaluate small, tactical changes that can have large impacts on ML performance
through these standard APIs. We would love to allow hardware vendors to be able
to make ML efficient on their hardware without the need for bespoke runtimes and
special access such that any ML workload produced by any tool runs well. We'd
consider the IREE experiment a success if what resulted was some worked examples
that help advance the entire ecosystem!

<a name="non-goals"></a>

## Non-Goals

*   Replace parts of the supported TensorFlow ecosystem of tools: The authors
    within Google work closely with TensorFlow and contribute to it regularly.
    However, IREE is exploring some different angles of the problem and is
    experimental. We will seek to leverage anything of value that we learn in an
    appropriate way to make TensorFlow better over time, but the two should not
    be conflated.
*   Providing an [SLA](https://en.wikipedia.org/wiki/Service-level_agreement) of
    any kind: IREE is infrastructure research, not a supported product. If it
    gains mind-share or traction, we would revisit that in conjunction with
    finding a more permanent way to align it with the broader constellation of
    ML tooling.

<a name="milestones"></a>

## Milestones

We are currently just at the starting line, with basic
[MNIST MLP](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py)
running end-to-end on both a CPU interpreter and Vulkan. As we scale out the
compiler we will be increasing the complexity of the models that can run and
demonstrating more of the optimizations we've found useful in previous efforts
to run them efficiently on edge devices.

A short-term
[Roadmap](https://github.com/google/iree/blob/master/docs/roadmap.md) is
available talking about the major areas where are focusing on in addition to the
more infrastructure-focused work listed below.

We'll be setting up GitHub milestones with issues tracking major feature work we
are planning. For now, our areas of work are:

*   Allocation tracking and aliasing in the compiler
*   Pipelined scheduler in the VM for issuing proper command buffers
*   New CPU interpreter that enables lightweight execution on ARM and x86
*   C code generator and API to demonstrate "runtimeless" mode
*   Quantization using the MLIR quantization framework
*   High-level integration and examples when working with TensorFlow 2.0
*   Switching from IREE's XLA-to-SPIR-V backend to the general MLIR SPIR-V
    backend

Things we are interested in but don't yet have in-progress:

*   Ahead-of-time compiled ARM NEON backend (perhaps via
    [SPIRV-LLVM](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/),
    [SPIRV-to-ISPC](https://github.com/GameTechDev/SPIRV-Cross), or some other
    technique)
*   HAL backends for Metal 2 and Direct3D 12
*   Profile-guided optimization support for scheduling feedback

<a name="status"></a>

## Current Status

### Documentation

Coming soon :)

### Build System and CI

*   We support Bazel for builds of all parts of the project.
*   We also maintain a CMake build for a subset of runtime components designed
    to be used in other systems.

### Code and Style

The project is currently _very_ early and a mix of code written prior to a lot
of the more recent ergonomics improvements in MLIR and its tablegen. Future
changes will replace the legacy code style with prettier forms and simplify the
project structure to make it easier to separate the different components. Some
entire portions of the code (such as the CPU interpreter) will likely be dropped
or rewritten. For now, assume churn!

The compiler portions of the code (almost exclusively under `iree/compiler/`)
follows the LLVM style guide and has the same system requirements as MLIR
itself. It general requires a more modern C++ compiler.

The runtime portions vary but most are designed to work with C++11 and use
[Abseil](https://github.com/abseil/abseil-cpp) to bring in future C++14 and
C++17 features.

### Hardware Support

We are mostly targeting Vulkan and Metal on recent mobile devices and as such
have limited our usage of hardware features and vendor extensions to those we
have broad access to there. This is mainly just to keep our focus tight and does
not preclude usage of features outside the standard sets or for other hardware
types (in fact, we have a lot of fun ideas for
`VK_NVX_device_generated_commands` and Metal 2.1's Indirect Command Buffers!).

<a name="dependencies"></a>

## Dependencies

NOTE: during the initial open source release we are still cleaning things up. If
there's weird dependencies/layering that makes life difficult for your
particular use case please file an issue so we can make sure to fix it.

### Compiler

The compiler has several layers that allow scaling the dependencies required
based on the source and target formats. In all cases
[MLIR](https://github.com/tensorflow/mlir) is required and for models not
originating from TensorFlow (or already in XLA HLO format) it is the only
dependency. When targeting the IREE Runtime VM and HAL
[FlatBuffers](https://google.github.io/flatbuffers/) is required for
serialization. Converting from TensorFlow models requires a dependency on
TensorFlow (however only those parts required for conversion).

### Runtime VM

The VM providing dynamic model deployment and advanced scheduling behavior
requires [Abseil](https://github.com/abseil/abseil-cpp) for its common types,
however contributions are welcome to make it possible to replace Abseil with
other libraries via shims/forwarding. The core types used by the runtime
(excluding command line flags and such in tools) are limited to types coming in
future C++ versions (variant, optional, string_view, etc), cheap types
(absl::Span), or simple standard containers (absl::InlinedVector).
[FlatBuffers](https://google.github.io/flatbuffers/) is used to load compiled
modules.

### Runtime HAL

The HAL and the provided implementations (Vulkan, etc) also use
[Abseil](https://github.com/abseil/abseil-cpp). Contributions are welcome to
allow other types to be swapped in. A C99 HAL API is planned for code generation
targets that will use no dependencies.

### Testing and Tooling

[Swiftshader](https://github.com/google/swiftshader) is used to provide fast
hardware-independent testing of the Vulkan and SPIR-V portions of the toolchain.

<a name="license"></a>

## License

IREE is licensed under the terms of the Apache license. See [LICENSE](LICENSE)
for more information.
