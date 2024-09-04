---
hide:
  - navigation
---

# IREE

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment[^1]) is
an [MLIR](https://mlir.llvm.org/)-based end-to-end compiler and runtime that
lowers Machine Learning (ML) models to a unified IR that scales up to meet the
needs of the datacenter and down to satisfy the constraints and special
considerations of mobile and edge deployments.

## Key features

<div class="grid cards" markdown>

- :material-clock-fast: **Ahead-of-time compilation**

    ---

    Scheduling and execution logic are compiled together

    [:octicons-arrow-right-24: Project architecture](#project-architecture)

- :octicons-rocket-24: **Support for advanced model features**

    ---

    Dynamic shapes, flow control, streaming, and more

    [:octicons-arrow-right-24: Importing from ML frameworks](#importing-models-from-ml-frameworks)

- :octicons-server-24: **Designed for CPUs, GPUs, and other accelerators**

    ---

    First class support for many popular devices and APIs

    [:octicons-arrow-right-24: Deployment configurations](#selecting-deployment-configurations)

- :fontawesome-solid-chart-gantt: **Low overhead, pipelined execution**

    ---

    Efficient power and resource usage on server and edge devices

    [:octicons-arrow-right-24: Benchmarking](./developers/performance/benchmarking.md)

- :fontawesome-regular-floppy-disk: **Binary size as low as 30KB on embedded systems**

    ---

    [:octicons-arrow-right-24: Running on bare-metal](./guides/deployment-configurations/bare-metal.md)

- :fontawesome-solid-magnifying-glass: **Debugging and profiling support**

    ---

    [:octicons-arrow-right-24: Profiling with Tracy](./developers/performance/profiling-with-tracy.md)

</div>

## Support matrix

IREE supports importing from a variety of ML frameworks:

- [x] JAX
- [x] ONNX
- [x] PyTorch
- [x] TensorFlow and TensorFlow Lite

The IREE compiler tools run on :fontawesome-brands-linux: Linux,
:fontawesome-brands-windows: Windows, and :fontawesome-brands-apple: macOS
and can generate efficient code for a variety of runtime platforms:

- [x] Linux
- [x] Windows
- [x] macOS
- [x] Android
- [x] iOS
- [x] Bare metal
- [ ] WebAssembly (experimental)

and architectures:

- [x] ARM
- [x] x86
- [x] RISC-V

Support for hardware accelerators and APIs is also included:

- [x] Vulkan
- [x] CUDA
- [x] ROCm
- [x] Metal (for Apple silicon devices)
- [ ] AMD AIE (experimental)
- [ ] WebGPU (experimental)

## Project architecture

IREE adopts a _holistic_ approach towards ML model compilation: the IR produced
contains both the _scheduling_ logic, required to communicate data dependencies
to low-level parallel pipelined hardware/API like
[Vulkan](https://www.khronos.org/vulkan/), and the _execution_ logic, encoding
dense computation on the hardware in the form of hardware/API-specific binaries
like [SPIR-V](https://www.khronos.org/spir/).

![IREE Architecture](./assets/images/iree_architecture_dark.svg#gh-dark-mode-only)
![IREE Architecture](./assets/images/iree_architecture.svg#gh-light-mode-only)

## Workflow overview

Using IREE involves the following general steps:

1. **Import your model**

    Develop your program using one of the
    [supported frameworks](./guides/ml-frameworks/index.md), then import into
    IREE

2. **Select your [deployment configuration](./guides/deployment-configurations/index.md)**

    Identify your target platform, accelerator(s), and other constraints

3. **Compile your model**

    Compile through IREE, picking settings based on your deployment
    configuration

4. **Run your model**

    Use IREE's runtime components to execute your compiled model

### Importing models from ML frameworks

IREE supports importing models from a growing list of
[ML frameworks](./guides/ml-frameworks/index.md) and model formats:

* [:simple-python: JAX](./guides/ml-frameworks/jax.md)
* [:simple-onnx: ONNX](./guides/ml-frameworks/onnx.md)
* [:simple-pytorch: PyTorch](./guides/ml-frameworks/pytorch.md)
* [:simple-tensorflow: TensorFlow](./guides/ml-frameworks/tensorflow.md) and
  [:simple-tensorflow: TensorFlow Lite](./guides/ml-frameworks/tflite.md)

### Selecting deployment configurations

IREE provides a flexible set of tools for various
[deployment scenarios](./guides/deployment-configurations/index.md). Fully
featured environments can use IREE for dynamic model deployments taking
advantage of multi-threaded hardware, while embedded systems can bypass IREE's
runtime entirely or interface with custom accelerators.

* What platforms are you targeting? Desktop? Mobile? An embedded system?
* What hardware should the bulk of your model run on? CPU? GPU?
* How fixed is your model itself? Can the weights be changed? Do you want
  to support loading different model architectures dynamically?

IREE supports the full set of these configurations using the same underlying
technology.

### Compiling models

Model compilation is performed ahead-of-time on a _host_ machine for any
combination of _targets_. The compilation process converts from layers and
operators used by high level frameworks down into optimized native code and
associated scheduling logic.

For example, compiling for
[GPU execution](./guides/deployment-configurations/gpu-vulkan.md) using Vulkan generates
SPIR-V kernels and Vulkan API calls. For
[CPU execution](./guides/deployment-configurations/cpu.md), native code with
static or dynamic linkage and the associated function calls are generated.

### Running models

IREE offers a low level C API, as well as several sets of
[API bindings](./reference/bindings/index.md) for compiling and running programs
using various languages.

## Communication channels

* :fontawesome-brands-github:
  [GitHub issues](https://github.com/iree-org/iree/issues): Feature requests,
  bugs, and other work tracking
* :fontawesome-brands-discord:
  [IREE Discord server](https://discord.gg/wEWh6Z9nMU): Daily development
  discussions with the core team and collaborators
* :fontawesome-solid-bullhorn: (New) [iree-announce email list](https://lists.lfaidata.foundation/g/iree-announce):
  Announcements
* :fontawesome-solid-envelope: (New) [iree-technical-discussion email list](https://lists.lfaidata.foundation/g/iree-technical-discussion):
  General and low-priority discussion
* :fontawesome-solid-envelope: (Legacy) [iree-discuss email list](https://groups.google.com/forum/#!forum/iree-discuss):
  Announcements, general and low-priority discussion

## Roadmap

IREE is in the early stages of development and is not yet ready for broad
adoption. We use both
[GitHub Projects](https://github.com/iree-org/iree/projects) and
[GitHub Milestones](https://github.com/iree-org/iree/milestones) to track
progress.

[^1]:
  Pronounced "eerie" and often styled with the :iree-ghost: emoji

*[IR]: Intermediate Representation
