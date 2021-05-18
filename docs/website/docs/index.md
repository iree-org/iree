# IREE

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment[^1]) is
an [MLIR](https://mlir.llvm.org/)-based end-to-end compiler that lowers Machine
Learning (ML) models to a unified IR optimized for real-time inference on
mobile/edge devices against heterogeneous hardware accelerators. IREE also
provides flexible deployment solutions for its compiled ML models.

## Key features

- [x] Ahead-of-time compilation of scheduling and execution logic together
- [x] Support for dynamic shapes, flow control, streaming, and other advanced
      model features
- [x] Optimized for many CPU and GPU architectures
- [x] Low overhead, pipelined execution for efficient power and resource usage
- [x] Binary size as low as 30KB on embedded systems
- [x] Debugging and profiling support

## Support matrix

IREE supports importing from a variety of ML frameworks:

- [x] TensorFlow
- [x] TensorFlow Lite
- [x] JAX
- [ ] PyTorch (planned)
- [ ] ONNX (hoped for)

The IREE compiler tools run on :fontawesome-brands-linux: Linux,
:fontawesome-brands-windows: Windows, and :fontawesome-brands-apple: macOS
and can generate efficient code for a variety of runtime platforms:

- [x] Linux
- [x] Windows
- [x] Android
- [ ] macOS (planned)
- [ ] iOS (planned)
- [ ] Bare metal (planned)
- [ ] WebAssembly (planned)

and architectures:

- [x] ARM
- [x] x86
- [x] RISC-V

Support for hardware accelerators and APIs is also included:

- [x] Vulkan
- [x] CUDA
- [ ] Metal (planned)
- [ ] WebGPU (planned)

## Project architecture

IREE adopts a _holistic_ approach towards ML model compilation: the IR produced
contains both the _scheduling_ logic, required to communicate data dependencies
to low-level parallel pipelined hardware/API like
[Vulkan](https://www.khronos.org/vulkan/), and the _execution_ logic, encoding
dense computation on the hardware in the form of hardware/API-specific binaries
like [SPIR-V](https://www.khronos.org/spir/).

<!-- TODO(scotttodd): text borders so this is easier to read on dark backgrounds -->
<!-- TODO(scotttodd): use the same .svg as the README (point README at this path) -->
![IREE Architecture](assets/images/iree_architecture.svg)

## Workflow overview

Using IREE involves these general steps:

1. **Import your model**

    Work in your framework of choice, then run your model through one of IREE's
    import tools.

2. **Select your deployment configuration**

    Identify your target platform, accelerator(s), and other constraints.

3. **Compile your model**

    Compile through IREE, picking compilation targets based on your
    deployment configuration.

4. **Run your model**

    Use IREE's runtime components to execute your compiled model.

### Importing models from ML frameworks

IREE supports importing models from a growing list of ML frameworks and model
formats:

* [TensorFlow](ml-frameworks/tensorflow.md)
* [TensorFlow Lite](ml-frameworks/tensorflow-lite.md)
* [JAX](ml-frameworks/jax.md)

### Selecting deployment configurations

IREE provides a flexible set of tools for various deployment scenarios.
Fully featured environments can use IREE for dynamic model deployments taking
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
[GPU execution](deployment-configurations/gpu-vulkan.md) using Vulkan generates
SPIR-V kernels and Vulkan API calls. For
[CPU execution](deployment-configurations/cpu-dylib.md), native code with
static or dynamic linkage and the associated function calls are generated.

### Running models

IREE offers a low level C API, as well as several specialized sets of
_bindings_ for running IREE models using other languages:

* [C API](bindings/c-api.md)
* [Python](bindings/python.md)
* [TensorFlow Lite](bindings/tensorflow-lite.md)

## Communication channels

*   :fontawesome-brands-github:
    [GitHub issues](https://github.com/google/iree/issues): Feature requests,
    bugs, and other work tracking
*   :fontawesome-brands-discord:
    [IREE Discord server](https://discord.gg/26P4xW4): Daily development
    discussions with the core team and collaborators
*   :fontawesome-solid-users: [iree-discuss email list](https://groups.google.com/forum/#!forum/iree-discuss):
    Announcements, general and low-priority discussion

## Roadmap

IREE is in the early stages of development and is not yet ready for broad
adoption. Check out the
[long-term design roadmap](https://github.com/google/iree/blob/main/docs/design_roadmap.md)
to get a sense of where we're headed.

We plan on a quarterly basis using [OKRs](https://en.wikipedia.org/wiki/OKR).
Review our latest
[objectives](https://github.com/google/iree/blob/main/docs/objectives.md) to
see what we're up to.

We use [GitHub Projects](https://github.com/google/iree/projects) to track
progress on IREE components and specific efforts and
[GitHub Milestones](https://github.com/google/iree/milestones) to track the
work associated with plans for each quarter.

[^1]:
  Pronounced "eerie" and often styled with the :iree-ghost: emoji

*[IR]: Intermediate Representation
