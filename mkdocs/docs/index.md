# IREE

IREE (**I**ntermediate **R**epresentation **E**xecution **E**nvironment[^1]) is
an [MLIR](https://mlir.llvm.org/)-based end-to-end compiler that lowers Machine
Learning (ML) models to a unified IR optimized for real-time mobile/edge
inference against heterogeneous hardware accelerators. IREE also provides
flexible deployment solutions for its compiled ML models.

<!-- TODO(??): more motivation, key features, supported platforms, binary size, etc. -->

## Introduction

IREE adopts a _holistic_ approach towards ML model compilation: the IR produced
contains both the _scheduling_ logic, required to communicate data dependencies
to low-level parallel pipelined hardware/API like
[Vulkan](https://www.khronos.org/vulkan/), and the _execution_ logic, encoding
dense computation on the hardware in the form of hardware/API-specific binaries
like [SPIR-V](https://www.khronos.org/spir/).

The architecture of IREE is best illustrated by the following picture:

<!-- TODO(scotttodd): text borders so this is easier to read on dark backgrounds -->
<!-- TODO(scotttodd): use the same .svg as the README (point README at this path) -->
![IREE Architecture](assets/images/iree_architecture.svg)

## Workflow overview

Using IREE involves these general steps:

1. **Import your model**

    Work in your framework of choice, then run your model through one of IREE's
    import tools.

2. **Select your deployment configuration**

    Identify your target platform, accelerator, and other constraints.

3. **Compile your model**

    Compile through IREE, picking a compilation target based on your
    deployment configuration.

4. **Run your model**

    Use IREE's runtime components to execute your compiled model.

### Importing models from ML frameworks

IREE supports importing models from multiple ML frameworks, including
TensorFlow, TensorFlow Lite, and JAX.

<!-- TODO(scotttodd): mention other properties of the integrations? utils/runtime/decorators -->

<!-- TODO(#5555): rename "frontends" to "ML frameworks"? -->
See the "Frontends" pages for guidance on working with each framework and
model format.

### Selecting deployment configurations

IREE provides a flexible set of tools for various deployment scenarios.

* What hardware should the bulk of your model run on? CPU? GPU?
* What platforms are you targeting? Desktop? Mobile? An embedded system?
* How fixed is your model itself? Can the weights be changed? Do you want
  to support loading different model architectures dynamically?

IREE supports the full matrix of these configurations using the same
underlying technology.

### Compiling models

!!! todo

    Overview for compiling models

<!-- (WIP) -->

<!-- Input -> MLIR dialects -> targets -->

<!-- compile on host machine, ahead of time -->

<!-- scheduling and execution -->

### Running models

!!! todo

    Overview for running models

<!-- (WIP) -->

<!-- runtime for dynamic deployment -->

<!-- runtime for bare metal -->

<!-- runtime for remote execution -->

<!-- runtime advanced features -->

<!-- virtual machine + HAL -->

## How IREE works

Being compilation-based means IREE does not have a traditional runtime that
dispatches "ops" to their fat kernel implementations. What IREE provides is a
toolbox for different deployment scenarios. It scales from running generated
code on a particular API (such as emitting C code calling external DSP kernels),
to a HAL (**H**ardware **A**bstraction **L**ayer) that allows the same generated
code to target multiple APIs (like Vulkan and Direct3D 12), to a full VM
allowing runtime model loading for flexible deployment options and heterogeneous
execution.

<!-- TODO(benvanik): expand on this in enough detail to differentiate IREE -->

## Communication Channels

*   [GitHub Issues](https://github.com/google/iree/issues): Preferred for
    technical issues and coordination on upcoming features.
*   [Google IREE Discord Server](https://discord.gg/26P4xW4): The core team and
    collaborators discuss daily development here; good for low-latency
    communication.
*   [Google Groups Email List](https://groups.google.com/forum/#!forum/iree-discuss):
    Good for general and low-priority discussion.

[^1]:
  Pronounced "eerie" and often styled with the :ghost: emoji

*[IR]: Intermediate Representation
