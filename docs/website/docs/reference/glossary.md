---
hide:
  - tags
tags:
  - JAX
  - PyTorch
  - TensorFlow
icon: octicons/book-16
---

# Glossary

IREE exists in an ecosystem of projects and acts as a bridge between machine
learning frameworks and a variety of hardware platforms. This glossary outlines
some of those projects and technologies.

!!! question - "Something missing?"

    Don't see a project of technology here that you think should be? We welcome
    contributions on [our GitHub page](https://github.com/iree-org/iree)!

## JAX

[JAX](https://github.com/google/jax) is Python framework supporting
high-performance machine learning research by bridging automatic differentiation
and ML compilers like [XLA](https://github.com/openxla/xla) and IREE.

See the
[JAX Integration guide](../guides/ml-frameworks/jax.md) for details on how to
use JAX programs with IREE.

## MLIR

[Multi-Level Intermediate Representation (MLIR)](https://mlir.llvm.org/) is
the compiler framework that IREE is built around. Beyond the tooling
this includes a set of common dialects and transformations that IREE
utilizes for its code generation system.

For general discussion on MLIR see the project's
[discourse](https://discourse.llvm.org/c/mlir/31) forum.

## Linalg

[Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) is an MLIR dialect
that defines Linear Algebra operations in a generalized fashion by modeling
iteration spaces together with compute payloads. Linalg includes a set of
commonly used operations as well as generic interfaces.

IREE uses the Linalg dialect during its code generation pipeline to define
tensor operations then generate loop structures for its various backend targets.

## OpenXLA

[OpenXLA](https://github.com/openxla/community) is a community-driven, open
source ML compiler ecosystem.

IREE interfaces with some of the OpenXLA projects, such as
[StableHLO](#stablehlo).

## PyTorch

[PyTorch](https://pytorch.org/) is an optimized tensor library for deep
learning.

PyTorch uses the [Torch-MLIR](https://github.com/llvm/torch-mlir) project to
interface with projects like IREE. See the
[PyTorch Integration guide](../guides/ml-frameworks/pytorch.md) for details on
how to use PyTorch programs with IREE.

## SPIR-V

[SPIR-V](https://www.khronos.org/spir/) is a shader and kernel intermediate
language for expressing parallel computation typically used for GPUs. It serves
as a hardware agnostic assembly format for distributing complex,
computationally intensive programs.

IREE uses the
[SPIR-V MLIR Dialect](https://mlir.llvm.org/docs/Dialects/SPIR-V/) in its code
generation pipeline for Vulkan and other compute APIs.

## StableHLO

[StableHLO](https://github.com/openxla/stablehlo) is a set of versioned
high-level operations (HLOs) for ML models with backward and forward
compatibility guarantees. StableHLO aims to improve interoperability between
frameworks (such as TensorFlow, JAX, and PyTorch) and ML compilers.

StableHLO has both a
[specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)
and an MLIR dialect.

IREE uses the StableHLO MLIR Dialect as one of its input formats.

## TOSA

[Tensor Operator Set Architecture (TOSA)](https://www.mlplatform.org/tosa)
provides a set of tensor operations commonly employed by Deep Neural Networks.
TOSA defines accuracy and compatibility constraints so frameworks that use it
can trust that applications will produce similar results on a variety of
hardware targets.

TOSA has both a [specification](https://www.mlplatform.org/tosa/tosa_spec.html)
and an [MLIR dialect](https://mlir.llvm.org/docs/Dialects/TOSA/).

IREE uses the TOSA MLIR dialect as one of its input formats.

## TFLite

[TensorFlow Lite (TFLite)](https://www.tensorflow.org/lite) is a library
for deploying models on mobile and other edge devices.

IREE supports running TFLite programs that have been imported into MLIR using
the TOSA dialect. See the
[TFLite Integration guide](../guides/ml-frameworks/tflite.md) for details on how
to use TFLite programs with IREE.

IREE also has bindings for the
[TFLite C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c),
see the
[`runtime/bindings/tflite/`](https://github.com/iree-org/iree/tree/main/runtime/bindings/tflite)
directory for details.
