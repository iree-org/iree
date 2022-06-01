# Glossary

IREE exists in an ecosystem of projects, each using their own terminology
and interacting in various ways. Below is a summation of these projects
with the problems they are built to address.

## JAX

[JAX](https://github.com/google/jax) is a front-end for writing and
executing Machine Learning models, including support for Google
Cloud TPUs.

## MLIR

[Multi Level Intermediate Representation](https://mlir.llvm.org/) is
the compiler framework that IREE is built around. Beyond the tooling
this includes a set of common dialects and transformations that IREE
utilizes for its code generation system.

For general discussion on MLIR see the project's
[discourse](https://discourse.llvm.org/c/mlir/31) group.

## LinAlg

[Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) is an MLIR dialect
that defines how Linear Algebra operations can be described in a
generalized fashion, including a set of commonly used operations.
IREE's code generation defines tensor operations using the Linalg
dialect, then uses it to generate the loop structures for the CPU and
GPU backends.

## SPIR-V

[SPIR-V](https://www.khronos.org/spir/) is a shader and kernel intermediate
language for expressing parallel computation typically used for GPUs. It serves
as a hardware agnostic assembly format for distributing complex,
computationally intensive programs. It is the preferred method for
shipping platform agnostic binaries to run on GPUs.

## TOSA

The [TOSA](https://developer.mlplatform.org/w/tosa/) specification defines a
set of common tensor operations to most machine learning frameworks.
This simplifies model compilation as separate front-end frameworks can target
TOSA's intermediate representation without compromising on the ability to
achieve efficient execution across multiple device types.

IREE uses the TOSA MLIR dialect as a prioritized ingestion format, transforming
multiple ML-platform ingestion formats into a TOSA compatible set of operations.

Changes to the TOSA specification require submitting a proposal on TOSA's
[platform development page](https://developer.mlplatform.org/w/tosa/#:~:text=Specification%20Contributions)

## TFLite

[TensorFlow lite](https://www.tensorflow.org/lite) is a model format and
execution system for performing on device inference. IREE supports TFLite
FlatBuffers translation into [TOSA](#tosa) operations and compilation
into the LinAlg dialect.
