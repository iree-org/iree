# Glossary

IREE is a composition of multiple acronym described projects making it
difficult to understand these components interact. Below is a summation
of these projects with the problems they instend to address.

## JAX

[JAX](https://github.com/google/jax) is a ML front-end for writing and
executing Machine Learning models, specifically focusing on Google's
TPUs.

## MLIR

[Multi Level Intermediate Representation](https://mlir.llvm.org/) is
the compiler framework that IREE is built around. Beyond the tooling
this includes a set of common dialects and transformations that IREE
utilizes for it's code generation system.

For general discussion on MLIR see the projects
[discourse](https://discourse.llvm.org/c/mlir/31) group.

## LinAlg

[Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) is an MLIR dialect
that defines how Linalg Algebra operations can be described in a
generalized fashion, including a set of commonly used linear algebra
operations. IREE's codegen defines tensor operations using the Linalg
dialect, and is used to generate the loop structures for the CPU and
GPU backends.

## SPIR-V

[SPIR-V](https://www.khronos.org/spir/) kernel language for expressing
parallel computation typically used for accelerators. It serves as
a hardware agnostic assembly format for distributing complex,
computationally complex programs. It is the preferred method for
shipping platform agnostic binaries to run on GPUs.

## TOSA

[TOSA](https://developer.mlplatform.org/w/tosa/) defines a set of common
tensor operations to most machine learning frameworks. TOSA's defines
a simple intermediate representation for ingesting ML models. This simplifies
model compilation by targetting multiple front-end languages to TOSA's
intermediate IR which guarantees efficient execution across multiple device
types.

IREE uses TOSA as a prioritized ingestion dialect, transforming multiple
ML-platform ingestion formats into a TOSA compatible set of operations.
Changes to the TOSA specification require submitting a proposal on
TOSA's platform development
[page](https://developer.mlplatform.org/w/tosa/#:~:text=Specification%20Contributions)

## TFLite

[TensorFlow lite](https://www.tensorflow.org/lite) is a model format and
execution system for performing on device inference. IREE supports TFLite
flatbuffers translation into [TOSA](#tosa-dialect) operations and compilation
into the LinAlg dialect.
