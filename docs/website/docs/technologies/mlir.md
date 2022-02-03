
# MLIR

[MLIR](https://mlir.llvm.org/) is the compiler framework that IREE is built
around. Beyond the tooling this includes a set of common Dialects and
transforms that IREE uses for it's code generation.

Any required changes to MLIR should be upstreamed by following the MLIR
contribution [guide](https://mlir.llvm.org/getting_started/Contributing/).
IREE periodically syncs to MLIR's HEAD so there may be a noticeable delay
between contributing to MLIR and having the feature available at IREE head.

For general discussion on MLIR see the projects
[discourse](https://discourse.llvm.org/c/mlir/31) group.

## Dialects

MLIR contains a set of common dialects useful for IREE. The list below is not
exhaustive, and instead describes how these dialects integrate with IREE.

### Linalg

The [Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) dialect defines how
Linalg Algebra operations can be described in a generalized fashion, including
a set of commonly used linear algebra operations. IREE's codegen defines tensor
operations using the Linalg dialect, and is used to generate the loop
structures for IREE's CPU and GPU backends.

### TOSA

[TOSA](https://developer.mlplatform.org/w/tosa/) is a standardized set of tensor
operations common in machine learning applications.

See the separate [TOSA](./tosa.md) section for more details.
