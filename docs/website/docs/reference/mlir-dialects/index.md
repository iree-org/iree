---
icon: simple/llvm
---

# MLIR dialects and passes

These pages contain automatically generated documentation for the MLIR dialects
defined in the IREE repository. IREE also makes extensive use of dialects and
passes from the upstream MLIR repository, which are documented at
[https://mlir.llvm.org/docs/Dialects/](https://mlir.llvm.org/docs/Dialects/) and
[https://mlir.llvm.org/docs/Passes/](https://mlir.llvm.org/docs/Passes/).

## IREE internal dialects

These dialects are an implementation detail of the IREE compiler, though they
can be used by plugins and other advanced integrations. The sources for most of
these dialects can be found in the
[`iree/compiler/Dialect/` directory](https://github.com/iree-org/iree/tree/main/compiler/src/iree/compiler/Dialect).

Dialect                     | Description
--------------------------- | -----------
[Check](./Check.md)         | Defines assertions for IREE tests
[Encoding](./Encoding.md)   | Tensor encoding attributes and related ops
[Flow](./Flow.md)           | Models execution data flow and partitioning
[HAL](./HAL.md)             | Represents operations against the IREE HAL[^1]
[HAL/Inline](./HALInline.md) | Inline HAL interop runtime module dialect
[HAL/Loader](./HALLoader.md) | HAL inline executable loader runtime module dialect
[IO/Parameters](./IOParameters.md) | External parameter resource management APIs
[IREECodegen](./IREECodegen.md) | Common functionality used by IREE code generation
[IREECPU](./IREECPU.md) | Common functionality used by CPU and VMVX focused IREE code generation
[IREEGPU](./IREEGPU.md) | Common functionality used by GPU focused IREE code generation
[IREEVectorExt](./IREEVectorExt.md) | Extensions to the Vector dialect for specific operations
[LinalgExt](./LinalgExt.md) | Extensions to the Linalg dialect for specific operations
[PCF](./PCF.md) | A dialect designed to model parallel control flow.
[Stream](./Stream.md)       | Model execution partitioning and scheduling
[TensorExt](./TensorExt.md) | Extensions to the Tensor dialect for specific operations
[Util](./Util.md)           | Types and ops common across IREE subdialects
[VM](./VM.md)               | Represents operations against an abstract virtual machine
[VMVX](./VMVX.md)           | Virtual Machine Vector Extensions

[^1]: Hardware Abstraction Layer
