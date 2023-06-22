# MLIR dialects

These pages contain automatically generated documentation for the MLIR dialects
defined in the IREE repository. IREE also makes extensive use of dialects from
the upstream MLIR repository, which are documented at
[https://mlir.llvm.org/docs/Dialects/](https://mlir.llvm.org/docs/Dialects/).

Dialect                     | Description
--------------------------- | -----------
[Check](./Check.md)         | Defines assertions for IREE tests
[Flow](./Flow.md)           | Models execution data flow and partitioning
[HAL](./HAL.md)             | Represents operations against the IREE HAL[^1]
[HALInline](./HALInline.md) | Inline HAL interop runtime module dialect
[HALLoader](./HALLoader.md) | HAL inline executable loader runtime module dialect
[Stream](./Stream.md)       | Model execution partitioning and scheduling
[Util](./Util.md)           | Types and ops common across IREE subdialects
[VM](./VM.md)               | Represents operations against an abstract virtual machine
[VMVX](./VMVX.md)           | Virtual Machine Vector Extensions

[^1]: Hardware Abstraction Layer
