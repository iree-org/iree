# IREE LLVM-based projects

Projects in this tree are targeted for interop with upstream LLVM/MLIR and
related projects. They follow LLVM standards, build system, API design and
packaging conventions. In general they are either:

* Meant to be used as an `LLVM_EXTERNAL_PROJECT`.
* A standalone project based on the LLVM build system.

We publish projects here when they are meant to consume or interoperate at a
build/source level with other projects in the ecosystem.

## Exceptions to LLVM coding standards

* File headers follow IREE conventions for copyright/license banner.
