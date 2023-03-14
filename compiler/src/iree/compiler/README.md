# IREE Compiler

This directory contains IREE's main [MLIR](https://mlir.llvm.org/)-based
compiler.

Paths of particular interest include:

```
└── compiler/
    ├── API/         (C and Python APIs)
    ├── Bindings/    (used to generate different ABI bindings)
    ├── Codegen/     (device code generation for assorted APIs)
    ├── Dialect/
    │   ├── Flow/    (tensor program modeling and compute workload partitioning)
    │   ├── HAL/     (Hardware Abstraction Layer for buffer and execution management)
    │   │   └── Target/
    │   │       ├── LLVM/
    │   │       ├── VMVX/
    │   │       ├── VulkanSPIRV/
    │   │       └── etc.
    │   ├── Stream/  (device placement and asynchronous scheduling)
    │   ├── Util/    (common types across other IREE dialects)
    │   └── VM/      (abstract Virtual Machine)
    ├── InputConversion/  (conversions from frontend/input dialects)
    └── Translation/      (translation pipeline definitions)

```

Noteworthy compiler components _not_ included here include:

```
├── integrations/
│   └── tensorflow/
│       └── iree_tf_compiler/  (passes and tools for importing TensorFlow programs)
└── llvm-external-projects/
    ├── iree-compiler-api/     (C and Python APIs for the compiler)
    └── iree-dialects/         (public IREE dialects for other projects to target)
```

The general flow of data from a frontend framework down to a compiled program
is:

```
program written in framework

    |
    |  import tool
    V

imported .mlir file

    |
    |  main IREE compiler
    V

compiled program
```

where import tools live outside of the compiler/ directory (and possibly
outside of the IREE repo itself).

Refer to IREE's
[presentations and talks](../../../../README.md#presentations-and-talks) and this
architecture diagram for details on how the pieces fit together:

![IREE Architecture](../../../../docs/website/docs/assets/images/iree_architecture.svg)

## Coding Style

Like the rest of the project, the compiler/ directory uses clang-format with the
`BasedOnStyle: Google` option. However, since the code in this directory makes
heavy use of LLVM and MLIR, it also adheres to LLVM style for variable naming
(use `int variableName` instead of `int variable_name` and pointer alignment
(use `int *a` instead of `int* a`).

Read more:

* https://google.github.io/styleguide/cppguide.html
* https://llvm.org/docs/CodingStandards.html
* https://clang.llvm.org/docs/ClangFormatStyleOptions.html
