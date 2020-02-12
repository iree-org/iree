# Getting Started

This document provides an overview of IREE's systems, including entry points to
get started exploring IREE's capabilities.

For information on how to set up a development environment, see
[Getting Started on Windows](getting_started_on_windows.md) and
[Getting Started on Linux](getting_started_on_linux.md).

## Project Code Layout

[iree/](../iree/)

*   Core IREE project

[integrations/](../integrations/)

*   Integrations between IREE and other frameworks, such as TensorFlow

[bindings/](../bindings/)

*   Language and platform bindings, such as Python

[colab/](../colab/)

*   Colab notebooks for interactively using IREE's Python bindings

## IREE Code Layout

[iree/base/](../iree/base/)

*   Common types and utilities used throughout IREE

[iree/compiler/](../iree/compiler/)

*   IREE's MLIR dialects, LLVM compiler passes, module translation code, etc.

[iree/hal/](../iree/hal/)

*   **H**ardware **A**bstraction **L**ayer for IREE's runtime, with
    implementations for hardware and software backends

[iree/schemas/](../iree/schemas/)

*   Shared data storage format definitions, primarily using
    [FlatBuffers](https://google.github.io/flatbuffers/)

[iree/tools/](../iree/tools/)

*   Assorted tools used to optimize, translate, and evaluate IREE

[iree/vm/](../iree/vm/)

*   Bytecode **V**irtual **M**achine used to work with IREE modules and invoke
    IREE functions

## Working with IREE's Components

IREE ingests MLIR in a high-level dialect like
[XLA/HLO](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir/xla),
after which it can perform its own compiler passes to eventually translate the
IR into an 'IREE module', which can be executed via IREE's runtime. IREE
contains programs for running each step in that pipeline under various
configurations (e.g. for tests, with a debugger attached, etc.).

### iree-opt

The `iree-opt` program invokes
[MlirOptMain](https://github.com/llvm/llvm-project/blob/master/mlir/lib/Support/MlirOptMain.cpp)
to run some set of IREE's optimization passes on a provided .mlir input file.
Test .mlir files that are checked in typically include a `RUN` block at the top
of the file that specifies which passes should be performed and if `FileCheck`
should be used to test the generated output.

For example, to run some passes on the
[reshape.mlir](../iree/compiler/Translation/SPIRV/XLAToSPIRV/test/reshape.mlir)
test file with Bazel on Linux, use this command:

```shell
$ bazel run //iree/tools:iree-opt -- \
  -split-input-file \
  -iree-index-computation \
  -simplify-spirv-affine-exprs=false \
  -convert-iree-to-spirv \
  -verify-diagnostics \
  $PWD/iree/compiler/Translation/SPIRV/XLAToSPIRV/test/reshape.mlir
```

### iree-translate

The `iree-translate` program translates from a .mlir input file into an IREE
module.

For example, to translate `simple.mlir` to an IREE module with bazel on Linux,
use this command:

```shell
$ bazel run //iree/tools:iree-translate -- \
  -iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=interpreter-bytecode \
  $PWD/iree/tools/test/simple.mlir \
  -o /tmp/module.fb
```

Custom translations may also be layered on top of `iree-translate` - see
[iree/samples/custom_modules/dialect](../iree/samples/custom_modules/dialect)
for a sample.

### iree-run-module

The `iree-run-module` program takes an already translated IREE module as input
and executes an exported main function using the provided inputs.

This program can be used in sequence with `iree-translate` to translate a .mlir
file to an IREE module and then execute it. Here is an example command that
executes the simple `module.fb` compiled from `simple.mlir` above on the IREE
interpreter

```shell
$ bazel run //iree/tools:iree-run-module -- \
  --input_file=/tmp/module.fb \
  --driver=interpreter \
  --entry_function=abs \
  --inputs="i32=-2"
```

### iree-run-mlir

The `iree-run-mlir` program takes a .mlir file as input, translates it to an
IREE bytecode module, and executes the module.

It is designed for testing and debugging, not production use cases, and
therefore does some additional work that usually must be explicit, like marking
every function as exported by default and running all of them.

For example, to execute the contents of a test .mlir file, use this command:

```shell
$ bazel run //iree/tools:iree-run-mlir -- \
  $PWD/iree/tools/test/simple.mlir \
  --input-value="i32=-2" \
  --iree-hal-target-backends=interpreter-bytecode
```

### iree-dump-module

The `iree-dump-module` program prints the contents of an IREE module FlatBuffer
file.

For example:

```shell
$ bazel run //iree/tools:iree-dump-module -- /tmp/module.fb
```
