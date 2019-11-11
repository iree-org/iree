<!--
  Copyright 2019 Google LLC

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

       https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# Getting Started

This document provides an overview of the systems in IREE, including entry
points to get started exploring IREE's capabilities.

For information on how to set up a development environment, see
[Getting Started on Windows](getting_started_on_windows.md) and
[Getting Started on Linux](getting_started_on_linux.md).

## Code Layout

[base/](../iree/base/)

*   Common types and utilities used throughout IREE

[compiler/](../iree/compiler/)

*   IREE's MLIR dialect, LLVM compiler passes, IREE module translation, etc.

[hal/](../iree/hal/)

*   **H**ardware **A**bstraction **L**ayer for IREE's runtime, containing
    implementations for different hardware and software backends

[rt/](../iree/rt/)

*   **R**un**t**ime API for interfacing with IREE

[schemas/](../iree/schemas/)

*   Shared data storage format definitions, primarily using
    [FlatBuffers](https://google.github.io/flatbuffers/)

[tools/](../iree/tools/)

*   Assorted tools used to optimize, translate, and evaluate IREE, including
    IREE's debugger

[vm/](../iree/vm/)

*   Bytecode **V**irtual **M**achine used to work with IREE modules and provide
    an interface for hosting applications to invoke IREE functions

## Working with IREE's Components

IREE ingests MLIR in a high-level dialect like
[XLA/HLO](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir/xla),
after which it can perform its own compiler passes to eventually translate the
IR into an 'IREE module', which can be executed via IREE's runtime. IREE
contains programs for running each step in that pipeline under various
configurations (e.g. for tests, with a debugger attached, etc.).

### iree-opt

The `iree-opt` program invokes
[MlirOptMain](https://github.com/tensorflow/mlir/blob/master/lib/Support/MlirOptMain.cpp)
to run some set of IREE's optimization passes on a provided .mlir input file.
Test .mlir files that are checked in typically include a `RUN` block at the top
of the file that specifies which passes should be performed and if `FileCheck`
should be used to test the generated output.

For example, to run some passes on the
[reshape.mlir](../iree/compiler/Translation/SPIRV/test/reshape.mlir) test file
with Bazel on Linux, use this command:

```shell
$ bazel run //iree/tools:iree-opt -- \
  -split-input-file \
  -convert-iree-to-spirv \
  -simplify-spirv-affine-exprs=false \
  -verify-diagnostics \
  $PWD/iree/compiler/Translation/SPIRV/test/reshape.mlir
```

### iree-translate

The `iree-translate` program invokes
[mlir-translate](https://github.com/tensorflow/mlir/blob/master/tools/mlir-translate/mlir-translate.cpp)
to translate from a .mlir input file into an IREE module.

For example, to translate `simple_compute_test.mlir` to an IREE module with
Bazel on Linux, use this command:

```shell
$ bazel run //iree/tools:iree-translate -- \
  -mlir-to-iree-module \
  $PWD/iree/samples/hal/simple_compute_test.mlir \
  -o /tmp/module.fb
```

### run_module

The `run_module` program takes an already translated IREE module as input and
executes an exported main function using the provided inputs.

This program can be used in sequence with `iree-translate` to translate a .mlir
file to an IREE module and then execute it. Here is an example command that runs
the `simple_mul` function in `simple_compute_test.mlir`.

```shell
$ bazel build -c opt //iree/tools:iree-translate //iree/tools:run_module
$ ./bazel-bin/iree/tools/run_module \
  --main_module=/tmp/module.fb \
  --main_function=simple_mul \
  --input_values="4xf32=1 2 3 4
                  4xf32=5 6 7 8"
```

### iree-run-mlir

The `iree-run-mlir` program takes a .mlir file as input, translates it to an
IREE bytecode module, and executes the module.

For example, to execute the contents of a test .mlir file, use this command:

```shell
$ bazel run //iree/tools:iree-run-mlir -- $PWD/test/e2e/scalars.mlir
```
