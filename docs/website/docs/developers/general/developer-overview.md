---
icon: octicons/book-16
---

# Developer overview

This guide provides an overview of IREE's project structure and main tools for
developers.

## Project code layout

* [/compiler/](https://github.com/iree-org/iree/blob/main/compiler/):
  MLIR dialects, LLVM compiler passes, module translation code, etc.
    * [bindings/](https://github.com/iree-org/iree/blob/main/compiler/bindings/):
    Python and other language bindings
* [/runtime/](https://github.com/iree-org/iree/tree/main/runtime/):
  Standalone runtime code including the VM and HAL drivers
    * [bindings/](https://github.com/iree-org/iree/tree/main/runtime/bindings/):
    Python and other language bindings
* [/integrations/](https://github.com/iree-org/iree/blob/main/integrations/):
  Integrations between IREE and other frameworks, such as TensorFlow
* [/tests/](https://github.com/iree-org/iree/blob/main/tests/):
  Tests for full compiler->runtime workflows
* [/tools/](https://github.com/iree-org/iree/blob/main/tools/):
  Developer tools (`iree-compile`, `iree-run-module`, etc.)
* [/samples/](https://github.com/iree-org/iree/blob/main/samples/): Also see the
  separate <https://github.com/iree-org/iree-experimental> repository

## IREE compiler code layout

* [API/](https://github.com/iree-org/iree/tree/main/compiler/src/iree/compiler/API):
  Public C API
* [Codegen/](https://github.com/iree-org/iree/tree/main/compiler/src/iree/compiler/Codegen):
  Code generation for compute kernels
* [Dialect/](https://github.com/iree-org/iree/tree/main/compiler/src/iree/compiler/Dialect):
  MLIR dialects (`Flow`, `HAL`, `Stream`, `VM`, etc.)
* [InputConversion/](https://github.com/iree-org/iree/tree/main/compiler/src/iree/compiler/InputConversion):
  Conversions from input dialects and preprocessing

## IREE runtime code layout

* [base/](https://github.com/iree-org/iree/blob/main/runtime/src/iree/base/):
  Common types and utilities used throughout the runtime
* [hal/](https://github.com/iree-org/iree/blob/main/runtime/src/iree/hal/):
  **H**ardware **A**bstraction **L**ayer for IREE's runtime, with
  implementations for hardware and software backends
* [schemas/](https://github.com/iree-org/iree/blob/main/runtime/src/iree/schemas/):
  Data storage format definitions, primarily using
  [FlatBuffers](https://google.github.io/flatbuffers/)
* [task/](https://github.com/iree-org/iree/blob/main/runtime/src/iree/task/):
  System for running tasks across multiple CPU threads
* [tooling/](https://github.com/iree-org/iree/blob/main/runtime/src/iree/tooling/):
  Utilities for tests and developer tools, not suitable for use as-is in
  downstream applications
* [vm/](https://github.com/iree-org/iree/blob/main/runtime/src/iree/vm/):
  Bytecode **V**irtual **M**achine used to work with IREE modules and invoke
  IREE functions

## Developer tools

IREE's core compiler accepts programs in supported input MLIR dialects (e.g.
`stablehlo`, `tosa`, `linalg`). Import tools and APIs may be used to convert
from framework-specific formats like TensorFlow
[SavedModel](https://www.tensorflow.org/guide/saved_model) to MLIR modules.
While programs are ultimately compiled down to modules suitable for running on
some combination of IREE's target deployment platforms, IREE's developer tools
can run individual compiler passes, translations, and other transformations step
by step.

### iree-opt

`iree-opt` is a tool for testing IREE's compiler passes. It is similar to
[mlir-opt](https://github.com/llvm/llvm-project/tree/main/mlir/tools/mlir-opt)
and runs sets of IREE's compiler passes on `.mlir` input files. See "conversion"
in [MLIR's Glossary](https://mlir.llvm.org/getting_started/Glossary/#conversion)
for more information. Transformations performed by `iree-opt` can range from
individual passes performing isolated manipulations to broad pipelines that
encompass a sequence of steps.

Test `.mlir` files that are checked in typically include a `RUN` block at the
top of the file that specifies which passes should be performed and if
`FileCheck` should be used to test the generated output.

Here's an example of a small compiler pass running on a
[test file](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/Util/Transforms/test/drop_compiler_hints.mlir):

```shell
$ ../iree-build/tools/iree-opt \
  --split-input-file \
  --mlir-print-ir-before-all \
  --iree-util-drop-compiler-hints \
  $PWD/compiler/src/iree/compiler/Dialect/Util/Transforms/test/drop_compiler_hints.mlir
```

For a more complex example, here's how to run IREE's complete transformation
pipeline targeting the VMVX backend on the
[fullyconnected.mlir](https://github.com/iree-org/iree/blob/main/tests/e2e/stablehlo_models/fullyconnected.mlir)
model file:

```shell
$ ../iree-build/tools/iree-opt \
  --iree-transformation-pipeline \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  $PWD/tests/e2e/stablehlo_models/fullyconnected.mlir
```

### iree-compile

`iree-compile` is IREE's main compiler driver for generating binaries from
supported input MLIR assembly.

For example, to translate `simple.mlir` to an IREE module:

```shell
$ ../iree-build/tools/iree-compile \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  $PWD/samples/models/simple_abs.mlir \
  -o /tmp/simple_abs_vmvx.vmfb
```

### iree-run-module

The `iree-run-module` program takes an already translated IREE module as input
and executes an exported function using the provided inputs.

This program can be used in sequence with `iree-compile` to translate a
`.mlir` file to an IREE module and then execute it. Here is an example command
that executes the simple `simple_abs_vmvx.vmfb` compiled from `simple_abs.mlir`
above on IREE's local-task CPU device:

```shell
$ ../iree-build/tools/iree-run-module \
  --module=/tmp/simple_abs_vmvx.vmfb \
  --device=local-task \
  --function=abs \
  --input=f32=-2
```

Input scalars are passed as `value` and input buffers are passed as
`[shape]xtype=[value]`.

* Input buffers may also be read from raw binary files or Numpy npy files.

MLIR type | Description | Input example
-- | -- | --
`i32` | Scalar | `--input=1234`
`tensor<i32>` | 0-D tensor | `--input=i32=1234`
`tensor<1xi32>` | 1-D tensor (shape [1]) | `--input=1xi32=1234`
`tensor<2xi32>` | 1-D tensor (shape [2]) | `--input="2xi32=12 34"`
`tensor<2x3xi32>` | 2-D tensor (shape [2, 3]) | `--input="2x3xi32=[1 2 3][4 5 6]"`

???+ example "Other usage examples"

    See these test files for advanced usage examples:

    <!-- TODO(scotttodd): switch these to 'mlir' syntax when available -->

    === "Basic tests"

        Source file: [`tools/test/iree-run-module.mlir`](https://github.com/iree-org/iree/tree/main/tools/test/iree-run-module.mlir)

        ```c++ title="tools/test/iree-run-module.mlir" linenums="1"
        --8<-- "tools/test/iree-run-module.mlir"
        ```

    === "Inputs"

        Source file: [`tools/test/iree-run-module-inputs.mlir`](https://github.com/iree-org/iree/tree/main/tools/test/iree-run-module-inputs.mlir)

        ```c++ title="tools/test/iree-run-module-inputs.mlir" linenums="1"
        --8<-- "tools/test/iree-run-module-inputs.mlir"
        ```

    === "Outputs"

        Source file: [`tools/test/iree-run-module-outputs.mlir`](https://github.com/iree-org/iree/tree/main/tools/test/iree-run-module-outputs.mlir)

        ```c++ title="tools/test/iree-run-module-outputs.mlir" linenums="1"
        --8<-- "tools/test/iree-run-module-outputs.mlir"
        ```

    === "Expected"

        Source file: [`tools/test/iree-run-module-expected.mlir`](https://github.com/iree-org/iree/tree/main/tools/test/iree-run-module-expected.mlir)

        ```c++ title="tools/test/iree-run-module-expected.mlir" linenums="1"
        --8<-- "tools/test/iree-run-module-expected.mlir"
        ```

### iree-check-module

The `iree-check-module` program takes an already translated IREE module as input
and executes it as a series of
[googletest](https://github.com/google/googletest) tests. This is the test
runner for the IREE [check framework](./testing-guide.md#iree-core-end-to-end-e2e-tests).

```shell
$ ../iree-build/tools/iree-compile \
  --iree-input-type=stablehlo \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  $PWD/tests/e2e/stablehlo_ops/abs.mlir \
  -o /tmp/abs.vmfb
```

```shell
$ ../iree-build/tools/iree-check-module \
  --device=local-task \
  --module=/tmp/abs.vmfb
```

### iree-run-mlir

The `iree-run-mlir` program takes a `.mlir` file as input, translates it to an
IREE bytecode module, and executes the module.

It is designed for testing and debugging, not production uses, and therefore
does some additional work that usually must be explicit, like marking every
function as exported by default and running all of them.

For example, to execute the contents of
[samples/models/simple_abs.mlir](https://github.com/iree-org/iree/blob/main/samples/models/simple_abs.mlir):

```shell
# iree-run-mlir <compiler flags> [input.mlir] <runtime flags>
$ ../iree-build/tools/iree-run-mlir \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=vmvx \
  $PWD/samples/models/simple_abs.mlir \
  --input=f32=-2
```

### iree-dump-module

The `iree-dump-module` program prints the contents of an IREE module FlatBuffer
file.

For example, to inspect the module translated above:

```shell
../iree-build/tools/iree-dump-module /tmp/simple_abs_vmvx.vmfb
```

### Useful generic flags

#### Read inputs from a file

All the IREE tools support reading input values from a file. This is quite
useful for debugging. Use `--help` for each tool to see what the flag to set.
The inputs are expected to be newline-separated. Each input should be either a
scalar or a buffer. Scalars should be in the format `type=value` and buffers
should be in the format `[shape]xtype=[value]`. For example:

``` text
1x5xf32=1,-2,-3,4,-5
1x5x3x1xf32=15,14,13,12,11,10,9,8,7,6,5,4,3,2,1
```

#### `--iree-flow-trace-dispatch-tensors`

This flag will enable tracing inputs and outputs for each dispatch function. It
is easier to narrow down test cases, since IREE breaks a ML workload into
multiple dispatch function. When the flag is on, IREE will insert trace points
before and after each dispatch function. The first trace op is for inputs, and
the second trace op is for outputs. There will be two events for one dispatch
function.
