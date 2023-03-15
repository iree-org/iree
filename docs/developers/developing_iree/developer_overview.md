# Developer Overview

This guide provides an overview of IREE's project structure and main tools for
developers.

** Note: project layout is evolving at the moment, see
   https://github.com/openxla/iree/issues/8955 **

## Project Code Layout

[iree/](https://github.com/openxla/iree/blob/main/iree/)

*   Core IREE project

[integrations/](https://github.com/openxla/iree/blob/main/integrations/)

*   Integrations between IREE and other frameworks, such as TensorFlow

[runtime/](https://github.com/openxla/iree/tree/main/runtime/)

*   IREE runtime code, with no dependencies on the compiler

[bindings/](https://github.com/openxla/iree/blob/main/bindings/)

*   Language and platform bindings, such as Python
*   Also see [runtime/bindings/](https://github.com/openxla/iree/tree/main/runtime/bindings)

[samples/](https://github.com/openxla/iree/blob/main/samples/)

*   Samples built using IREE's runtime and compiler
*   Also see the separate https://github.com/iree-org/iree-samples repository

## IREE Compiler Code Layout

[iree/compiler/](https://github.com/openxla/iree/blob/main/iree/compiler/)

*   IREE's MLIR dialects, LLVM compiler passes, module translation code, etc.

## IREE Runtime Code Layout

[iree/base/](https://github.com/openxla/iree/blob/main/runtime/src/iree/base/)

*   Common types and utilities used throughout the runtime

[iree/hal/](https://github.com/openxla/iree/blob/main/runtime/src/iree/hal/)

*   **H**ardware **A**bstraction **L**ayer for IREE's runtime, with
    implementations for hardware and software backends

[iree/schemas/](https://github.com/openxla/iree/blob/main/runtime/src/iree/schemas/)

*   Shared data storage format definitions, primarily using
    [FlatBuffers](https://google.github.io/flatbuffers/)

[tools/](https://github.com/openxla/iree/blob/main/tools/)

*   Assorted tools used to optimize, translate, and evaluate IREE

[iree/vm/](https://github.com/openxla/iree/blob/main/runtime/src/iree/vm/)

*   Bytecode **V**irtual **M**achine used to work with IREE modules and invoke
    IREE functions

## Developer Tools

IREE's compiler components accept programs and code fragments in several
formats, including high level TensorFlow Python code, serialized TensorFlow
[SavedModel](https://www.tensorflow.org/guide/saved_model) programs, and lower
level textual MLIR files using combinations of supported dialects like `mhlo`
and IREE's internal dialects. While input programs are ultimately compiled down
to modules suitable for running on some combination of IREE's target deployment
platforms, IREE's developer tools can run individual compiler passes,
translations, and other transformations step by step.

### iree-opt

`iree-opt` is a tool for testing IREE's compiler passes. It is similar to
[mlir-opt](https://github.com/llvm/llvm-project/tree/master/mlir/tools/mlir-opt)
and runs sets of IREE's compiler passes on `.mlir` input files. See "conversion"
in [MLIR's Glossary](https://mlir.llvm.org/getting_started/Glossary/#conversion)
for more information. Transformations performed by `iree-opt` can range from
individual passes performing isolated manipulations to broad pipelines that
encompass a sequence of steps.

Test `.mlir` files that are checked in typically include a `RUN` block at the
top of the file that specifies which passes should be performed and if
`FileCheck` should be used to test the generated output.

Here's an example of a small compiler pass running on a
[test file](https://github.com/openxla/iree/blob/main/iree/compiler/Dialect/Util/Transforms/test/drop_compiler_hints.mlir):

```shell
$ ../iree-build/tools/iree-opt \
  --split-input-file \
  --mlir-print-ir-before-all \
  --iree-drop-compiler-hints \
  $PWD/iree/compiler/Dialect/Util/Transforms/test/drop_compiler_hints.mlir
```

For a more complex example, here's how to run IREE's complete transformation
pipeline targeting the VMVX backend on the
[fullyconnected.mlir](https://github.com/openxla/iree/blob/main/tests/e2e/models/fullyconnected.mlir)
model file:

```shell
$ ../iree-build/tools/iree-opt \
  --iree-transformation-pipeline \
  --iree-hal-target-backends=vmvx \
  $PWD/tests/e2e/models/fullyconnected.mlir
```

Custom passes may also be layered on top of `iree-opt`, see
[samples/custom_modules/dialect](https://github.com/openxla/iree/blob/main/samples/custom_modules/dialect)
for a sample.

### iree-compile

`iree-compile` is IREE's main compiler driver for generating binaries from
supported input MLIR assembly.

For example, to translate `simple.mlir` to an IREE module:

```shell
$ ../iree-build/tools/iree-compile \
  --iree-hal-target-backends=vmvx \
  $PWD/samples/models/simple_abs.mlir \
  -o /tmp/simple_abs_vmvx.vmfb
```

### iree-run-module

The `iree-run-module` program takes an already translated IREE module as input
and executes an exported main function using the provided inputs.

This program can be used in sequence with `iree-compile` to translate a
`.mlir` file to an IREE module and then execute it. Here is an example command
that executes the simple `simple_abs_vmvx.vmfb` compiled from `simple_abs.mlir`
above on IREE's VMVX driver:

```shell
$ ../iree-build/tools/iree-run-module \
  --module=/tmp/simple_abs_vmvx.vmfb \
  --device=local-task \
  --function=abs \
  --input=f32=-2
```

### iree-check-module

The `iree-check-module` program takes an already translated IREE module as input
and executes it as a series of
[googletest](https://github.com/google/googletest) tests. This is the test
runner for the IREE
[check framework](https://github.com/openxla/iree/tree/main/docs/developing_iree/testing_guide.md#end-to-end-tests).

```shell
$ ../iree-build/tools/iree-compile \
  --iree-input-type=mhlo \
  --iree-hal-target-backends=vmvx \
  $PWD/tests/e2e/xla_ops/abs.mlir \
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
[samples/models/simple_abs.mlir](https://github.com/openxla/iree/blob/main/samples/models/simple_abs.mlir):

```shell
# iree-run-mlir <compiler flags> [input.mlir] <runtime flags>
$ ../iree-build/tools/iree-run-mlir \
  --iree-hal-target-backends=vmvx \
  $PWD/samples/models/simple_abs.mlir \
  --input=f32=-2
```

### iree-dump-module

The `iree-dump-module` program prints the contents of an IREE module FlatBuffer
file.

For example, to inspect the module translated above:

```shell
$ ../iree-build/tools/iree-dump-module /tmp/simple_abs_vmvx.vmfb
```

### Useful generic flags

There are a few useful generic flags when working with IREE tools:

#### Read inputs from a file

All the IREE tools support reading input values from a file. This is quite
useful for debugging. Use `-help` for each tool to see what the flag to set. The
inputs are expected to be newline-separated. Each input should be either a
scalar or a buffer. Scalars should be in the format `type=value` and buffers
should be in the format `[shape]xtype=[value]`. For example:

```
1x5xf32=1,-2,-3,4,-5
1x5x3x1xf32=15,14,13,12,11,10,9,8,7,6,5,4,3,2,1
```

#### `iree-flow-trace-dispatch-tensors`

This flag will enable tracing inputs and outputs for each dispatch function. It
is easier to narrow down test cases, since IREE breaks a ML workload into
multiple dispatch function. When the flag is on, IREE will insert trace points
before and after each dispatch function. The first trace op is for inputs, and
the second trace op is for outputs. There will be two events for one dispatch
function.

### Useful Vulkan driver flags

For IREE's Vulkan runtime driver, there are a few useful flags defined in
[driver_module.cc](https://github.com/openxla/iree/blob/main/iree/hal/drivers/vulkan/registration/driver_module.cc):
