# Developer Overview

This guide provides an overview of IREE's project structure and main tools for
developers.

## Project Code Layout

[iree/](https://github.com/google/iree/blob/main/iree/)

*   Core IREE project

[integrations/](https://github.com/google/iree/blob/main/integrations/)

*   Integrations between IREE and other frameworks, such as TensorFlow

[bindings/](https://github.com/google/iree/blob/main/bindings/)

*   Language and platform bindings, such as Python

[colab/](https://github.com/google/iree/blob/main/colab/)

*   Colab notebooks for interactively using IREE's Python bindings

## IREE Code Layout

[iree/base/](https://github.com/google/iree/blob/main/iree/base/)

*   Common types and utilities used throughout IREE

[iree/compiler/](https://github.com/google/iree/blob/main/iree/compiler/)

*   IREE's MLIR dialects, LLVM compiler passes, module translation code, etc.
    Code here should not depend on anything in the runtime

[iree/hal/](https://github.com/google/iree/blob/main/iree/hal/)

*   **H**ardware **A**bstraction **L**ayer for IREE's runtime, with
    implementations for hardware and software backends

[iree/schemas/](https://github.com/google/iree/blob/main/iree/schemas/)

*   Shared data storage format definitions, primarily using
    [FlatBuffers](https://google.github.io/flatbuffers/)

[iree/tools/](https://github.com/google/iree/blob/main/iree/tools/)

*   Assorted tools used to optimize, translate, and evaluate IREE

[iree/vm/](https://github.com/google/iree/blob/main/iree/vm/)

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
[test file](https://github.com/google/iree/blob/main/iree/compiler/Dialect/IREE/Transforms/test/drop_compiler_hints.mlir):

```shell
$ bazel run iree/tools:iree-opt -- \
  -split-input-file \
  -print-ir-before-all \
  -iree-drop-compiler-hints \
  $PWD/iree/compiler/Dialect/IREE/Transforms/test/drop_compiler_hints.mlir
```

For a more complex example, here's how to run IREE's complete transformation
pipeline targeting the VMLA backend on the
[fullyconnected.mlir](https://github.com/google/iree/blob/main/iree/test/e2e/models/fullyconnected.mlir)
model file:

```shell
$ bazel run iree/tools:iree-opt -- \
  -iree-transformation-pipeline \
  -iree-hal-target-backends=vmla \
  $PWD/iree/test/e2e/models/fullyconnected.mlir
```

Custom passes may also be layered on top of `iree-opt`, see
[iree/samples/custom_modules/dialect](https://github.com/google/iree/blob/main/iree/samples/custom_modules/dialect)
for a sample.

### iree-translate

`iree-translate` converts MLIR input into external formats like IREE modules. It
is similar to
[mlir-translate](https://github.com/llvm/llvm-project/tree/master/mlir/tools/mlir-translate),
see "translation" in
[MLIR's Glossary](https://mlir.llvm.org/getting_started/Glossary/#translation)
for more information.

For example, to translate `simple.mlir` to an IREE module:

```shell
$ bazel run iree/tools:iree-translate -- \
  -iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=vmla \
  $PWD/iree/tools/test/simple.mlir \
  -o /tmp/simple.module
```

Custom translations may also be layered on top of `iree-translate`, see
[iree/samples/custom_modules/dialect](https://github.com/google/iree/blob/main/iree/samples/custom_modules/dialect)
for a sample.

### iree-run-module

The `iree-run-module` program takes an already translated IREE module as input
and executes an exported main function using the provided inputs.

This program can be used in sequence with `iree-translate` to translate a
`.mlir` file to an IREE module and then execute it. Here is an example command
that executes the simple `simple.module` compiled from `simple.mlir` above on
IREE's VMLA driver:

```shell
$ bazel run iree/tools:iree-run-module -- \
  --input_file=/tmp/simple.module \
  --driver=vmla \
  --entry_function=abs \
  --inputs="i32=-2"
```

### iree-check-module

The `iree-check-module` program takes an already translated IREE module as input
and executes it as a series of
[googletest](https://github.com/google/googletest) tests. This is the test
runner for the IREE
[check framework](https://github.com/google/iree/tree/main/docs/developing_iree/testing_guide.md#end-to-end-tests).

```shell
$ bazel run iree/tools:iree-translate -- \
  -iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=vmla \
  $PWD/iree/test/e2e/xla_ops/abs.mlir \
  -o /tmp/abs.module
```

```shell
$ bazel run iree/modules/check:iree-check-module -- \
  /tmp/abs.module \
  --driver=vmla
```

### iree-run-mlir

The `iree-run-mlir` program takes a `.mlir` file as input, translates it to an
IREE bytecode module, and executes the module.

It is designed for testing and debugging, not production uses, and therefore
does some additional work that usually must be explicit, like marking every
function as exported by default and running all of them.

For example, to execute the contents of
[iree/tools/test/simple.mlir](https://github.com/google/iree/blob/main/iree/tools/test/simple.mlir):

```shell
$ bazel run iree/tools:iree-run-mlir -- \
  $PWD/iree/tools/test/simple.mlir \
  --input-value="i32=-2" \
  --iree-hal-target-backends=vmla
```

### iree-dump-module

The `iree-dump-module` program prints the contents of an IREE module FlatBuffer
file.

For example, to inspect the module translated above:

```shell
$ bazel run iree/tools:iree-dump-module -- /tmp/simple.module
```

### Useful generic flags

There are a few useful generic flags when working with IREE tools:

#### `--iree_minloglevel` and `--iree_v`

These flags can control IREE tool output verbosity. `--iree_minloglevel` and
`--iree_v` set the minimal and maximal verbosity levels respectively. They both
accept a number where 0, 1, 2, 3 stands for info, warning, error, and fatal
error respectively.

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

For IREE's Vulkan runtime driver, there are a few useful
[flags](https://github.com/google/iree/blob/main/iree/hal/vulkan/vulkan_driver.cc):

#### `--vulkan_renderdoc`

This flag tells IREE to load RenderDoc, connect to it's in-application API, and
trigger capturing on its own. For example, this command runs `iree-run-mlir` on
a simple MLIR file with some sample input values and saves a RenderDoc capture
to the default location on your system (e.g. `/tmp/RenderDoc/`):

```shell
$ bazel build iree/tools:iree-run-mlir
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/renderdoc/lib/path \
  bazel-bin/iree/tools/iree-run-mlir \
    $PWD/iree/samples/vulkan/simple_mul.mlir \
    -iree-hal-target-backends=vulkan-spirv \
    -input-value="4xf32=1,2,3,4" \
    -input-value="4xf32=2,4,6,8" \
    -run-arg="--vulkan_renderdoc"
```

This flag also works for other IREE execution tools like `iree-run-module`,
`iree-check-module`.

You can also launch IREE's headless programs through RenderDoc itself, just be
sure to set the command line arguments appropriately. Saving capture settings in
RenderDoc can help if you find yourself doing this frequently.

Note: RenderDoc version 1.7 or higher is needed to record captures from IREE's
headless compute programs.
