---
layout: default
permalink: developing-iree/tensorflow-integrations
title: TensorFlow Integrations
nav_order: 5
parent: Developing IREE
---

# TensorFlow e2e tests
{: .no_toc }

<!-- TODO(meadowlark): Update this doc once the API is stable. -->

> Note
> {: .label .label-blue }
> The TensorFlow integrations are currently being
  refactored. The `bazel` build is deprecated. Refer to
  https://google.github.io/iree/get-started/getting-started-python for a general
  overview of how to build and execute the e2e tests.

This is a collection of e2e tests that compile a TensorFlow model with IREE (and
potentially TFLite), run it on multiple backends, and crosscheck the results.

## Pre-Requisites

You will need a TensorFlow 2.0+ nightly installed in your python environment:
the python binary in `$PYTHON_BIN` should be able to `import tensorflow` and
that TensorFlow should be version 2.0+. This can be checked with
`tensorflow.version`.

See [Install TensorFlow with pip](https://www.tensorflow.org/install/pip) for
instructions.

## Vulkan Setup

If you do not have your environment setup to use IREE with Vulkan (see
[this doc](https://google.github.io/iree/get-started/generic-vulkan-env-setup)),
then you can run the manual test targets with
`--target_backends=tf,iree_vmla,iree_llvmaot` (that is, by omitting
`iree_vulkan` from the list of backends to run the tests on).

The test suites can be run excluding Vulkan by specifying
`--test_tag_filters="-driver=vulkan"` in the `bazel test` invocation, or by
adding `test --test_tag_filters="-driver=vulkan"` to your `user.bazelrc`.

## Compiling `tf.Module`s

Compatible TensorFlow modules can be compiled to specific IREE backends using
`IreeCompiledModule`. This also optionally saves compilation artifacts to a
specified directory. These artifacts include MLIR across various lowerings and
the compiled VM FlatBuffer. A basic example of creating and calling an
`IreeCompiledModule` can be found in
[`module_utils_test.py`](https://github.com/google/iree/blob/main/integrations/tensorflow/bindings/python/iree/tf/support/module_utils_test.py)

When using Keras models or tf.Modules with functions that IREE can't compile,
`exported_names` should be specified. For example:

```python
from iree.tf.support import module_utils
vmla_module = module_utils.IreeCompiledModule(
    module_class=KerasTFModuleClass,
    backend_info=module_utils.BackendInfo('iree_vmla'),
    exported_names=['predict'])
vmla_module.predict(...)
```

## Running Tests

For locally running tests and iterating on backend development, `bazel run` is
preferred.

```shell
# Run conv_test on all backends.
bazel run //integrations/tensorflow/e2e:conv_test_manual

# Run conv_test comparing TensorFlow to itself (e.g. to debug randomization).
bazel run //integrations/tensorflow/e2e:conv_test_manual -- --target_backends=tf

# Run conv_test comparing the VMLA backend and TensorFlow.
bazel run //integrations/tensorflow/e2e:conv_test_manual -- --target_backends=iree_vmla

# Run conv_test comparing the VMLA backend to itself multiple times.
bazel run //integrations/tensorflow/e2e:conv_test_manual -- \
  --reference_backend=iree_vmla --target_backends=iree_vmla,iree_vmla
```

For reproducibility of the unit tests `CompiledModule()` sets the random seeds
of `tf`, `numpy` and `python` by calling `tf_utils.set_random_seed()` before
model creation.

## Writing Tests

There are two ways to write tests – via `tf_test_utils.tf_function_unit_test` and
via test methods on a child of `tf_test_utils.TracedModuleTestCase`.

### Via `tf_test_utils.tf_function_unit_test`

This is preferred in the cases where

1. Only a single call to the module needs to be tested at once
2. The inputs are simple to automatically generate or specify inline.
3. The functions that you want to test are generated automatically from a
   configuration (e.g. in `.../e2e/keras/layers/layers_test.py`)

Tests are specified by writing modules that inherit from
`tf_test_utils.TestModule` (which is a thin wrapper around `tf.Module`) with
methods decorated with `@tf_test_utils.tf_function_unit_test` (with is a thin
wrapper around `tf.function`).

#### Basic example

We use part of `.../e2e/conv_test.py` as an example. The first component is
the `TestModule` itself:

```python
class Conv2dModule(tf_test_utils.TestModule):

  # This decorator tells the testing infra to generate a unittest for this
  # function. The 'input_signature' is required. If no other arguments are
  # specified then uniform random data is generated from the input signature
  # to numerically test the function.
  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([1, 4, 5, 1], tf.float32),
      tf.TensorSpec([1, 1, 1, 1], tf.float32),
  ])
  def conv2d_1451x1111_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")

  @tf_test_utils.tf_function_unit_test(input_signature=[
      tf.TensorSpec([2, 4, 5, 1], tf.float32),
      tf.TensorSpec([1, 1, 1, 1], tf.float32),
  ])
  def conv2d_2451x1111_valid(self, img, kernel):
    return tf.nn.conv2d(img, kernel, [1, 1, 1, 1], "VALID", name="result")
```

Second, you need to write a test case that inherits from
`tf_test_utils.TracedModuleTestCase`. This is essentially boiler plate that
tells `tf.test.main()` what `tf.Module` to test and allows us to generate
the unittests we specified above.

```python
class ConvTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._modules = tf_test_utils.compile_tf_module(Conv2dModule)
```

Finally, in the `main` function, you need to call
`.generate_unit_tests(module_class)` on your `TestCase` to actually generate
the unittests that we specified:

```python
def main(argv):
  del argv  # Unused
  if hasattr(tf, 'enable_v2_behavior'):
    tf.enable_v2_behavior()
  # Generates unittests for all @tf_test_utils.tf_function_unit_test decorated
  # functions on the module class.
  # Note: if you are automatically generating functions to test they need to be
  # specified via a `classmethod` prior to this call _as well_ as via `__init__`
  # to properly handle stateful `tf.function`s.
  ConvTest.generate_unit_tests(Conv2dModule)
  tf.test.main()


if __name__ == '__main__':
  app.run(main)
```

This generates two unittests: `test_conv2d_1451x1111_valid` and
`test_conv2d_2451x1111_valid`.

#### Configuring `@tf_test_utils.tf_function_unit_test`

By default `@tf_test_utils.tf_function_unit_test` uses uniform random input data
to numerically test the function, but you can specify an `input_generator` or
`input_args` to test data-specific behaviors:

- `input_generator` can be `tf_utils.uniform`, `tf_utils.ndarange`, or any
function which takes an `shape` and `dtype` as positional args and returns an
`np.ndarray`.
- `input_args` is a list of `np.ndarray`s to use as positional arguments.

The comparison `atol` and `rtol` can also be specified in the decorator.

### Via test methods

This is preferred in the cases where

1. The `tf.function` that you want to test is already defined on the module
   (e.g. on a downloaded model like in `mobile_bert_test.py`)
2. The inputs are difficult to specify inline and require multiple function
   calls / reshaping to create
3. You want to test multiple consecutive calls to a `tf.function` (e.g. to test
   mutated state in `ring_buffer_test.py`)

Our tests use a class `TracedModule` to capture and store all of the inputs and
outputs of a `CompiledModule` in a `Trace`. Each unittest on a `TestCase` uses
the `compare_backends` method. This method runs the function it is passed with a
`TracedModule` once for each reference and target backend. The inputs and
outputs to these modules are then checked for correctness, using the reference
backend as a source of truth.

We use `simple_arithmetic_test.py` as an example:

```python
# Create a tf.Module with one or more `@tf.function` decorated methods to test.
class SimpleArithmeticModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    return a * b

# Inherit from `TracedModuleTestCase`.
class SimpleArithmeticTest(tf_test_utils.TracedModuleTestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Compile a `tf.Module` named `SimpleArithmeticModule` into
    # `CompiledModule`s for each reference and target backend.
    self._modules = tf_test_utils.compile_tf_module(SimpleArithmeticModule)

  # Unit test.
  def test_simple_mul(self):

    # Trace function.
    def simple_mul(module):
      # A random seed is automatically set before each call to `simple_mul`.
      a = tf_utils.uniform([4])
      b = np.array([400., 5., 6., 7.], dtype=np.float32)

      # The inputs `a` and `b` are recorded along with the output `c`
      c = module.simple_mul(a, b)

      # The inputs `a` and `b` are recorded along with the (unnamed) output
      # module.simple_mul returns.
      module.simple_mul(a, c)

    # Calls `simple_mul` once for each backend, recording the inputs and outputs
    # to `module` and then comparing them.
    self.compare_backends(simple_mul, self._modules)
```

## Test Suites

Test targets are automatically generated for each test file and for each backend
to check numerical correctness against TensorFlow. Tests targets that pass are
placed into the `e2e_tests` test suite. Tests that fail on particular backends
are recorded in lists in the `BUILD` files. For example, if
`experimental_new_test.py` fails on the `iree_llvmaot` and `iree_vulkan`
backend then the following lines should be added to the `BUILD` file:

```build
LLVM_FAILING = [
    ...
    "experimental_new_test.py",
    ...
]

VULKAN_FAILING = [
    ...
    "experimental_new_test.py",
    ...
]
```

Test targets for these backends are placed into the `e2e_tests_failing` test
suite. Test targets in these test suites can be run as follows:

```shell
# Run all e2e tests that are expected to pass.
bazel test //integrations/tensorflow/e2e:e2e_tests

# Run all e2e tests that are expected to fail.
bazel test //integrations/tensorflow/e2e:e2e_tests_failing

# Run a specific failing e2e test target.
# Note that generated test targets are prefixed with their test suite name.
# Also, if broadcasting_test starts working on iree_vulkan after the time
# of writing then this command will fail.
bazel test //integrations/tensorflow/e2e:e2e_tests_failing_broadcasting_test__tf__iree_vulkan
```

## Generated Artifacts

By default, running an E2E test generates a number of compilation, debugging and
benchmarking artifacts. These artifacts will be saved

- in `/tmp/iree/modules/` when using `bazel run` or `bazel_test` with
  `--test_arg=--artifacts_dir=/tmp/iree/modules/`.
- in `bazel-testlogs/integrations/tensorflow/e2e/test_suite_target_name` when
  using `bazel test` without specifying `--artifacts_dir`.

The generated directory structure for each module is as follows:

```shell
/tmp/iree/modules/ModuleName
  ├── reproducer__backend.mlir
  │   # If there is a compilation error, a MLIR file that reproduces the error
  │   # for a specific backend is included.
  ├── tf_input.mlir
  │   # MLIR for ModuleName in TF's input dialect.
  ├── iree_input.mlir
  │   # tf_input.mlir translated to IREE MLIR.
  ├── iree_vmla
  │   # Or any other IREE backend.
  │   ├── compiled.vmfb
  │   │   # A flatbuffer containing IREE's compiled code.
  │   └── traces
  │       # Directory with a trace for each unittest in vision_model_test.py.
  │       ├── trace_function_1
  │       │   # Directory storing logs and serialization for a specific trace.
  │       │   │── flagfile
  │       │   │   # An Abseil flagfile containing arguments
  │       │   │   # iree-benchmark-module needs to benchmark this trace.
  │       │   └── log.txt
  │       │       # A more detailed version of the test logs.
  │       │── trace_function_2
  │       └── ...
  ├── tflite  # If TFLite supports compiling ModuleName.
  │   ├── method_1.tflite  # Methods on ModuleName compiled to bytes with TFLite
  │   │   # A method on ModuleName compiled to bytes with TFLite, which can
  │   │   # be ingested by TFLite's benchmark_model binary.
  │   ├── method_2.tflite
  │   └── traces
  │       └── ...
  └── tf_ref  # Directory storing the tensorflow reference traces.
      └── traces
          └── ...
```

Traces for a particular test can be loaded via the `Trace.load(trace_dir)`
method. For example:

```python
ref_trace = Trace.load("/tmp/iree/modules/ModuleName/tf_ref/traces/predict/")
tar_trace = Trace.load("/tmp/iree/modules/ModuleName/iree_vmla/traces/predict/")
abs_diff = np.abs(ref_trace.calls[0].outputs[0] - tar_trace.calls[0].outputs[0])
print(np.mean(abs_diff))
```

Traces are named after the trace functions defined in their unittests. So in the
`SimpleArithmeticModule` example above, the `trace_dir` would be
`/tmp/iree/modules/SimpleArithmeticModule/iree_vmla/traces/simple_mul/`.

## Benchmarking E2E Modules

We use our end-to-end TensorFlow integrations tests to generate tested
compilation and benchmarking artifacts. This allows us to validate that our
benchmarks are behaving as we expect them to, and to run them using valid inputs
for each model. An overview of how to run benchmarks on IREE and TFLite can be
found in [this doc](https://google.github.io/iree/developing-iree/e2e-benchmarking).

## Debugging Tests

If the compiler fails to compile the program, then it will create a crash
reproducer (see
[MLIR documentation](https://mlir.llvm.org/docs/PassManagement/#crash-and-failure-reproduction)),
which then allows reproducing the bug with an appropriate "opt" tool. Further
debugging iteration can happen in opt.

TODO(silvasean): debugging miscompiles

## Testing SignatureDef SavedModels

TensorFlow 1.x SavedModels can be tested using
`tf_test_utils.compile_tf_signature_def_saved_model` instead of
`tf_test_utils.compile_tf_module`. See `mobile_bert_squad_test.py` for a
concrete example. The compilation artifacts will be saved under whatever
you specify for `module_name`.