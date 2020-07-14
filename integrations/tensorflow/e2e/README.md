# TensorFlow e2e tests

This is a collection of e2e tests that save a TensorFlow model, compile it with
IREE, run it on multiple backends and crosscheck the results.

## Pre-Requisites

You will need a TensorFlow 2.0+ nightly installed in your python environment:
the python binary in `$PYTHON_BIN` should be able to `import tensorflow` and
that TensorFlow should be version 2.0+. This can be checked with
`tensorflow.version`.

See [Install TensorFlow with pip](https://www.tensorflow.org/install/pip) for
instructions.

## Vulkan setup

If you do not have your environment setup to use IREE with Vulkan (see
[the doc](../../../docs/vulkan_and_spirv.md)), then you can run the manual test
targets with `--target_backends=tf,iree_vmla,iree_llvmjit` (that is, by omitting
`iree_vulkan` from the list of backends to run the tests on).

The test suites can be run excluding Vulkan by specifying
`--test_tag_filters="-driver=vulkan"` in the `bazel test` invocation.

## Compiling `tf.Module`s

Compatible TensorFlow modules can be compiled to specific IREE backends using
`IreeCompiledModule.compile(...)`. This also optionally saves
compilation artifacts to a specified directory. These artifacts include: MLIR
across various lowerings, a TensorFlow SavedModel, and the compiled VM
FlatBuffer. A basic example of creating and calling an `IreeCompiledModule` can
be found in
[`tf_utils_test.py`](https://github.com/google/iree/blob/main/integrations/tensorflow/bindings/python/pyiree/tf/support/tf_utils_test.py)

When using Keras models or tf.Modules with functions that IREE can't compile,
`exported_names` should be specified. For example:

```python
from pyiree.tf.support import tf_utils
vmla_module = tf_utils.IreeCompiledModule(
    constructor=KerasTFModuleClass,
    backend_info=tf_utils.BackendInfo.ALL['iree_vmla'],
    exported_names=['predict'])
vmla_module.predict(...)
```

## Running tests

For locally running tests and iterating on backend development, `bazel run` is
preferred.

```shell
# Run math_test on all backends.
bazel run :math_test_manual

# Run math_test on the VMLA backend only.
bazel run :math_test_manual -- --target_backends=iree_vmla

# Same as above, but add `tf` backend to cross-check numerical correctness.
bazel run :math_test_manual -- --target_backends=tf,iree_vmla

# Run math_test and output on failure.
bazel test :math_test_manual --test_output=errors

# Run an individual test interactively.
bazel run :math_test_manual -- --test_output=streamed
```

If you specify the same backend multiple times, for example
`--target_backends=iree_vmla,iree_vmla`. The same backends are grouped and in
this example `iree_vmla` will run once. If you specify `tf,iree_vmla` as
backends, then we will test both backends and compare them with each other. If
you specify `tf` backend only, then we will also test `tf` vs `tf` to capture
any model initialization/randomization issues (it is a special case for debug
purpose). For reproducibility of the unit tests we set random seed of `tf` and
`numpy` by calling `tf_utils.set_random_seed()` before model creation.

## Test Suites

Test targets are automatically generated for each test file and for each backend
to check numerical correctness against TensorFlow. Tests targets that pass are
placed into the `e2e_tests` test suite. Tests that fail on particular backends
are recorded in lists in the `BUILD` files. For example, if
`experimental_new_test.py` fails on the `iree_llvmjit` and `iree_vulkan`
backends then the following lines should be added to the `BUILD` file:

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
bazel test :e2e_tests

# Run all e2e tests that are expected to fail.
bazel test :e2e_tests_failing

# Run a specific failing e2e test target.
# Note that generated test targets are prefixed with their test suite name.
bazel test :e2e_tests_failing_broadcasting_test__tf__iree_vulkan
```

## Debugging tests

If the compiler fails to compile the program, then it will create a crash
reproducer (see [MLIR documentation](https://mlir.llvm.org/docs/WritingAPass/)),
which then allows reproducing the bug with an appropriate "opt" tool. Further
debugging iteration can happen in opt.

TODO(silvasean): debugging miscompiles

## Test harnesses

### Simple function tests

See `simple_arithmetic_test.py` for some basic examples.
