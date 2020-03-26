# TensorFlow e2e tests

This is a collection of e2e tests that, in various fashion saves a TensorFlow
model, compiles it with IREE and runs/evaluates it on all backends.

## Pre-Requisites

You will need a TensorFlow 2.0+ nightly installed in your python environment:
the python binary in `$PYTHON_BIN` should be able to `import tensorflow` and
that TensorFlow should be version 2.0+. This can be checked with
`tensorflow.version`.

See [Install TensorFlow with pip](https://www.tensorflow.org/install/pip) for
instructions.

## Vulkan setup

If you do not have your environment setup to use IREE with Vulkan (see
[the doc](../../../docs/vulkan_and_spirv.md)), then you can run the tests with
`IREE_DEFAULT_BACKENDS=tf,iree_vmla` (that is, by omitting `iree_vulkan` from
the list of backends to use).

## Running tests

```shell
# Run all tests with defaults and output on failure.
bazel test ... --test_output=errors

# Run an individual test interactively.
bazel test simple_arithmetic_test --test_output=streamed

# Run tests with an altered list of backends.
bazel test ... --test_output=errors -- \
    --override_backends=tf,iree_vmla,iree_vulkan

# (alternative) Run tests with an altered list of backends.
bazel test ... --test_env=IREE_OVERRIDE_BACKENDS=tf,iree_vmla,iree_vulkan \
    --test_output=errors
```

## Debugging tests

If the compiler fails to compile the program, then it will create a crash
reproducer (see documentation [here](https://mlir.llvm.org/docs/WritingAPass/)),
which then allows reproducing the bug with an appropriate "opt" tool. Further
debugging iteration can happen in opt.

TODO(silvasean): debugging miscompiles

## Test harnesses

### Simple function tests

See `simple_arithmetic_test.py` for some examples of writing a test case that
runs on multiple backends.

### Limiting a test to only certain backends

The `@tf_test_utils.compile_modules` decorator on tests takes a `backends=`
keyword argument. This argument should be a Python list of backends, which
accepts the same keys as the `--override_backends` flags.

Example:

```
@tf_test_utils.compile_modules(backends=["tf"], mlp=(Mlp, ["predict"]))
class DynamicMlpTest(tf_test_utils.SavedModelTestCase):
  ... the test case ...
```

Limiting this statically in the code can be useful for tests that are known to
fail on certain backends but are still useful to have checked in.

The priority order for which backends are ultimately used is:

1.  The backends specified in `--override_backends`.

2.  The backends specified in `IREE_OVERRIDE_BACKENDS`.

3.  The backends specified in the `tf_test_utils.compile_modules` decorator.

4.  The backends specified in `IREE_DEFAULT_BACKENDS`.

5.  All known backends.
