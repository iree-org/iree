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

By default, tests run on TensorFlow and the IREE CPU interpreter, as it never
needs additional environment setup. If you have your environment setup to use
IREE with Vulkan (see [the doc](../../../docs/vulkan_and_spirv.md)), then you
can enable the backends by setting the environment variable
`IREE_TEST_BACKENDS=tf,iree_interpreter,iree_vulkan`.

You can also pass this as a command line argument when running individual tests:
`--target_backends=tf,iree_interpreter,iree_vulkan`.

## Running tests

```shell
# Run all tests with defaults and output on failure.
bazel test ... --test_output=errors

# Run an individual test interactively.
bazel test simple_arithmetic_test --test_output=streamed

# Run tests with an altered list of backends.
bazel test ... --test_output=errors -- \
    --target_backends=tf,iree_interpreter,iree_vulkan

# (alternative) Run tests with an altered list of backends.
bazel test ... --test_env=IREE_TEST_BACKENDS=tf,iree_interpreter,iree_vulkan \
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
