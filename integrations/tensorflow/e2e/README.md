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
`IREE_AVAILABLE_BACKENDS=tf,iree_vmla` (that is, by omitting `iree_vulkan` from
the list of available backends).

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

If you specify the same backend multiple times, for example
--override_backends=iree_vmla,iree_vmla. The same backends are grouped and in
this example iree_vmla will run once. If you specify tf,iree_vmla as backends,
then we will test both backends and compare them with each other. If you specify
tf backend only, then we will also test tf vs tf to capture any model
initialization/randomization issues (it is a special case for debug purpose).
For reproducibility of the unit tests we set random seed of tf and numpy by
calling tf_test_utils.set_random_seed() before model creation.

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

1.  The backends specified in the `IREE_OVERRIDE_BACKENDS` environment variable.

1.  The backends specified in the `tf_test_utils.compile_modules` decorator.

1.  All known backends.

Additionally, the environment variable `IREE_AVAILABLE_BACKENDS` specifies which
backends should be considered available in a particular environment. Once the
list of backends above is formed, any backends not listed in
`IREE_AVAILABLE_BACKENDS` are removed. This is the final list of backends which
are run for the test.

The default behavior if `IREE_AVAILABLE_BACKENDS` is not provided is that all
known backends are considered available.

TODO(silvasean): `IREE_AVAILABLE_BACKENDS` is mainly to allow masking off the
Vulkan backend in environments where it is not a available. Currently, the
behavior when all backends get masked off is to emit a warning, which can result
in spuriously "passing" tests. This is only an issue for tests that currently
only run on Vulkan (which should decrease over time as e.g. VMLA gets more
coverage).
