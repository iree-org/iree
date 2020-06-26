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
[the doc](../../../docs/vulkan_and_spirv.md)), then you can run the tests with
`IREE_AVAILABLE_BACKENDS=tf,iree_vmla,iree_llvmjit` (that is, by omitting
`iree_vulkan` from the list of available backends).

## Running tests

```shell
# For locally running tests and iterating on backend development,
# `bazel run` is preferred.
bazel run :math_test_manual -- --override_backends=iree_vmla

# Same as above, but add `tf` backend to cross-check numerical correctness.
bazel run :math_test_manual -- --override_backends=tf,iree_vmla

# Run all tests with defaults and output on failure.
bazel test ... --test_output=errors

# Run an individual test interactively.
bazel run :math_test_manual -- --test_output=streamed
```

If you specify the same backend multiple times, for example
`--override_backends=iree_vmla,iree_vmla`. The same backends are grouped and in
this example `iree_vmla` will run once. If you specify `tf,iree_vmla` as
backends, then we will test both backends and compare them with each other. If
you specify `tf` backend only, then we will also test `tf` vs `tf` to capture
any model initialization/randomization issues (it is a special case for debug
purpose). For reproducibility of the unit tests we set random seed of `tf` and
`numpy` by calling `tf_test_utils.set_random_seed()` before model creation.

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

### Limiting a test to only certain backends

The BUILD file specifies which targets work on which backends and controls which
backends tests are run on by using the `--override_backends` flag.

The `@tf_test_utils.compile_modules` decorator on tests also takes a `backends=`
keyword argument. Many tests still specify this, but it is ignored in the CI,
which runs with `bazel test`. When running with `bazel run` this indicates the
set of backends to use in the absence of the `--override_backends` flags (and
accepts the same arguments).

Example:

```
@tf_test_utils.compile_modules(backends=["tf"], mlp=(Mlp, ["predict"]))
class DynamicMlpTest(tf_test_utils.SavedModelTestCase):
  ... the test case ...
```

Limiting backends is useful for tests that are known to fail on certain backends
but are still useful to have checked in.

The priority order for which backends are ultimately used is:

1.  The backends specified in `--override_backends`.

2.  The backends specified in the `IREE_OVERRIDE_BACKENDS` environment variable.

3.  The backends specified in the `tf_test_utils.compile_modules` decorator.

4.  All known backends.

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
