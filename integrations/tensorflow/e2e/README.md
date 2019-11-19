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
`IREE_TEST_BACKENDS=tf,iree.interpreter,iree.vulkan`.

## Running tests

```shell
# Run all tests with defaults and output on failure.
bazel test ... --test_output=errors

# Run an individual test interactively.
bazel test simple_arithmetic_test --test_output=streamed

# Run tests with an altered list of backends.
bazel test ... --test_env=IREE_TEST_BACKENDS=tf,iree_interpreter,iree_vulkan \
    --test_output=errors
```

## Test harnesses

### Simple function tests

See `simple_arithmetic_test.py` for some examples of single function tests.
These are done by extending a tf_test_utils.SavedModelTestCase and then
annotating individual test methods with
`@tf_test_utils.per_backend_test("function_name")` to get a function that will
run and compare on all backends.
