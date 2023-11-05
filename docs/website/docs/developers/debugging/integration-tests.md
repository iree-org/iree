# Integration test debugging

This document includes tips for triaging integration test correctness issues.
Feel free to reach out to @hanhanW or ask questions on Discord for more help.

## General tips

### Narrow down reproducers

* Models themselves can be large, and IREE breaks models into dispatches/kernels
and then launches those individually. Program outputs could diverge starting
from any individual launch. To get a smaller reproducer, you can use
[--iree-flow-trace-dispatch-tensors](../general/developer-overview.md#-iree-flow-trace-dispatch-tensors).
* You can compare the logs between builds/backends to get an idea about which
dispatch results in wrong outputs. The dumped inputs can be reused in a
flagfile.

Once a suspicious dispatch is identified, we can create a test case based on
the dispatch function. The dispatch function can be derived after the
`OutlineDispatchRegions` pass. The function signatures have to be modified
manually. You'll have to put `flow.dispatch.tensor.load` variables to function
arguments, and replace `flow.dispatch.tensor.store` with `return` op.

Note: This only works when dispatch formation logics are identical between runs.

## iree-samples repository tests

Follow [README](https://github.com/iree-org/iree-samples#readme) to run the model.
The MLIR files will be generated. You'll find the saved file from log. E.g.,

``` shell
[ RUN      ] MobilenetV2Int8Test.test_compile_tflite
I0401 17:27:04.084272 140182373025024 test_util.py:119] Setting up for IREE
I0401 17:27:04.085064 140182373025024 binaries.py:218] Invoke IREE Pipeline:
  /tmp/iree-samples/iree-samples.venv/lib/python3.9/site-packages/iree/tools/tflite/iree-import-tflite
    /tmp/iree-samples/tflitehub/tmp/mobilenet_v2_int8_test.py/model.tflite
    --mlir-print-debuginfo
    --save-temp-tfl-input=/tmp/iree-samples/tflitehub/tmp/mobilenet_v2_int8_test.py/tflite.mlir
    --save-temp-iree-input=/tmp/iree-samples/tflitehub/tmp/mobilenet_v2_int8_test.py/tosa.mlir
```

Unfortunately, the artifacts are not dumped in the runs. There is an
[issue](https://github.com/openxla/iree/issues/8756) for tracking this. A
workaround can be found in the issue.

## TensorFlow integration tests

These are steps to reproduce/address failures in TF/TFLite integration tests.
These instructions are most stable on Linux, though they may work with a few
tweaks on Windows and macOS.

All steps here assume starting from the IREE root directory.

1. First create a Python virtual environment to install packages into:

    ```bash
    python -m venv iree-tf.venv
    source iree-tf.venv/bin/activate

    # Install test requirements
    python -m pip install -r ./integrations/tensorflow/test/requirements.txt
    ```

2. Install IREE's tools and Python bindings or build them from source

    Install distributed packages

    ```bash
    # Install packages from nightly releases
    # This should work for most cases, as the importers change infrequently
    python -m pip install \
      iree-compiler iree-runtime iree-tools-tf iree-tools-tflite \
      --find-links https://iree.dev/pip-release-links.html
    ```

    _OR_ build from source

    ```bash
    # Build Python bindings from source
    cmake -G Ninja -B ../iree-build/ -DIREE_BUILD_PYTHON_BINDINGS=ON .
    cmake --build ../iree-build/

    # Add IREE built-from-source Python packages to PYTHONPATH
    source .env

    # Install IREE TF/TFLite Python packages
    python -m pip install integrations/tensorflow/python_projects/iree_tf
    python -m pip install integrations/tensorflow/python_projects/iree_tflite
    ```

3. Run the python test command line

    The command can be obtained from the run file. For example, if
    `iree_tfl_tests/llvmcpu_posenet_i8.run` failed,

    ```bash
    cd integrations/tensorflow/test/
    cat iree_tfl_tests/llvmcpu_posenet_i8.run

    # REQUIRES: llvmcpu
    # RUN: %PYTHON -m iree_tfl_tests.posenet_i8_test --target_backend=llvmcpu --artifacts_dir=%t

    cd python/
    python -m iree_tfl_tests.posenet_i8_test --target_backend=llvmcpu --artifacts_dir=/tmp/posenet_i8_failure
    ```

    Note that the command can only be run under
    `integrations/tensorflow/test/python` directory.

4. Extract intermediate files and use with native tools

    The test will create an `iree_input.mlir` in the temp directory specified.
    Those can then be fed into `iree-compile` (built locally to reproduce the
    error)

    ```bash
    iree-compile \
      --iree-hal-target-backends=llvm-cpu \
      --iree-input-type=stablehlo \
      iree_input.mlir
    ```
