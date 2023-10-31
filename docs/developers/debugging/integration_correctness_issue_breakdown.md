This doc describes tips about triaging correctness issue. Feel free to reach out
to @hanhanW or ask questions on Discord if you need help or tips on triaging
correctness issue.

# Decouple the reproduce from integrations

## TF integration tests

See [instructions for reproducing failures in TF/TFLite integration tests](https://github.com/hanhanW/iree/blob/main/docs/developers/debugging/tf_integrations_test_repro.md).

For input data, they are not dumped within the flagfile. You can construct the
function inputs by looking into `log.txt`. There is an [issue](https://github.com/openxla/iree/issues/8658)
for tracking this.

## iree-samples

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

Unfortunately, the artifacts are not dumped in the runs. There is an [issue](https://github.com/openxla/iree/issues/8756)
for tracking this. A workaround can be found in the issue.

# Narrow down the repro

The model itself is big. IREE breaks a model into dispatches and launches the
kernels. The inputs and outputs could be diverged starting from one of
launches. To get a smaller reproduce, you can use [-iree-flow-trace-dispatch-tensors](https://github.com/openxla/iree/blob/main/docs/developers/developing_iree/developer_overview.md#iree-flow-trace-dispatch-tensors).
You can compare the logs between builds/backends, and get the idea about which
dispatch results in wrong outputs. The dumped inputs can be reused in a
flagfile.

Since we get the suspicious dispatch, we are able to create a test case based on
the dispatch function. The dispatch function can be derived after the
`OutlineDispatchRegions` pass. The function signatures have to be modified
manually. You'll have to put `flow.dispatch.tensor.load` variables to function
arguments, and replace `flow.dispatch.tensor.store` with `return` op.

At this stage, the reproduce is narrowed down to a single dispatch function.

Note: This only works when dispatch formation logics are identical between runs.
