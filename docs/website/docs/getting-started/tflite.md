# TFLite Integration

IREE supports compiling and running TensorFlow Lite programs stored as [TFLite
flatbuffers](https://www.tensorflow.org/lite/guide). These files can be
imported into an IREE-compatible format then compiled to a series of backends.

## Prerequisites

Install TensorFlow-Lite specific dependencies using pip:

```shell
python -m pip install \
  iree-compiler \
  iree-runtime \
  iree-tools-tflite
```

- [Command Line](./tflite-cmd.md)
- [Python API](./tflite-python.md)

## Troubleshooting

Failures during the import step usually indicate a failure to lower from 
TensorFlow Lite's operations to TOSA, the intermediate representation used by
IREE. Many TensorFlow Lite operations are not fully supported, particularly
those than use dynamic shapes. File an issue to IREE's TFLite model support
[project](https://github.com/google/iree/projects/42). 


## Additional Samples

* The
[tflitehub folder](https://github.com/google/iree-samples/tree/main/tflitehub)
in the [iree-samples repository](https://github.com/google/iree-samples)
contains test scripts to compile, run, and compare various TensorFlow Lite
models sourced from [TensorFlow Hub](https://tfhub.dev/).

* An example smoke test of the
[TensorFlow Lite C API](https://github.com/google/iree/tree/main/bindings/tflite)
is available
[here](https://github.com/google/iree/blob/main/bindings/tflite/smoke_test.cc).

| Colab notebooks |  |
| -- | -- |
Text classification with TFLite and IREE | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/iree/blob/main/colab/tflite_text_classification.ipynb)

!!! todo

    [Issue#3954](https://github.com/google/iree/issues/3954): Add documentation
    for an Android demo using the
    [Java TFLite bindings](https://github.com/google/iree/tree/main/bindings/tflite/java),
    once it is complete at
    [not-jenni/iree-android-tflite-demo](https://github.com/not-jenni/iree-android-tflite-demo).

