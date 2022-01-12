# TensorFlow Lite Integration

IREE supports compiling and running pre-trained TensorFlow Lite (TFLite)
models.  It converts a model to
[TOSA MLIR](https://mlir.llvm.org/docs/Dialects/TOSA/), then compiles it into a
VM module.

## Prerequisites

Download a pre-trained TFLite model from the list of
[hosted models](https://www.tensorflow.org/lite/guide/hosted_models), or use the
[TensorFlow Lite converter](https://www.tensorflow.org/lite/convert) to convert
a TensorFlow model to a .tflite flatbuffer.

Install IREE pip packages, either from pip or by
[building from source](../building-from-source/python-bindings-and-importers.md):

```shell
python -m pip install \
  iree-compiler \
  iree-runtime \
  iree-tools-tflite
```

!!! warning
    The TensorFlow Lite package is currently only available on Linux and macOS.
    It is not available on Windows yet (see
    [this issue](https://github.com/google/iree/issues/6417)).

## Importing models

Fist, import the TFLite model to TOSA MLIR:

```shell
iree-import-tflite \
  sample.tflite \
  -o sample.mlir
```

Next, compile the TOSA MLIR to a VM flatbuffer, using either the command line
tools or the [Python API](https://google.github.io/iree/bindings/python/):

#### Using the command-line tool

``` shell
iree-translate \
  --iree-mlir-to-vm-bytecode-module \
  --iree-input-type=tosa \
  --iree-hal-target-backends=vmvx \
  sample.mlir \
  -o sample.vmfb
```

#### Using the python API

``` python
from iree.compiler import compile_str
with open('sample.mlir') as sample_tosa_mlir:
  compiled_flatbuffer = compile_str(sample_tosa_mlir.read(),
    input_type="tosa",
    target_backends=["vmvx"],
    extra_args=["--iree-native-bindings-support=false",
      "--iree-tflite-bindings-support"])
```

!!! todo

    [Issue#5462](https://github.com/google/iree/issues/5462): Link to
    TensorFlow Lite bindings documentation once it has been written.

The flatbuffer can then be loaded to a VM module and run through IREE's runtime.

## Samples

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
