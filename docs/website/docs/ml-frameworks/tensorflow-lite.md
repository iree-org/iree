# TensorFlow Lite Integration

IREE supports compiling and running pre-trained TFLite models.  It converts a model to [TOSA MLIR](https://mlir.llvm.org/docs/Dialects/TOSA/), then compiles it into a VM module.

## Prerequisites

Download a pre-trained TFLite model from the list of [hosted models](https://www.tensorflow.org/lite/guide/hosted_models).


Install IREE pip packages, either from pip or by
[building from source](../building-from-source/optional-features.md#building-python-bindings):

```shell
python -m pip install \
  iree-compiler-snapshot \
  iree-runtime-snapshot \
  iree-tools-tflite-snapshot \
  -f https://github.com/google/iree/releases
```

## Importing models

Fist, import the TFLite model to TOSA MLIR:

```shell
iree-import-tflite \
  sample.tflite \
  -o sample.mlir
```

Next, compile the TOSA MLIR to a VM flatbuffer, using either the command line tools or the [Python API](https://google.github.io/iree/bindings/python/):

#### Using the command-line tool

``` shell
iree-translate \
  --iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=vmvx \
  sample.mlir \
  -o sample.vmfb
```

#### Using the python API

``` python
from iree.compiler import compile_str
with open('sample.mlir') as sample_tosa_mlir:
  compiled_flatbuffer = compile_str(sample_tosa_mlir.read(), target_backends=["vmvx"])
```

The flatbuffer can then be loaded to a VM module and run through IREE's runtime.

## Samples

| Colab notebooks |  |
| -- | -- |
Text classification with TFLite and IREE | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/iree/blob/main/colab/tflite_text_classification.ipynb)

!!! todo

    [Issue#3954](https://github.com/google/iree/issues/3954): Add documentation for an Android demo using the [TFLite bindings](https://google.github.io/iree/bindings/tensorflow-lite/), once it is complete at [not-jenni/iree-android-tflite-demo](https://github.com/not-jenni/iree-android-tflite-demo).
